from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from jax.flatten_util import ravel_pytree
from nlls_gram import DrawNNXModule, LevenbergMarquardt, MultiStart

from models.asset_pricing_shared import (
    DataSettings,
    StructuralParameters,
    dividends,
    mu_f_array,
)
from models.utilities import (
    ACTIVATIONS,
    STATUS_TO_REASON,
    NNXResidualArgs,
    OptimizerSettings,
    Stopwatch,
    make_train_t_grid,
)


@dataclass(frozen=True)
class PricePolicySettings:
    width: int = 16
    depth: int = 2
    hidden_activation: str = "tanh"
    hidden_bias: bool = True
    output_bias: bool = True
    input_scale: float = 40.0


class PricePolicyNN(nnx.Module):
    def __init__(
        self,
        settings: PricePolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        self.input_scale = nnx.static(settings.input_scale)
        self.layers = nnx.List(
            [
                nnx.Linear(
                    1,
                    settings.width,
                    use_bias=settings.hidden_bias,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(settings.depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.width,
                    settings.width,
                    use_bias=settings.hidden_bias,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.head = nnx.Linear(
            settings.width,
            1,
            use_bias=settings.output_bias,
            kernel_init=nnx.initializers.normal(1e-3),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, t):
        h = jnp.atleast_1d(t / self.input_scale)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return self.head(h)[0]


# No-arbitrage residual dp/dt = r p - x along the dividend path.
def residual_fn(theta, args, p):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    t = args.batch["t"]
    price, dprice_dt = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(t)
    residual = dprice_dt - (p.r * price - dividends(t, p))
    return residual / jnp.sqrt(residual.size)


# Model paths and residuals on a time grid, with relative errors against the
# closed-form benchmark.
def snapshot(model, t, p):
    price, dprice_dt = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(t)
    dividend_values = dividends(t, p)
    frame = pd.DataFrame(
        {
            "t": np.asarray(t),
            "p": np.asarray(price),
            "dp_dt": np.asarray(dprice_dt),
            "dividends": np.asarray(dividend_values),
            "residual": np.asarray(dprice_dt - (p.r * price - dividend_values)),
        }
    )
    r = float(p.r)
    benchmark = np.asarray(jax.device_get(mu_f_array(t, p)))
    frame["p_benchmark"] = benchmark
    frame["p_rel_error"] = np.abs(frame["p"] - benchmark) / np.abs(benchmark)
    frame["bubble_coefficient"] = np.exp(-r * frame["t"]) * (frame["p"] - benchmark)
    frame["discounted_price"] = np.exp(-r * frame["t"]) * frame["p"]
    return frame


def asset_pricing_nn(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(test_points=101),
    nn_set: PricePolicySettings = PricePolicySettings(),
    # Library-default LM damping schedule; pinned here so the CLI reaches it.
    # atol is on the residual norm (the loss stops at atol^2).
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-3,
        damping_decrease=0.5,
        damping_increase=4.0,
        geodesic_acceptance_ratio=0.75,
        max_steps=400,
        atol=3.162277660168379e-08,
        num_attempts=5,
    ),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    train_t = make_train_t_grid(data_set, dtype)
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    nn_set = replace(nn_set, input_scale=max(float(data_set.train_T), 1.0))

    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = PricePolicyNN(
        nn_set,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = NNXResidualArgs(
        batch={"t": train_t}, graphdef=graphdef, nondiff=nondiff
    )
    p = StructuralParameters.create(dtype, r=r, c=c, g=g, x_0=x_0)

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    solve_options = dict(
        max_steps=lm_set.max_steps,
        atol=lm_set.atol,
        max_steps_is_success=False,
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=lm_set.num_attempts,
        draw=DrawNNXModule(PricePolicyNN, nn_set, dtype=dtype),
    )

    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        residual_args,
        p=p,
        multi_start=multi_start,
        **solve_options,
    )
    solve_time = timer.elapsed(result)
    model = nnx.merge(graphdef, result.x, nondiff)

    train_df = snapshot(model, train_t, p)
    test_df = snapshot(model, test_t, p)
    max_train_residual = float(train_df["residual"].abs().max())
    solution_accepted = bool(result.multi_start.accepted)

    results = {
        "r": r,
        "c": c,
        "g": g,
        "x_0": x_0,
        "seed": seed,
        # Measured from the trained parameters, not requested: a float64 leak
        # in the solve promotes them.
        "residual_dtype": str(ravel_pytree(nnx.state(model, nnx.Param))[0].dtype),
        "num_train_points": int(train_t.size),
        "train_T": float(train_t[-1]),
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(test_df["p"].iloc[0]),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "max_train_residual": max_train_residual,
        "max_test_residual": float(test_df["residual"].abs().max()),
        "p_rel_error_mean": float(test_df["p_rel_error"].mean()),
        "p_rel_error_max": float(test_df["p_rel_error"].max()),
        "bubble_coefficient_0": float(test_df["bubble_coefficient"].iloc[0]),
        "bubble_coefficient_T": float(test_df["bubble_coefficient"].iloc[-1]),
        "max_abs_bubble_coefficient": float(test_df["bubble_coefficient"].abs().max()),
        "steps": int(result.steps),
        "attempt": int(result.multi_start.attempt) + 1,
        "solve_time": solve_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted"
            if solution_accepted
            else STATUS_TO_REASON.get(int(result.status), "training_failure")
        ),
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    return results, {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": {
            "nn_set": asdict(nn_set),
            "data_set": asdict(data_set),
            "lm_set": asdict(lm_set),
            "parameters": {"r": r, "c": c, "g": g},
            "initial_conditions": {"x_0": x_0},
        },
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": p,
            "residual_args": residual_args,
            "multi_start": multi_start,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_nn)
