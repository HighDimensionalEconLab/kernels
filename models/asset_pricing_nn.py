from __future__ import annotations

from dataclasses import asdict, dataclass

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
    ToleranceSettings,
    dividends,
    mu_f_array,
)
from models.neural_networks import (
    ACTIVATIONS,
    NNXResidualArgs,
    TimeMLP,
)
from models.utilities import (
    STATUS_TO_REASON,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    gate_rejections,
    make_train_t_grid,
    value_and_grad_fwd,
)


@dataclass(frozen=True)
class PricePolicySettings:
    width: int = 16
    depth: int = 2
    hidden_activation: str = "tanh"
    hidden_bias: bool = True
    output_bias: bool = True


# No-arbitrage residual dp/dt = r p - x along the dividend path.
def residual_fn(theta, args, p):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    t = args.batch["t"]
    paths, derivatives = value_and_grad_fwd(model)(t)
    price, dprice_dt = paths[:, 0], derivatives[:, 0]
    residual = dprice_dt - (p.r * price - dividends(t, p))
    values = {"p": price, "dp_dt": dprice_dt, "residual": residual}
    return residual / jnp.sqrt(residual.size), values


# Model paths and residuals on a time grid, with relative errors against the
# closed-form benchmark.
def snapshot(theta, args, p):
    t = args.batch["t"]
    values = residual_fn(theta, args, p)[1]
    frame = pd.DataFrame(
        {
            "t": np.asarray(t),
            "p": np.asarray(values["p"]),
            "dp_dt": np.asarray(values["dp_dt"]),
            "dividends": np.asarray(dividends(t, p)),
            "residual": np.asarray(values["residual"]),
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
    data_set: DataSettings = DataSettings(),
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
    tolerance_set: ToleranceSettings = ToleranceSettings(
        train_residual=1e-6, validation_residual=1e-6
    ),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    train_t = make_train_t_grid(data_set, dtype)
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_t = jnp.linspace(
        0.0, float(train_t[-1]), data_set.validation_points, dtype=dtype
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = TimeMLP(
        1,
        width=nn_set.width,
        depth=nn_set.depth,
        activation=ACTIVATIONS[nn_set.hidden_activation],
        use_bias=nn_set.hidden_bias,
        use_output_bias=nn_set.output_bias,
        input_scale=0.025,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = NNXResidualArgs(
        batch={"t": train_t}, graphdef=graphdef, nondiff=nondiff
    )
    p = StructuralParameters.create(dtype=dtype, r=r, c=c, g=g, x_0=x_0)

    solver = LevenbergMarquardt(
        residual_fn,
        has_aux=True,
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
        draw=DrawNNXModule(
            TimeMLP,
            1,
            width=nn_set.width,
            depth=nn_set.depth,
            activation=ACTIVATIONS[nn_set.hidden_activation],
            use_bias=nn_set.hidden_bias,
            use_output_bias=nn_set.output_bias,
            input_scale=0.025,
            dtype=dtype,
        ),
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

    train_df = snapshot(result.x, residual_args, p)
    test_df = snapshot(
        result.x,
        NNXResidualArgs(batch={"t": test_t}, graphdef=graphdef, nondiff=nondiff),
        p,
    )
    validation_df = snapshot(
        result.x,
        NNXResidualArgs(batch={"t": validation_t}, graphdef=graphdef, nondiff=nondiff),
        p,
    )
    max_train_residual = float(train_df["residual"].abs().max())
    max_validation_residual = float(validation_df["residual"].abs().max())
    mean_validation_residual = float(validation_df["residual"].abs().mean())
    p_0 = float(test_df["p"].iloc[0])
    rejection_reasons = []
    if p_0 < 0.0:
        rejection_reasons.append("negative_p_0")
    rejection_reasons += gate_rejections(
        tolerance_set, max_train_residual, mean_validation_residual
    )
    solution_accepted = not rejection_reasons

    results = {
        "parameters": p.to_dict(),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": {
            "init_damping": lm_set.init_damping,
            "damping_decrease": lm_set.damping_decrease,
            "damping_increase": lm_set.damping_increase,
            "geodesic_acceleration": lm_set.geodesic_acceleration,
            "geodesic_acceptance_ratio": lm_set.geodesic_acceptance_ratio,
            "max_steps": lm_set.max_steps,
            "atol": lm_set.atol,
            "num_attempts": lm_set.num_attempts,
        },
        "tolerance_set": asdict(tolerance_set),
        "seed": seed,
        "p_0": p_0,
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "solver_status": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "mean_validation_residual": mean_validation_residual,
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
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    return results, {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
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
