from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from jax.flatten_util import ravel_pytree
from nlls_gram import DrawNNXModule, LevenbergMarquardt, LMStatus, MultiStart

from asset_pricing_benchmark import mu_f_array
from utilities import ACTIVATIONS

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 101


@dataclass(frozen=True)
class PricePolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    hidden_bias: bool = True
    output_bias: bool = True
    input_scale: float = 40.0


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 400
    train_loss_threshold: float = 1e-15
    num_attempts: int = 1


class PricePolicyNN(nnx.Module):
    def __init__(
        self,
        settings: PricePolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.input_scale = settings.input_scale
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


def make_train_t_grid(data_set, dtype):
    if data_set.train_points_list is None:
        values = np.linspace(
            0.0,
            data_set.train_T,
            data_set.train_points,
            dtype=np.float64,
        ).tolist()
    else:
        values = list(data_set.train_points_list)
    values = sorted(set(float(value) for value in values))
    return jnp.asarray(values, dtype=dtype)


def policy_values(model, t, r, c, g, x_0):
    price, price_derivative = jax.vmap(
        lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),))
    )(t)
    dividends = (x_0 + c / g) * jnp.exp(g * t) - c / g
    residual = price_derivative - (r * price - dividends)
    return {
        "t": t,
        "p": price,
        "dp_dt": price_derivative,
        "dividends": dividends,
        "residual": residual,
    }


def snapshot(model, t, r, c, g, x_0):
    values = jax.device_get(policy_values(model, t, r, c, g, x_0))
    frame = pd.DataFrame({key: np.asarray(value) for key, value in values.items()})
    benchmark = np.asarray(jax.device_get(mu_f_array(t, c, g, r, x_0)))
    frame["p_benchmark"] = benchmark
    frame["p_rel_error"] = np.abs(frame["p"] - benchmark) / np.abs(benchmark)
    frame["bubble_coefficient"] = np.exp(-r * frame["t"]) * (frame["p"] - benchmark)
    frame["discounted_price"] = np.exp(-r * frame["t"]) * frame["p"]
    return frame


def asset_pricing_nn(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: PricePolicySettings = PricePolicySettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
    measure_compilation: bool = True,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    train_t = make_train_t_grid(data_set, dtype)
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    nn_set = replace(nn_set, input_scale=max(float(data_set.train_T), 1.0))

    graphdef, _ = nnx.split(
        PricePolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(0)), nnx.Param
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = PricePolicyNN(
        nn_set,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    _, theta_0 = nnx.split(model, nnx.Param)

    def residual_fn(theta):
        candidate = nnx.merge(graphdef, theta)
        residual = policy_values(candidate, train_t, r, c, g, x_0)["residual"]
        return residual / jnp.sqrt(residual.size)

    solver = LevenbergMarquardt(
        residual_fn,
        linear_solve_dtype=jnp.float64,
    )
    atol = math.sqrt(stop_set.train_loss_threshold)
    draw = DrawNNXModule(PricePolicyNN, nn_set, dtype=dtype)

    compile_time = np.nan
    if measure_compilation:
        compile_start = time.perf_counter()
        compiled = solver.solve(
            theta_0,
            max_steps=1,
            atol=atol,
            multi_start=MultiStart(
                key=jax.random.key(0),
                num_starts=stop_set.num_attempts,
                draw=draw,
            ),
            max_steps_is_success=False,
        )
        jax.block_until_ready(compiled.info.loss)
        compile_time = time.perf_counter() - compile_start

    start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=stop_set.max_steps,
        atol=atol,
        multi_start=MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=draw,
        ),
        max_steps_is_success=False,
    )
    jax.block_until_ready(result.info.loss)
    train_time = time.perf_counter() - start
    model = nnx.merge(graphdef, result.x)

    train_df = snapshot(model, train_t, r, c, g, x_0)
    test_df = snapshot(model, test_t, r, c, g, x_0)
    max_train_residual = float(train_df["residual"].abs().max())
    solution_accepted = bool(result.multi_start.accepted)

    results = {
        "r": r,
        "c": c,
        "g": g,
        "x_0": x_0,
        "seed": seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "linear_solve_dtype": "float64",
        "solver": "nlls-gram",
        "ansatz": "sequential_time_level_nn",
        "metric": "identity_parameter_metric",
        "selection": "minimum_identity_metric_parameter_step",
        "terminal_residual_count": 0,
        "tvc_training_residual_count": 0,
        "benchmark_training_residual_count": 0,
        "num_train_points": int(train_t.size),
        "train_T": float(train_t[-1]),
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(test_df["p"].iloc[0]),
        "train_loss": float(result.info.loss),
        "max_train_residual": max_train_residual,
        "max_test_residual": float(test_df["residual"].abs().max()),
        "p_rel_error_mean": float(test_df["p_rel_error"].mean()),
        "p_rel_error_max": float(test_df["p_rel_error"].max()),
        "bubble_coefficient_0": float(test_df["bubble_coefficient"].iloc[0]),
        "bubble_coefficient_T": float(test_df["bubble_coefficient"].iloc[-1]),
        "max_abs_bubble_coefficient": float(test_df["bubble_coefficient"].abs().max()),
        "steps": int(result.steps),
        "attempt": int(result.multi_start.attempt) + 1,
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "train_time": train_time,
        "wall_time": train_time,
        "compile_time": compile_time,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
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
            "stop_set": asdict(stop_set),
            "parameters": {"r": r, "c": c, "g": g},
            "initial_conditions": {"x_0": x_0},
        },
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_nn)
