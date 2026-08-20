from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    DrawNNXModule,
    LevenbergMarquardt,
    MultiStart,
)

from models.open_economy_growth import (
    ModelParameters,
    investment_expenditure,
    path_residuals,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.utilities import (
    ACTIVATIONS,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    make_sampled_train_grid,
)

RESIDUAL_NAMES = (
    "capital",
    "assets",
    "tobin_q",
    "marginal_wealth",
    "consumption_foc",
)


# Acceptance gates (CLI-reachable so a float32 caller can pass its own bar;
# the float32 replication driver does) plus the solve budget.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 1e-2
    benchmark_match: float = 1e-3
    max_steps: int = 400
    train_loss_threshold: float = 1e-14
    # frac_accepted measures acceptance within the restart budget, not first-try
    # robustness: a seed that stalls short of the loss threshold is redrawn.
    num_attempts: int = 5


# The kernel driver's collocation grid, reused verbatim so the sequence-space
# NN and the sequence-space kernel are solved on identical dates.
@dataclass(frozen=True)
class DataSettings:
    train_T: float = 160.0
    train_points: int = 96
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 15
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401


@dataclass(frozen=True)
class SequencePathSettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    input_scale: float = 160.0


# Constant normalizations of the five path blocks; static structure, not
# traced data, so the residual keeps them out of the p argument.
@dataclass(frozen=True)
class PathScales:
    output: float
    log_consumption: float
    log_q_ss: float


@struct.dataclass
class DynamicModelParameters:
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    r: jax.Array
    eta: jax.Array
    r_star: jax.Array
    chi: jax.Array

    @classmethod
    def from_model(cls, p, dtype):
        return cls(
            alpha=jnp.asarray(p.alpha, dtype=dtype),
            gamma=jnp.asarray(p.gamma, dtype=dtype),
            delta=jnp.asarray(p.delta, dtype=dtype),
            r=jnp.asarray(p.r, dtype=dtype),
            eta=jnp.asarray(p.eta, dtype=dtype),
            r_star=jnp.asarray(p.r_star, dtype=dtype),
            chi=jnp.asarray(p.delta ** (1.0 - p.eta), dtype=dtype),
        )


@struct.dataclass
class ResidualParameters:
    k_0: jax.Array
    a_0: jax.Array
    model: DynamicModelParameters


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    t: jax.Array
    scales: PathScales = struct.field(pytree_node=False)


class SequencePathNN(nnx.Module):
    def __init__(
        self,
        settings: SequencePathSettings,
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
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        # Near-zero head: the drawn paths start at the model's normalizations
        # (k = k_0, a = a_0, q = q_ss, c = g_ss), matching the kernel's zero
        # coordinate rather than an arbitrary point of the state space.
        self.head = nnx.Linear(
            settings.width,
            5,
            kernel_init=nnx.initializers.normal(1e-3),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, t):
        h = jnp.atleast_1d(t / self.input_scale)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return self.head(h)


def path_values(model, t, p, scales):
    # k(0) = k_0 and a(0) = a_0 hold exactly by anchoring the two accumulating
    # paths on the network's own value at zero; c, mu, and q stay free there,
    # which is the NN counterpart of the kernel's three unpenalized offsets.
    raw_at_zero = model(jnp.zeros((), dtype=t.dtype))
    raw, raw_dt = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(t)

    log_k = jnp.log(p.k_0) + (raw[:, 0] - raw_at_zero[0])
    a = p.a_0 + scales.output * (raw[:, 1] - raw_at_zero[1])
    log_q = scales.log_q_ss + raw[:, 2]
    log_c = scales.log_consumption + raw[:, 3]
    log_mu = -p.model.gamma * scales.log_consumption + raw[:, 4]

    return {
        "t": t,
        "k": jnp.exp(log_k),
        "a": a,
        "q": jnp.exp(log_q),
        "c": jnp.exp(log_c),
        "mu": jnp.exp(log_mu),
        "dlog_k_dt": raw_dt[:, 0],
        "da_dt": scales.output * raw_dt[:, 1],
        "dlog_q_dt": raw_dt[:, 2],
        "dlog_marginal_wealth_dt": raw_dt[:, 4],
    }


def dae_residuals(values, p):
    return path_residuals(
        values["k"],
        values["a"],
        values["q"],
        values["c"],
        values["mu"],
        jnp.ones_like(values["k"]),
        values["dlog_k_dt"],
        values["da_dt"],
        values["dlog_q_dt"],
        values["dlog_marginal_wealth_dt"],
        p,
    )


def residual_fn(theta, args, p):
    values = path_values(
        nnx.merge(args.graphdef, theta, args.nondiff), args.t, p, args.scales
    )
    residuals = dae_residuals(values, p.model)
    residual = jnp.concatenate([residuals[name] for name in RESIDUAL_NAMES])
    return residual / jnp.sqrt(residual.size)


def open_economy_growth_nn(
    k_0: float = 1.0,
    a_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: SequencePathSettings = SequencePathSettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.5,
        damping_increase=2.0,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32

    train_t = make_sampled_train_grid(data_set, dtype)

    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_t = jnp.linspace(
        0.0, float(train_t[-1]), data_set.validation_points, dtype=dtype
    )
    steady = steady_state(parameters)
    output_scale = float(steady["output"])
    scales = PathScales(
        output=output_scale,
        log_consumption=math.log(float(steady["g"])),
        log_q_ss=math.log(float(steady["q"])),
    )
    nn_set = replace(nn_set, input_scale=max(float(train_t[-1]), 1.0))
    residual_parameters = ResidualParameters(
        k_0=jnp.asarray(k_0, dtype=dtype),
        a_0=jnp.asarray(a_0, dtype=dtype),
        model=DynamicModelParameters.from_model(parameters, dtype),
    )

    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = SequencePathNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key))
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = ResidualArgs(
        graphdef=graphdef, nondiff=nondiff, t=train_t, scales=scales
    )

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    # solve()'s atol is on the residual norm; the settings threshold is on the
    # sum of squared residuals.
    atol = math.sqrt(tolerance_set.train_loss_threshold)
    solve_options = dict(
        max_steps=tolerance_set.max_steps,
        atol=atol,
        max_steps_is_success=False,
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=tolerance_set.num_attempts,
        draw=DrawNNXModule(SequencePathNN, nn_set, dtype=dtype),
    )

    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        residual_args,
        p=residual_parameters,
        multi_start=multi_start,
        **solve_options,
    )
    solve_time = timer.elapsed(result)
    model = nnx.merge(graphdef, result.x, nondiff)
    attempt = int(result.multi_start.attempt) + 1

    def evaluate_grid(points):
        values = path_values(model, points, residual_parameters, scales)
        residuals = dae_residuals(values, residual_parameters.model)
        return {
            **values,
            "x": investment_expenditure(values["k"], values["q"], parameters),
            "output": production(values["k"], jnp.ones_like(values["k"]), parameters),
            "current_account": values["da_dt"],
            **residuals,
        }

    train_eval = evaluate_grid(train_t)
    validation_eval = evaluate_grid(validation_t)
    test_eval = evaluate_grid(test_t)
    jax.block_until_ready((train_eval, validation_eval, test_eval))

    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in RESIDUAL_NAMES
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in RESIDUAL_NAMES
    )

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(
        k_0, a_0, parameters=parameters
    )
    benchmark_values = benchmark(np.asarray(test_t))
    k_relative_error = jnp.abs(
        test_eval["k"] - jnp.asarray(benchmark_values["k"])
    ) / jnp.asarray(benchmark_values["k"])
    q_relative_error = jnp.abs(
        test_eval["q"] - jnp.asarray(benchmark_values["q"])
    ) / jnp.asarray(benchmark_values["q"])
    c_relative_error = jnp.abs(
        test_eval["c"] - jnp.asarray(benchmark_values["c"])
    ) / jnp.asarray(benchmark_values["c"])
    a_scaled_error = (
        jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
    )
    max_benchmark_error = max(
        float(jnp.max(k_relative_error)),
        float(jnp.max(q_relative_error)),
        float(jnp.max(c_relative_error)),
        float(jnp.max(a_scaled_error)),
    )

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name])) and jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "mu", "x")
        )
        and all(
            jnp.all(jnp.isfinite(grid["a"]))
            for grid in (train_eval, validation_eval, test_eval)
        )
    )
    pv_gap = present_value_gap(
        validation_eval["t"],
        validation_eval["k"],
        validation_eval["q"],
        validation_eval["c"],
        a_0,
        parameters,
    )
    tail = tail_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        parameters,
    )
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        train_eval["t"][-1],
        parameters,
    )

    rejection_reasons = []
    if not bool(result.multi_start.accepted):
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    if max_validation_residual > tolerance_set.validation_residual:
        rejection_reasons.append("validation_residual")
    acceptance = acceptance_fields(rejection_reasons)
    solution_accepted = acceptance["solution_accepted"]

    n_train = len(train_t)
    results = {
        **asdict(parameters),
        "r_star_resolved": parameters.r_star,
        "k_0": k_0,
        "a_0": a_0,
        "a_0_over_output_ss": a_0 / output_scale,
        "seed": seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["capital"].dtype),
        "train_T": float(train_t[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "k_ss": float(steady["k"]),
        "q_ss": float(steady["q"]),
        "output_ss": output_scale,
        "g_ss": float(steady["g"]),
        "c": float(train_eval["c"][0]),
        "q_0": float(test_eval["q"][0]),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_benchmark_error": max_benchmark_error,
        "terminal_residual_count": 0,
        "differential_residual_count": 4 * n_train,
        "algebraic_residual_count": n_train,
        "dae_residual_count": 5 * n_train,
        "free_initial_value_count": 3,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "ex_post_present_value_gap": float(jnp.abs(pv_gap)),
        "ex_post_current_account": float(jnp.abs(train_eval["current_account"][-1])),
        "ex_post_stable_gap": abs(float(tail["stable"])),
        "ex_post_present_value_tail_gap": abs(float(tail["present_value"])),
        "ex_post_discounted_assets": float(tvc["discounted_assets"]),
        "ex_post_asset_tvc": float(tvc["asset_tvc"]),
        "ex_post_capital_tvc": float(tvc["capital_tvc"]),
        "benchmark_present_value_residual": abs(
            float(benchmark_diagnostics["present_value_residual"])
        ),
        "max_marginal_wealth_residual": float(
            jnp.max(jnp.abs(train_eval["marginal_wealth"]))
        ),
        "max_consumption_foc_residual": float(
            jnp.max(jnp.abs(train_eval["consumption_foc"]))
        ),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "attempt": attempt,
        "solve_time": solve_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
        **acceptance,
    }

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    frame_names = (
        "t",
        "k",
        "a",
        "q",
        "c",
        "mu",
        "x",
        "output",
        "current_account",
        *RESIDUAL_NAMES,
    )
    train_df = pd.DataFrame(
        {name: np.asarray(jax.device_get(train_eval[name])) for name in frame_names}
    )
    test_df = pd.DataFrame(
        {name: np.asarray(jax.device_get(test_eval[name])) for name in frame_names}
    )
    for name in ("k", "a", "q", "c", "x", "output", "current_account"):
        test_df[f"{name}_benchmark"] = np.asarray(benchmark_values[name])
    test_df["k_relative_error"] = np.asarray(k_relative_error)
    test_df["q_relative_error"] = np.asarray(q_relative_error)
    test_df["c_relative_error"] = np.asarray(c_relative_error)
    test_df["a_scaled_error"] = np.asarray(a_scaled_error)

    hyperparameters = {
        "parameters": asdict(parameters),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
    }
    return results, {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": residual_parameters,
            "residual_args": residual_args,
            "multi_start": multi_start,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_nn)
