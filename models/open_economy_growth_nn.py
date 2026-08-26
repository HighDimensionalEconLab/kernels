from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    DrawNNXModule,
    LevenbergMarquardt,
    MultiStart,
)

from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.open_economy_growth_shared import (
    OPEN_ECONOMY_GROWTH_CONSTANTS as CONSTANTS,
)
from models.open_economy_growth_shared import (
    DataSettings,
    StructuralParameters,
    current_account,
    investment_expenditure,
    investment_rate_from_q,
    present_value_gap,
    production,
    q_drift,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.utilities import (
    ACTIVATIONS,
    NNXResidualArgs,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    make_sampled_train_grid,
    unit_softplus,
)

RESIDUAL_NAMES = (
    "capital",
    "assets",
    "tobin_q",
    "marginal_wealth",
    "consumption_foc",
)


# Acceptance gates, CLI-reachable so a lower-precision caller can pass its
# own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 1e-2


@dataclass(frozen=True)
class SequencePathSettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    input_scale: float = 160.0


class SequencePathNN(nnx.Module):
    def __init__(
        self,
        settings: SequencePathSettings,
        *,
        k_scale: float,
        a_scale: float,
        q_scale: float,
        c_scale: float,
        mu_scale: float,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        self.input_scale = nnx.static(settings.input_scale)
        # Non-Param Variables: the output scales are data (no recompilation
        # when they change) but sit outside the differentiated Param state.
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        self.a_scale = nnx.Variable(jnp.asarray(a_scale, dtype=dtype))
        self.q_scale = nnx.Variable(jnp.asarray(q_scale, dtype=dtype))
        self.c_scale = nnx.Variable(jnp.asarray(c_scale, dtype=dtype))
        self.mu_scale = nnx.Variable(jnp.asarray(mu_scale, dtype=dtype))
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
        # Near-zero head: the drawn paths start at the declared scales
        # (k = k_0, a = 0, q = 1/eta, c and mu at the k_0 flow scale); the
        # initial conditions are imposed as residual rows.
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
        raw = self.head(h)
        # Assets are signed, so their raw head output scales linearly; the
        # other four paths are positive via unit_softplus.
        return jnp.stack(
            [
                self.k_scale[...] * unit_softplus(raw[0]),
                self.a_scale[...] * raw[1],
                self.q_scale[...] * unit_softplus(raw[2]),
                self.c_scale[...] * unit_softplus(raw[3]),
                self.mu_scale[...] * unit_softplus(raw[4]),
            ]
        )


def residual_fn(theta, args, p):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    t = args.batch["t"]
    paths, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
        t
    )
    k, a, q, c, mu = (paths[:, index] for index in range(5))
    da_dt = derivatives[:, 1]
    z = jnp.ones_like(k)
    chi = p.delta ** (1.0 - p.eta)
    values = {
        "k": k,
        "a": a,
        "q": q,
        "c": c,
        "mu": mu,
        "current_account": da_dt,
        "capital": derivatives[:, 0] / k
        - (chi * investment_rate_from_q(q, p) ** p.eta - p.delta),
        "assets": da_dt - current_account(k, a, q, c, z, p),
        "tobin_q": derivatives[:, 2] / q - q_drift(k, q, z, p) / q,
        "marginal_wealth": derivatives[:, 4] / mu - (p.r - p.r_star),
        "consumption_foc": mu * c**p.gamma - 1.0,
    }
    at_zero = model(jnp.asarray(0.0, dtype=t.dtype))
    ic = jnp.stack([at_zero[0] - p.k_0, at_zero[1] - p.a_0])
    residual = jnp.concatenate([*(values[name] for name in RESIDUAL_NAMES), ic])
    return residual / jnp.sqrt(residual.size), values


def open_economy_growth_nn(
    alpha: float = CONSTANTS.alpha,
    gamma: float = CONSTANTS.gamma,
    delta: float = CONSTANTS.delta,
    r: float = CONSTANTS.r,
    eta: float = CONSTANTS.eta,
    r_star: float = CONSTANTS.r_star,
    k_0: float = 1.0,
    a_0: float = 0.0,
    seed: int = 0,
    use_float64: bool = True,
    # The kernel driver's front-loaded collocation design on the same
    # [0, 160] span at half its density (96 vs 192 points).
    data_set: DataSettings = DataSettings(train_points=96),
    nn_set: SequencePathSettings = SequencePathSettings(),
    # atol is on the residual norm (the loss stops at atol^2).
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.5,
        damping_increase=2.0,
        max_steps=400,
        atol=1e-07,
        num_attempts=5,
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
    p = StructuralParameters.create(
        dtype=dtype,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        r=r,
        eta=eta,
        r_star=r_star,
        k_0=k_0,
        a_0=a_0,
    )
    # Normalizations from the initial conditions and structural parameters
    # alone: time-0 output for the signed asset path, the flow sustainable at
    # k_0 for consumption (mu consistent via the FOC), and the replacement-
    # investment price 1/eta for q.
    k_scale = float(k_0)
    a_scale = float(k_0**alpha)
    q_scale = 1.0 / float(eta)
    c_scale = float(k_0**alpha - delta * k_0)
    mu_scale = float(c_scale ** (-gamma))
    nn_set = replace(nn_set, input_scale=max(float(train_t[-1]), 1.0))

    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = SequencePathNN(
        nn_set,
        k_scale=k_scale,
        a_scale=a_scale,
        q_scale=q_scale,
        c_scale=c_scale,
        mu_scale=mu_scale,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = NNXResidualArgs(
        batch={"t": train_t}, graphdef=graphdef, nondiff=nondiff
    )

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
            SequencePathNN,
            nn_set,
            k_scale=k_scale,
            a_scale=a_scale,
            q_scale=q_scale,
            c_scale=c_scale,
            mu_scale=mu_scale,
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
    attempt = int(result.multi_start.attempt) + 1

    def evaluate_grid(points):
        values = residual_fn(result.x, residual_args.replace(batch={"t": points}), p)[1]
        return {
            "t": points,
            **values,
            "x": investment_expenditure(values["k"], values["q"], p),
            "output": production(values["k"], jnp.ones_like(values["k"]), p),
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

    steady = steady_state(p)
    output_scale = float(steady["output"])
    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(p)
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
        p,
    )
    tail = tail_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        p,
    )
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        train_eval["t"][-1],
        p,
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
        "parameters": p.to_dict(),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "r_star_resolved": r_star,
        "a_0_over_output_ss": a_0 / output_scale,
        "seed": seed,
        "train_T": float(train_t[-1]),
        "num_train_points": n_train,
        "k_ss": float(steady["k"]),
        "q_ss": float(steady["q"]),
        "output_ss": output_scale,
        "g_ss": float(steady["g"]),
        "c": float(train_eval["c"][0]),
        "q_0": float(test_eval["q"][0]),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_benchmark_error": max_benchmark_error,
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
    jsonargparse.CLI(open_economy_growth_nn)
