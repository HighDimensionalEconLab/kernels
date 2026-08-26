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

from models.multicountry_growth_shared import (
    MULTICOUNTRY_GROWTH_CONSTANTS as CONSTANTS,
)
from models.multicountry_growth_shared import (
    DataSettings,
    InitialConditionSettings,
    StructuralParameters,
    draw_initial_conditions,
    investment_expenditure,
    long_run_diagnostics,
    path_residuals,
    production,
    relative_welfare_weights,
    resolved_steady_state,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
    validate_initial_conditions,
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
    "asset_clearing",
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
        num_countries: int,
        settings: SequencePathSettings,
        *,
        k_scale: tuple[float, ...],
        a_scale: float,
        q_scale: float,
        c_scale: float,
        mu_scale: tuple[float, ...],
        rate_scale: float,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.num_countries = nnx.static(num_countries)
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        self.input_scale = nnx.static(settings.input_scale)
        # Non-Param Variables: the output scales are data (no recompilation
        # when they change) but sit outside the differentiated Param state.
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        self.a_scale = nnx.Variable(jnp.asarray(a_scale, dtype=dtype))
        self.q_scale = nnx.Variable(jnp.asarray(q_scale, dtype=dtype))
        self.c_scale = nnx.Variable(jnp.asarray(c_scale, dtype=dtype))
        self.mu_scale = nnx.Variable(jnp.asarray(mu_scale, dtype=dtype))
        self.rate_scale = nnx.Variable(jnp.asarray(rate_scale, dtype=dtype))
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
        # (k = k_0, a = 0, q = 1/eta, c and mu at the world flow scale,
        # rate = r); the initial conditions are imposed as residual rows.
        self.head = nnx.Linear(
            settings.width,
            5 * num_countries + 1,
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
        blocks = raw[: 5 * self.num_countries].reshape(5, self.num_countries)
        # Assets are signed, so their raw block scales linearly; the other
        # paths and the world rate are positive via unit_softplus.
        return {
            "k": self.k_scale[...] * unit_softplus(blocks[0]),
            "a": self.a_scale[...] * blocks[1],
            "q": self.q_scale[...] * unit_softplus(blocks[2]),
            "c": self.c_scale[...] * unit_softplus(blocks[3]),
            "mu": self.mu_scale[...] * unit_softplus(blocks[4]),
            "interest_rate": self.rate_scale[...] * unit_softplus(raw[-1]),
        }


def residual_fn(theta, args, p):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    t = args.batch["t"]
    values, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
        t
    )
    residuals = path_residuals(
        values["k"],
        values["a"],
        values["q"],
        values["c"],
        values["mu"],
        p.productivities,
        values["interest_rate"],
        derivatives["k"] / values["k"],
        derivatives["a"],
        derivatives["q"] / values["q"],
        derivatives["mu"] / values["mu"],
        p,
    )
    at_zero = model(jnp.asarray(0.0, dtype=t.dtype))
    # Population demeaning is a no-op at the primal point (the driver
    # anchors cleared positions) and projects AD perturbations of a_0 onto
    # the feasible clearing set, so every direction is meaningful.
    a_0 = p.a_0 - jnp.sum(p.populations * p.a_0) / jnp.sum(p.populations)
    ic = jnp.concatenate([at_zero["k"] - p.k_0, at_zero["a"] - a_0])
    residual = jnp.concatenate(
        [
            residuals["capital"].reshape(-1),
            residuals["assets"].reshape(-1),
            residuals["tobin_q"].reshape(-1),
            residuals["marginal_wealth"].reshape(-1),
            residuals["consumption_foc"].reshape(-1),
            residuals["asset_clearing"],
            ic,
        ]
    )
    return residual / jnp.sqrt(residual.size)


def multicountry_growth_nn(
    alpha: float = CONSTANTS.alpha,
    gammas: tuple[float, ...] = CONSTANTS.gammas,
    delta: float = CONSTANTS.delta,
    r: float = CONSTANTS.r,
    eta: float = CONSTANTS.eta,
    populations: tuple[float, ...] = CONSTANTS.populations,
    productivities: tuple[float, ...] = CONSTANTS.productivities,
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    seed: int = 0,
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
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
    num_countries = len(populations)
    # Identical initial conditions are degenerate (no cross-country
    # variation), so unspecified k_0/a_0 are DRAWN from the seeded
    # initial_set distributions rather than defaulted to symmetric values.
    if k_0 is None or a_0 is None:
        drawn_k_0, drawn_a_0 = draw_initial_conditions(initial_set, populations)
        if k_0 is None:
            k_0 = drawn_k_0
        if a_0 is None:
            a_0 = drawn_a_0
    validate_initial_conditions(
        StructuralParameters.create(
            dtype=jnp.float64,
            alpha=alpha,
            gammas=gammas,
            delta=delta,
            r=r,
            eta=eta,
            populations=populations,
            productivities=productivities,
            k_0=k_0,
            a_0=a_0,
        )
    )

    train_t = make_sampled_train_grid(data_set, dtype)

    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_t = jnp.linspace(
        0.0, float(train_t[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_t)

    populations_array = jnp.asarray(populations, dtype=dtype)
    total_population = jnp.sum(populations_array)
    # Demeaning clears the sub-tolerance clearing residue the validation
    # admits; the demeaned values are what the paths anchor to, so they are
    # also what the results and hyperparameters report.
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    a_0_array = a_0_array - jnp.sum(populations_array * a_0_array) / total_population
    a_0 = tuple(float(value) for value in a_0_array)
    p = StructuralParameters.create(
        dtype=dtype,
        alpha=alpha,
        gammas=gammas,
        delta=delta,
        r=r,
        eta=eta,
        populations=populations,
        productivities=productivities,
        k_0=k_0,
        a_0=a_0_array,
    )
    # Normalizations from the initial conditions and structural parameters
    # alone: population-weighted time-0 output for the signed asset paths,
    # time-0 output net of replacement investment for consumption
    # (per-country mu consistent via the FOC), the replacement-investment
    # price 1/eta for q, and r for the world rate.
    initial_output = production(p.k_0, p.productivities, p)
    a_scale = float(jnp.sum(populations_array * initial_output) / total_population)
    c_scale = float(
        jnp.sum(populations_array * (initial_output - p.delta * p.k_0))
        / total_population
    )
    k_scale = tuple(float(value) for value in k_0)
    mu_scale = tuple(float(c_scale ** (-gamma)) for gamma in gammas)
    q_scale = 1.0 / eta
    rate_scale = r
    nn_set = replace(nn_set, input_scale=max(float(train_t[-1]), 1.0))

    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = SequencePathNN(
        num_countries,
        nn_set,
        k_scale=k_scale,
        a_scale=a_scale,
        q_scale=q_scale,
        c_scale=c_scale,
        mu_scale=mu_scale,
        rate_scale=rate_scale,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = NNXResidualArgs(
        batch={"t": train_t}, graphdef=graphdef, nondiff=nondiff
    )

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
        draw=DrawNNXModule(
            SequencePathNN,
            num_countries,
            nn_set,
            k_scale=k_scale,
            a_scale=a_scale,
            q_scale=q_scale,
            c_scale=c_scale,
            mu_scale=mu_scale,
            rate_scale=rate_scale,
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
        values, derivatives = jax.vmap(
            lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),))
        )(points)
        k, q, mu = values["k"], values["q"], values["mu"]
        residuals = path_residuals(
            k,
            values["a"],
            q,
            values["c"],
            mu,
            p.productivities,
            values["interest_rate"],
            derivatives["k"] / k,
            derivatives["a"],
            derivatives["q"] / q,
            derivatives["mu"] / mu,
            p,
        )
        x = investment_expenditure(k, q, p)
        output = production(k, p.productivities, p)
        resources = output - x
        # Goods-market clearing is implied by the budgets and the clearing
        # row (Walras), so the per-capita goods gap is a reported
        # diagnostic, not a residual.
        resource_gap = (
            jnp.sum(populations_array * (resources - values["c"]), axis=-1)
            / total_population
        )
        return {
            "t": points,
            **values,
            "x": x,
            "output": output,
            "resources": resources,
            "current_account": derivatives["a"],
            "relative_welfare_weights": relative_welfare_weights(mu),
            "resource_gap": resource_gap,
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
    max_resource_gap = float(jnp.max(jnp.abs(validation_eval["resource_gap"])))

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "mu", "x")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account", "interest_rate")
        )
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

    steady = steady_state(p, dtype)
    output_scale = float(steady["output_scale"])
    resolved_steady = resolved_steady_state(validation_eval["mu"][-1], p)
    tail = tail_diagnostics(validation_eval, p)
    tvc = tvc_diagnostics(validation_t, validation_eval, p)
    long_run = long_run_diagnostics(
        validation_t,
        validation_eval,
        a_0_array,
        p,
        check_T=data_set.long_run_check_T,
        window=data_set.long_run_window,
    )
    relative_weights = test_eval["relative_welfare_weights"][0]
    log_relative_weights = jnp.log(relative_weights)

    results = {
        "parameters": p.to_dict(),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "initial_set": asdict(initial_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "seed": seed,
        "train_T": float(train_t[-1]),
        "num_train_points": n_train,
        "output_scale": output_scale,
        "consumption_scale": float(steady["consumption_scale"]),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_resource_gap": max_resource_gap,
        "differential_residual_count": 4 * num_countries * n_train,
        "algebraic_residual_count": (num_countries + 1) * n_train,
        "dae_residual_count": (5 * num_countries + 1) * n_train,
        "free_initial_value_count": 3 * num_countries + 1,
        "long_run_c_flatness": float(long_run["c_flatness"]),
        "long_run_k_flatness": float(long_run["k_flatness"]),
        "long_run_a_flatness": float(long_run["a_flatness"]),
        "long_run_k_theory_gap": float(long_run["k_theory_gap"]),
        "long_run_mpk_equalization_gap": float(long_run["mpk_equalization_gap"]),
        "long_run_interest_rate_gap": float(long_run["interest_rate_gap"]),
        "ex_post_k_log_gap": float(jnp.max(tail["k_log_gap"])),
        "ex_post_q_log_gap": float(jnp.max(tail["q_log_gap"])),
        "ex_post_asset_scaled_gap": float(jnp.max(tail["asset_scaled_gap"])),
        "ex_post_interest_rate_gap": float(tail["interest_rate_gap"]),
        "ex_post_discounted_assets": float(jnp.max(tvc["discounted_assets"])),
        "ex_post_asset_tvc": float(jnp.max(tvc["asset_tvc"])),
        "ex_post_capital_tvc": float(jnp.max(tvc["capital_tvc"])),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "attempt": attempt,
        "solve_time": solve_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        **acceptance,
    }
    if long_run["common_gamma"]:
        results.update(
            {
                "long_run_consumption_ratio_drift": float(
                    long_run["consumption_ratio_drift"]
                ),
                "long_run_wealth_proportionality_gap": float(
                    long_run["wealth_proportionality_gap"]
                ),
                "long_run_wealth_rank_violation": float(
                    long_run["wealth_rank_violation"]
                ),
            }
        )
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"a_0_{suffix}_over_output_ss": a_0[country] / output_scale,
                f"q_0_{suffix}": float(test_eval["q"][0, country]),
                f"c_0_{suffix}": float(test_eval["c"][0, country]),
                f"m_0_{suffix}": float(test_eval["mu"][0, country]),
                f"relative_weight_{suffix}": float(relative_weights[country]),
                f"log_relative_weight_{suffix}": float(log_relative_weights[country]),
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"long_run_a_{suffix}": float(resolved_steady["a"][country]),
                f"long_run_c_{suffix}": float(resolved_steady["c"][country]),
            }
        )
        if long_run["common_gamma"]:
            results[f"initial_wealth_{suffix}"] = float(
                long_run["initial_wealth"][country]
            )

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    def frame(values, points):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "resource_gap": np.asarray(values["resource_gap"]),
            "asset_clearing": np.asarray(values["asset_clearing"]),
        }
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "mu",
                "x",
                "output",
                "resources",
                "current_account",
                "capital",
                "assets",
                "tobin_q",
                "marginal_wealth",
                "consumption_foc",
                "relative_welfare_weights",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        return pd.DataFrame(data)

    train_df = frame(train_eval, train_t)
    validation_df = frame(validation_eval, validation_t)
    test_df = frame(test_eval, test_t)

    return results, {
        "model": model,
        "train_df": train_df,
        "validation_df": validation_df,
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
    jsonargparse.CLI(multicountry_growth_nn)
