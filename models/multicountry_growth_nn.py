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

from models.multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    MulticountryParameterValues,
    draw_initial_conditions,
    investment_expenditure,
    long_run_diagnostics,
    multicountry_parameter_values,
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
    STATUS_TO_REASON,
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
    "asset_clearing",
)


# Acceptance gates (CLI-reachable so a float32 caller can pass its own bar;
# the float32 replication driver does) plus the solve budget.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 1e-2
    max_steps: int = 400
    train_loss_threshold: float = 1e-14
    # frac_accepted measures acceptance within the restart budget, not
    # first-try robustness: a seed that stalls short of the loss threshold
    # is redrawn.
    num_attempts: int = 5


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
    # Long-run validity checks are evaluated at t ~ long_run_check_T, with
    # flatness measured against t = long_run_check_T - long_run_window.
    long_run_check_T: float = 100.0
    long_run_window: float = 10.0


@dataclass(frozen=True)
class SequencePathSettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    input_scale: float = 160.0


# Constant normalizations of the path blocks; static structure, not traced
# data, so the residual keeps them out of the p argument.
@dataclass(frozen=True)
class PathScales:
    output: float
    log_consumption: float
    log_q_ss: float


@struct.dataclass
class ResidualParameters:
    k_0: jax.Array
    a_0: jax.Array
    model: MulticountryParameterValues


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    t: jax.Array
    scales: PathScales = struct.field(pytree_node=False)


class SequencePathNN(nnx.Module):
    def __init__(
        self,
        num_countries: int,
        settings: SequencePathSettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.num_countries = nnx.static(num_countries)
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
        # Near-zero head: the drawn paths start at the model's
        # normalizations (k = k_0, a = a_0, q = 1/eta, c and mu at the
        # world consumption scale, rate = r), matching the kernel's zero
        # coordinate rather than an arbitrary point of the state space.
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
        return self.head(h)


def path_values(model, t, p, scales):
    # k_i(0) = k_0_i and a_i(0) = a_0_i hold exactly by anchoring the
    # accumulating paths on the network's own value at zero; q_i, c_i,
    # mu_i, and the rate stay free there, which is the NN counterpart of
    # the kernel's 3N + 1 unpenalized offsets.
    raw_at_zero = model(jnp.zeros((), dtype=t.dtype))
    raw, raw_dt = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(t)
    num_countries = model.num_countries
    blocks = raw[:, : 5 * num_countries].reshape(-1, 5, num_countries)
    blocks_at_zero = raw_at_zero[: 5 * num_countries].reshape(5, num_countries)
    blocks_dt = raw_dt[:, : 5 * num_countries].reshape(-1, 5, num_countries)

    # Population demeaning is a no-op at the primal point (the driver
    # anchors cleared positions) and projects AD perturbations of a_0 onto
    # the feasible clearing set, so every direction is meaningful.
    populations = p.model.populations
    a_0 = p.a_0 - jnp.sum(populations * p.a_0) / jnp.sum(populations)
    log_k = jnp.log(p.k_0) + (blocks[:, 0] - blocks_at_zero[0])
    a = a_0 + scales.output * (blocks[:, 1] - blocks_at_zero[1])
    log_q = scales.log_q_ss + blocks[:, 2]
    log_c = scales.log_consumption + blocks[:, 3]
    log_mu = -p.model.gammas * scales.log_consumption + blocks[:, 4]
    log_rate = jnp.log(p.model.r) + raw[:, -1]

    return {
        "t": t,
        "k": jnp.exp(log_k),
        "a": a,
        "q": jnp.exp(log_q),
        "c": jnp.exp(log_c),
        "mu": jnp.exp(log_mu),
        "interest_rate": jnp.exp(log_rate),
        "dlog_k_dt": blocks_dt[:, 0],
        "da_dt": scales.output * blocks_dt[:, 1],
        "dlog_q_dt": blocks_dt[:, 2],
        "dlog_mu_dt": blocks_dt[:, 4],
    }


def dae_residuals(values, p):
    return path_residuals(
        values["k"],
        values["a"],
        values["q"],
        values["c"],
        values["mu"],
        p.productivities,
        values["interest_rate"],
        values["dlog_k_dt"],
        values["da_dt"],
        values["dlog_q_dt"],
        values["dlog_mu_dt"],
        p,
    )


def residual_fn(theta, args, p):
    values = path_values(
        nnx.merge(args.graphdef, theta, args.nondiff), args.t, p, args.scales
    )
    residuals = dae_residuals(values, p.model)
    residual = jnp.concatenate(
        [
            residuals["capital"].reshape(-1),
            residuals["assets"].reshape(-1),
            residuals["tobin_q"].reshape(-1),
            residuals["marginal_wealth"].reshape(-1),
            residuals["consumption_foc"].reshape(-1),
            residuals["asset_clearing"],
        ]
    )
    return residual / jnp.sqrt(residual.size)


def multicountry_growth_nn(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    seed: int = 0,
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
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
    num_countries = parameters.num_countries
    # Identical initial conditions are degenerate (no cross-country
    # variation), so unspecified k_0/a_0 are DRAWN from the seeded
    # initial_set distributions rather than defaulted to symmetric values.
    if k_0 is None or a_0 is None:
        drawn_k_0, drawn_a_0 = draw_initial_conditions(initial_set, parameters)
        if k_0 is None:
            k_0 = drawn_k_0
        if a_0 is None:
            a_0 = drawn_a_0
    validate_initial_conditions(k_0, a_0, parameters)

    train_t = make_sampled_train_grid(data_set, dtype)

    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_t = jnp.linspace(
        0.0, float(train_t[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_t)

    populations = jnp.asarray(parameters.populations, dtype=dtype)
    total_population = jnp.sum(populations)
    steady = steady_state(parameters, dtype)
    output_scale = float(steady["output_scale"])
    scales = PathScales(
        output=output_scale,
        log_consumption=math.log(float(steady["consumption_scale"])),
        log_q_ss=math.log(1.0 / parameters.eta),
    )
    nn_set = replace(nn_set, input_scale=max(float(train_t[-1]), 1.0))
    # Demeaning clears the sub-tolerance clearing residue the validation
    # admits; the demeaned values are what the paths anchor to, so they are
    # also what the results and hyperparameters report.
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    a_0_array = a_0_array - jnp.sum(populations * a_0_array) / total_population
    a_0 = tuple(float(value) for value in a_0_array)
    residual_parameters = ResidualParameters(
        k_0=jnp.asarray(k_0, dtype=dtype),
        a_0=a_0_array,
        model=multicountry_parameter_values(parameters, dtype),
    )

    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = SequencePathNN(num_countries, nn_set, dtype=dtype, rngs=nnx.Rngs(init_key))
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
    # solve()'s atol is on the residual norm; the settings threshold is on
    # the sum of squared residuals.
    atol = math.sqrt(tolerance_set.train_loss_threshold)
    solve_options = dict(
        max_steps=tolerance_set.max_steps,
        atol=atol,
        max_steps_is_success=False,
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=tolerance_set.num_attempts,
        draw=DrawNNXModule(SequencePathNN, num_countries, nn_set, dtype=dtype),
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
        parameters_p = residual_parameters.model
        residuals = dae_residuals(values, parameters_p)
        x = investment_expenditure(values["k"], values["q"], parameters_p)
        output = production(values["k"], parameters_p.productivities, parameters_p)
        resources = output - x
        # Goods-market clearing is implied by the budgets and the clearing
        # row (Walras), so the per-capita goods gap is a reported
        # diagnostic, not a residual.
        resource_gap = (
            jnp.sum(populations * (resources - values["c"]), axis=-1) / total_population
        )
        return {
            **values,
            "x": x,
            "output": output,
            "resources": resources,
            "current_account": values["da_dt"],
            "relative_welfare_weights": relative_welfare_weights(values["mu"]),
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

    resolved_steady = resolved_steady_state(validation_eval["mu"][-1], parameters)
    tail = tail_diagnostics(validation_eval, parameters)
    tvc = tvc_diagnostics(validation_t, validation_eval, parameters)
    long_run = long_run_diagnostics(
        validation_t,
        validation_eval,
        a_0_array,
        parameters,
        check_T=data_set.long_run_check_T,
        window=data_set.long_run_window,
    )
    relative_weights = test_eval["relative_welfare_weights"][0]
    log_relative_weights = jnp.log(relative_weights)

    results = {
        "alpha": parameters.alpha,
        "delta": parameters.delta,
        "r": parameters.r,
        "eta": parameters.eta,
        "num_countries": num_countries,
        "common_gamma": bool(long_run["common_gamma"]),
        "seed": seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["capital"].dtype),
        "solver": "nlls-gram",
        "selection": "minimum_identity_metric_parameter_step",
        "initialization": "near_zero_head",
        "train_T": float(train_t[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "output_scale": output_scale,
        "consumption_scale": float(steady["consumption_scale"]),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_resource_gap": max_resource_gap,
        "terminal_residual_count": 0,
        "differential_residual_count": 4 * num_countries * n_train,
        "algebraic_residual_count": (num_countries + 1) * n_train,
        "dae_residual_count": (5 * num_countries + 1) * n_train,
        "free_initial_value_count": 3 * num_countries + 1,
        "long_run_check_T": data_set.long_run_check_T,
        "long_run_window": data_set.long_run_window,
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
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "solve_time": solve_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
        "initial_seed": initial_set.seed,
        "k_0_log_sigma": initial_set.k_0_log_sigma,
        "a_0_sigma": initial_set.a_0_sigma,
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
                "long_run_c_rank_matches_wealth": long_run["c_rank_matches_wealth"],
                "long_run_c_rank_matches_a_0": long_run["c_rank_matches_a_0"],
                "long_run_terminal_a_rank_matches_a_0": long_run[
                    "terminal_a_rank_matches_a_0"
                ],
            }
        )
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"gamma_{suffix}": parameters.gammas[country],
                f"population_{suffix}": parameters.populations[country],
                f"productivity_{suffix}": parameters.productivities[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
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

    hyperparameters = {
        "parameters": asdict(parameters),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
    }

    return results, {
        "model": model,
        "train_df": train_df,
        "validation_df": validation_df,
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
    jsonargparse.CLI(multicountry_growth_nn)
