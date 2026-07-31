from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    QR,
    AnnealRidge,
    Cholesky,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

from kernels import (
    integrated_kernel_matrices,
    normalize_kernel,
    rkhs_norm_squared,
)
from multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    marginal_product,
    multicountry_output_scale,
    multicountry_reference_mu,
    multicountry_steady_state,
    multicountry_tail_diagnostics,
    multicountry_tvc_diagnostics,
    production,
    relative_welfare_weights,
)
from multicountry_growth_benchmark import multicountry_growth_benchmark

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
ACCOUNTING_RESIDUAL_TOL = 5e-5
INITIAL_CLEARING_TOL = 1e-10
BENCHMARK_MATCH_TOL = 2e-3
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


def multicountry_dae_path_values(
    k,
    a,
    q,
    mu,
    consumption,
    investment_rate,
    interest_rate,
    dlog_k_dt,
    da_dt,
    dlog_q_dt,
    dlog_mu_dt,
    parameters,
):
    productivities = jnp.asarray(parameters.productivities, dtype=k.dtype)
    populations = jnp.asarray(parameters.populations, dtype=k.dtype)
    rate = interest_rate[:, None]
    investment = k * investment_rate
    output = production(k, productivities, parameters)
    mpk = marginal_product(k, productivities, parameters)
    resources = output - investment
    capital_drift = k * (
        parameters.chi * investment_rate**parameters.eta - parameters.delta
    )
    budget_drift = rate * a + resources - consumption
    q_drift = (
        q
        * (
            rate
            + parameters.delta
            - (1.0 - parameters.eta) * parameters.chi * investment_rate**parameters.eta
        )
        - mpk
    )
    resource = jnp.sum(
        populations * (resources - consumption),
        axis=-1,
    )
    return {
        "k": k,
        "a": a,
        "q": q,
        "c": consumption,
        "s": investment_rate,
        "x": investment,
        "output": output,
        "resources": resources,
        "current_account": da_dt,
        "interest_rate": interest_rate,
        "marginal_wealth": mu,
        "relative_welfare_weights": relative_welfare_weights(mu),
        "capital": dlog_k_dt - capital_drift / k,
        "assets": da_dt - budget_drift,
        "tobin_q": dlog_q_dt - q_drift / q,
        "marginal_wealth_euler": dlog_mu_dt - (parameters.r - rate),
        "consumption_foc": mu * consumption**parameters.gamma - 1.0,
        "investment_foc": parameters.chi
        * parameters.eta
        * q
        * investment_rate ** (parameters.eta - 1.0)
        - 1.0,
        "resource": resource,
        "asset_clearing": jnp.sum(populations * a, axis=-1),
    }


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern52"
    ell: float = 10.0
    sigma: float = 1.0


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
    benchmark_T: float = 200.0
    benchmark_points: int = 241


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 120
    # Stopping mirrors the primal driver: gtol alone, calibrated between the
    # ridge-floor stationarity and the last intermediate continuation
    # level's. The algebraic FOC rows leave per-level stationarity lower
    # than the primal driver's (measured ~1e-12 at ridge 1e-11), so the
    # primal 1e-11 fires a level early and strands the train residual above
    # its gate at larger N; 5e-13 sits below every intermediate level's
    # stationarity and above the floor's ~1.5e-13 (measured through N=5).
    # The converged residual floor moves with N and the drawn initial
    # conditions, so a fixed positive atol is draw-fragile.
    residual_atol: float = 0.0
    gtol: float = 5e-13
    # "auto" (whitened cholesky) or "qr"; both dense paths run whitened.
    linear_solver: str = "auto"
    geodesic_acceleration: bool = True
    ridge: float = 3e-12
    # ridge_init=None solves at the fixed ridge above; a value enables ridge
    # continuation annealed from ridge_init down to `ridge` (the floor).
    ridge_init: float | None = 1e-7
    ridge_decrease: float = 0.01
    ridge_grad_rtol: float = 1e-3
    ridge_stall_rtol: float = 0.0


def make_train_grid(data_set, dtype):
    if data_set.train_points_list is not None:
        train_data = jnp.sort(jnp.asarray(data_set.train_points_list, dtype=dtype))
    elif data_set.sampling == "uniform":
        train_data = jnp.linspace(
            0.0,
            data_set.train_T,
            data_set.train_points,
            dtype=dtype,
        )
    elif data_set.sampling == "front_loaded":
        dense_points = data_set.train_points - data_set.tail_points
        if (
            dense_points < 2
            or data_set.tail_points < 1
            or not 0.0 < data_set.dense_T < data_set.train_T
        ):
            raise ValueError(
                "front_loaded sampling requires at least 2 dense points, "
                "1 tail point, and 0 < dense_T < train_T"
            )
        dense_data = jnp.linspace(
            0.0,
            data_set.dense_T,
            dense_points,
            dtype=dtype,
        )
        tail_data = jnp.linspace(
            data_set.dense_T,
            data_set.train_T,
            data_set.tail_points + 1,
            dtype=dtype,
        )[1:]
        train_data = jnp.concatenate([dense_data, tail_data])
    else:
        raise ValueError("sampling must be 'front_loaded' or 'uniform'")
    if len(train_data) < 3 or not bool(jnp.all(jnp.diff(train_data) > 0.0)):
        raise ValueError("training points must be strictly increasing")
    if float(train_data[0]) != 0.0:
        raise ValueError("training points must start at zero")
    return train_data


def multicountry_growth_kernel_dae(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
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
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive capital stock per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one NFA position per country")

    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_train_grid(data_set, dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.benchmark_T < max(data_set.test_T, float(train_data[-1])):
        raise ValueError("benchmark_T must cover the train and test horizons")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0,
        float(train_data[-1]),
        data_set.validation_points,
        dtype=dtype,
    )
    benchmark_grid = np.linspace(
        0.0,
        data_set.benchmark_T,
        data_set.benchmark_points,
    )
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    output_scale = multicountry_output_scale(parameters, dtype)
    initial_asset_gap = jnp.sum(populations * a_0_array) / output_scale
    if abs(float(initial_asset_gap)) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")

    q_reference = jnp.full(
        num_countries,
        1.0 / parameters.eta,
        dtype=dtype,
    )
    mu_reference = multicountry_reference_mu(
        parameters,
        dtype,
    )
    consumption_reference = mu_reference ** (-1.0 / parameters.gamma)
    investment_rate_reference = jnp.full(
        num_countries,
        parameters.delta,
        dtype=dtype,
    )
    metric_data = train_data.astype(jnp.float64)
    K_metric, K_tilde_metric = integrated_kernel_matrices(
        metric_data,
        metric_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K_metric = (K_metric + K_metric.T) / 2.0
    K = K_metric.astype(dtype)
    K_tilde = K_tilde_metric.astype(dtype)

    # theta contains 6N + 1 RKHS coefficient blocks followed by the finite
    # initial offsets for q, m, c, s, and the common rate.
    def path_coordinates(theta, K_eval, K_tilde_eval):
        alpha_k, alpha_a, alpha_q, alpha_m, alpha_c, alpha_s, alpha_r, initial = theta
        q_offsets = initial[:num_countries]
        mu_offsets = initial[num_countries : 2 * num_countries]
        consumption_offsets = initial[2 * num_countries : 3 * num_countries]
        investment_rate_offsets = initial[3 * num_countries : 4 * num_countries]
        interest_rate_offset = initial[-1]
        log_k = jnp.log(k_0_array) + K_tilde_eval @ alpha_k.T
        a = a_0_array + output_scale * (K_tilde_eval @ alpha_a.T)
        log_q = jnp.log(q_reference) + q_offsets + K_tilde_eval @ alpha_q.T
        log_mu = jnp.log(mu_reference) + mu_offsets + K_tilde_eval @ alpha_m.T
        log_consumption = (
            jnp.log(consumption_reference)
            + consumption_offsets
            + K_tilde_eval @ alpha_c.T
        )
        log_investment_rate = (
            jnp.log(investment_rate_reference)
            + investment_rate_offsets
            + K_tilde_eval @ alpha_s.T
        )
        log_interest_rate = (
            jnp.log(parameters.r) + interest_rate_offset + K_tilde_eval @ alpha_r
        )
        return {
            "k": jnp.exp(log_k),
            "a": a,
            "q": jnp.exp(log_q),
            "mu": jnp.exp(log_mu),
            "consumption": jnp.exp(log_consumption),
            "investment_rate": jnp.exp(log_investment_rate),
            "interest_rate": jnp.exp(log_interest_rate),
            "dlog_k_dt": K_eval @ alpha_k.T,
            "da_dt": output_scale * (K_eval @ alpha_a.T),
            "dlog_q_dt": K_eval @ alpha_q.T,
            "dlog_mu_dt": K_eval @ alpha_m.T,
        }

    def evaluate_coordinates(theta, K_eval, K_tilde_eval):
        return multicountry_dae_path_values(
            **path_coordinates(theta, K_eval, K_tilde_eval),
            parameters=parameters,
        )

    differential_residual_count = 4 * num_countries * n_train
    country_algebraic_residual_count = 2 * num_countries * n_train
    global_algebraic_residual_count = n_train
    residual_count = (
        differential_residual_count
        + country_algebraic_residual_count
        + global_algebraic_residual_count
    )
    residual_scale = 1.0 / jnp.sqrt(float(residual_count))

    def residual_fn(theta):
        values = evaluate_coordinates(theta, K, K_tilde)
        # No algebraic row contains the rate (clearing is a stock condition on
        # the states, so the DAE has index > 1); the rate path is disciplined
        # by the assets and Euler rows. The t=0 clearing row is identically
        # zero because a_0 clears, and is kept for uniformity.
        global_constraints = values["asset_clearing"]
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
                values["consumption_foc"].reshape(-1),
                values["investment_foc"].reshape(-1),
                global_constraints,
            ]
        )

    num_rkhs_functions = 6 * num_countries + 1
    num_initial_values = 4 * num_countries + 1
    # The metric takes the upper Cholesky factor F with K = F'F; the finite
    # initial offsets stay in the unpenalized free block.
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(K_metric, upper=True).astype(dtype),
        repeats=num_rkhs_functions,
    )
    coefficient_0 = jnp.zeros((num_countries, n_train), dtype=dtype)
    theta_0 = (
        coefficient_0,
        coefficient_0,
        coefficient_0,
        coefficient_0,
        coefficient_0,
        coefficient_0,
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(num_initial_values, dtype=dtype),
    )
    if lm_set.ridge_init is None:
        initial_ridge = lm_set.ridge
        continuation_options = {}
    else:
        initial_ridge = lm_set.ridge_init
        anneal = AnnealRidge(
            ridge_floor=lm_set.ridge,
            decrease=lm_set.ridge_decrease,
            grad_rtol=lm_set.ridge_grad_rtol,
            stall_rtol=lm_set.ridge_stall_rtol,
        )
        continuation_options = {
            "callback": anneal,
            "user_state": anneal.init_state(dtype),
        }
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=initial_ridge,
        linear_solver={"auto": Cholesky(), "cholesky": Cholesky(), "qr": QR()}[
            lm_set.linear_solver
        ],
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
    )

    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
        **continuation_options,
    )
    jax.block_until_ready(result)
    solve_time = time.perf_counter() - solve_start
    theta = result.x

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        return evaluate_coordinates(theta, K_eval, K_tilde_eval)

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    differential_residual_names = (
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth_euler",
    )
    country_algebraic_residual_names = (
        "consumption_foc",
        "investment_foc",
    )
    train_global_constraints = train_eval["asset_clearing"]
    max_train_residual = max(
        max(
            float(jnp.max(jnp.abs(train_eval[name])))
            for name in (
                *differential_residual_names,
                *country_algebraic_residual_names,
            )
        ),
        float(jnp.max(jnp.abs(train_global_constraints))),
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name])))
        for name in (
            *differential_residual_names,
            *country_algebraic_residual_names,
            "resource",
        )
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"]))
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "s", "x", "marginal_wealth", "interest_rate")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account")
        )
    )
    diagnostic_reasons = []
    if not finite_positive:
        diagnostic_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        diagnostic_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        diagnostic_reasons.append("validation_residual")
    if max_asset_clearing_residual > ACCOUNTING_RESIDUAL_TOL:
        diagnostic_reasons.append("asset_clearing")
    solution_accepted = int(result.status) == LMStatus.CONVERGED
    equation_diagnostics_pass = not diagnostic_reasons

    benchmark_available = num_countries == 2
    benchmark_values = None
    benchmark_diagnostics = None
    if benchmark_available:
        benchmark, benchmark_diagnostics = multicountry_growth_benchmark(
            k_0,
            (a_0[0],),
            benchmark_grid,
            parameters=parameters,
        )
        benchmark_values = benchmark(np.asarray(test_data))
        k_relative_error = jnp.abs(
            test_eval["k"] - jnp.asarray(benchmark_values["k"])
        ) / jnp.asarray(benchmark_values["k"])
        q_relative_error = jnp.abs(
            test_eval["q"] - jnp.asarray(benchmark_values["q"])
        ) / jnp.asarray(benchmark_values["q"])
        c_relative_error = jnp.abs(
            test_eval["c"] - jnp.asarray(benchmark_values["c"])
        ) / jnp.asarray(benchmark_values["c"])
        x_relative_error = jnp.abs(
            test_eval["x"] - jnp.asarray(benchmark_values["x"])
        ) / jnp.asarray(benchmark_values["x"])
        a_scaled_error = (
            jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
        )
        interest_rate_scaled_error = (
            jnp.abs(
                test_eval["interest_rate"]
                - jnp.asarray(benchmark_values["interest_rate"])
            )
            / parameters.r
        )
        max_benchmark_error = max(
            float(jnp.max(k_relative_error)),
            float(jnp.max(q_relative_error)),
            float(jnp.max(c_relative_error)),
            float(jnp.max(x_relative_error)),
            float(jnp.max(a_scaled_error)),
            float(jnp.max(interest_rate_scaled_error)),
        )
        benchmark_match = max_benchmark_error <= BENCHMARK_MATCH_TOL
    else:
        k_relative_error = jnp.asarray(jnp.nan)
        q_relative_error = jnp.asarray(jnp.nan)
        c_relative_error = jnp.asarray(jnp.nan)
        x_relative_error = jnp.asarray(jnp.nan)
        a_scaled_error = jnp.asarray(jnp.nan)
        interest_rate_scaled_error = jnp.asarray(jnp.nan)
        max_benchmark_error = float("nan")
        benchmark_match = False

    resolved_steady = multicountry_steady_state(
        validation_eval["marginal_wealth"][-1],
        parameters,
    )
    tail = multicountry_tail_diagnostics(validation_eval, parameters)
    tvc = multicountry_tvc_diagnostics(validation_data, validation_eval, parameters)
    relative_weights = test_eval["relative_welfare_weights"][0]
    log_relative_weights = jnp.log(relative_weights)
    coefficient_groups = {
        "k": theta[0],
        "a": theta[1],
        "q": theta[2],
        "m": theta[3],
        "c": theta[4],
        "s": theta[5],
    }
    rkhs_norms = {
        f"{name}_{country + 1}": rkhs_norm_squared(coefficients[country], K_metric)
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_r = rkhs_norm_squared(theta[6], K_metric)
    rkhs_norm_total = sum(rkhs_norms.values()) + rkhs_norm_r

    initial = theta[7]
    q_offsets = initial[:num_countries]
    mu_offsets = initial[num_countries : 2 * num_countries]
    consumption_offsets = initial[2 * num_countries : 3 * num_countries]
    investment_rate_offsets = initial[3 * num_countries : 4 * num_countries]
    results = {
        **asdict(parameters),
        "num_countries": num_countries,
        "residual_dtype": str(jnp.dtype(dtype)),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_consumption_foc_residual": float(
            jnp.max(jnp.abs(validation_eval["consumption_foc"]))
        ),
        "max_investment_foc_residual": float(
            jnp.max(jnp.abs(validation_eval["investment_foc"]))
        ),
        "max_benchmark_error": max_benchmark_error,
        "benchmark_available": benchmark_available,
        "benchmark_match": benchmark_match,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "x_relative_error_max": float(jnp.max(x_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "interest_rate_scaled_error_max": float(jnp.max(interest_rate_scaled_error)),
        "terminal_residual_count": 0,
        "differential_residual_count": differential_residual_count,
        "country_algebraic_residual_count": country_algebraic_residual_count,
        "global_algebraic_residual_count": global_algebraic_residual_count,
        "dae_residual_count": residual_count,
        "unknown_count": num_rkhs_functions * n_train + num_initial_values,
        "dynamic_rkhs_function_count": 4 * num_countries,
        "country_algebraic_rkhs_function_count": 2 * num_countries,
        "common_algebraic_rkhs_function_count": 1,
        "rkhs_function_count": num_rkhs_functions,
        "free_initial_value_count": num_initial_values,
        "metric_selection_dimension": 4 * num_countries,
        "selection": "minimum_rkhs_norm_ridge",
        "initialization": "zero_coordinate",
        "global_closure": "explicit_assets_and_algebraic_clearing",
        "ex_post_k_log_gap": float(jnp.max(tail["k_log_gap"])),
        "ex_post_q_log_gap": float(jnp.max(tail["q_log_gap"])),
        "ex_post_asset_scaled_gap": float(jnp.max(tail["asset_scaled_gap"])),
        "ex_post_interest_rate_gap": float(tail["interest_rate_gap"]),
        "ex_post_discounted_assets": float(jnp.max(tvc["discounted_assets"])),
        "ex_post_asset_tvc": float(jnp.max(tvc["asset_tvc"])),
        "ex_post_capital_tvc": float(jnp.max(tvc["capital_tvc"])),
        "rkhs_norm_r": float(rkhs_norm_r),
        "rkhs_norm_total": float(rkhs_norm_total),
        "solve_time": solve_time,
        "wall_time": solve_time,
        "solver": "nlls-gram",
        "linear_solver": lm_set.linear_solver,
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "final_ridge": float(result.lm_state.ridge),
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "solution_accepted": solution_accepted,
        "rejection_reason": "accepted" if solution_accepted else "solver_status",
        "equation_diagnostics_pass": equation_diagnostics_pass,
        "equation_diagnostic_reason": (
            "passed" if equation_diagnostics_pass else ",".join(diagnostic_reasons)
        ),
        "interest_rate_0": float(parameters.r * jnp.exp(initial[-1])),
    }
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"population_{suffix}": parameters.populations[country],
                f"productivity_{suffix}": parameters.productivities[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
                f"a_0_{suffix}_over_output_ss": a_0[country] / float(output_scale),
                f"q_0_{suffix}": float(
                    q_reference[country] * jnp.exp(q_offsets[country])
                ),
                f"m_0_{suffix}": float(
                    mu_reference[country] * jnp.exp(mu_offsets[country])
                ),
                f"c_0_{suffix}": float(
                    consumption_reference[country]
                    * jnp.exp(consumption_offsets[country])
                ),
                f"s_0_{suffix}": float(
                    investment_rate_reference[country]
                    * jnp.exp(investment_rate_offsets[country])
                ),
                f"relative_weight_{suffix}": float(relative_weights[country]),
                f"log_relative_weight_{suffix}": float(log_relative_weights[country]),
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"long_run_a_{suffix}": float(resolved_steady["a"][country]),
                **{
                    f"rkhs_norm_{name}_{suffix}": float(rkhs_norms[f"{name}_{suffix}"])
                    for name in coefficient_groups
                },
            }
        )
    if benchmark_available:
        results.update(
            {
                "log_relative_weight": float(log_relative_weights[1]),
                "relative_weight": float(relative_weights[1]),
                "benchmark_log_relative_weight": benchmark_diagnostics[
                    "log_relative_weight"
                ],
                "benchmark_boundary_residual": benchmark_diagnostics[
                    "boundary_residual_max"
                ],
                "stable_roots": benchmark_diagnostics["stable_roots"],
                "unstable_roots": benchmark_diagnostics["unstable_roots"],
                "center_roots": benchmark_diagnostics["center_roots"],
                "minimum_rate_denominator": benchmark_diagnostics[
                    "minimum_rate_denominator"
                ],
            }
        )

    def frame(values, points):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "resource_residual": np.asarray(values["resource"]),
            "asset_clearing_residual": np.asarray(values["asset_clearing"]),
        }
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "s",
                "x",
                "output",
                "resources",
                "current_account",
                "capital",
                "assets",
                "tobin_q",
                "marginal_wealth",
                "marginal_wealth_euler",
                "consumption_foc",
                "investment_foc",
                "relative_welfare_weights",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        return pd.DataFrame(data)

    train_df = frame(train_eval, train_data)
    validation_df = frame(validation_eval, validation_data)
    test_df = frame(test_eval, test_data)
    if benchmark_available:
        for country in range(num_countries):
            suffix = country + 1
            for name in ("k", "a", "q", "c", "x", "output", "current_account"):
                test_df[f"{name}_{suffix}_benchmark"] = benchmark_values[name][
                    :, country
                ]
        test_df["interest_rate_benchmark"] = benchmark_values["interest_rate"]

    hyperparameters = {
        "parameters": asdict(parameters),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
    }

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        return evaluate_coordinates(theta, K_eval, K_tilde_eval)

    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_growth_kernel_dae)
