from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    LSMR,
    QR,
    Auto,
    LMStatus,
    RidgeLevenbergMarquardt,
    identity_right_preconditioner,
    repeated_block_whitener,
    ridge_continuation,
)

from kernels import (
    integrated_kernel_matrices,
    normalize_kernel,
    rkhs_norm_squared,
)
from multicountry_growth import (
    MulticountryParameters,
    multicountry_output_scale,
    multicountry_path_values,
    multicountry_reference_marginal_wealth,
    multicountry_steady_state,
    multicountry_tail_diagnostics,
    multicountry_tvc_diagnostics,
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


@dataclass(frozen=True)
class ToleranceSettings:
    # Solution acceptance gates. The train gate bounds the max scaled DAE
    # residual on the collocation grid; at a fixed ridge the converged
    # residual floors at the O(ridge) bias, so a run at a deliberately
    # larger ridge (e.g. an iterative linear solver whose Krylov iteration
    # count scales like 1/sqrt(ridge)) must pass a ridge-aware value here
    # rather than the float64 production default.
    train_residual: float = TRAIN_RESIDUAL_TOL
    validation_residual: float = VALIDATION_RESIDUAL_TOL
    accounting: float = ACCOUNTING_RESIDUAL_TOL
    benchmark_match: float = BENCHMARK_MATCH_TOL


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
    max_steps: int = 80
    # Conjunctive stopping under the WHITENED penalty (repeated_block_whitener
    # poses the subproblem on y = L_bar theta): gtol bounds the whitened
    # stationarity ||L_bar^{-T} J'r + ridge [y_head; 0]||, residual_atol
    # additionally demands the scaled residual norm solved (the converged
    # residual floors at O(ridge * ||dual||), so residual_atol sits above
    # it). The ridge keeps the minimizer O(ridge) from the pure
    # RKHS-seminorm interpolant with the free initial values unpenalized
    # (no epsilon shift anywhere).
    residual_atol: float = 5e-8
    # "auto" (whitened cholesky) is the default: the whitened normal matrix
    # J~'J~ + ridge*I_head has its spectral floor at the ridge, so the
    # cholesky path keeps float64 accuracy at this ridge -- measured ~2x
    # faster wall-clock than the previously required "qr" at identical
    # solution quality. "qr" and "lsmr" remain valid (all paths run
    # whitened).
    linear_solver: str = "auto"
    iterative_tol: float = 1e-12
    iterative_maxiter: int = 4000
    geodesic_acceleration: bool = True
    ridge: float = 3e-12
    # The whitened calibration recipe reads
    # gtol ~ 1e-3 * ridge * sqrt(rkhs_norm_total) ~ 1e-15 at ridge 3e-12
    # (sqrt(q) ~ 0.35); the retained 1e-11 is a looser MEASURED value
    # carried over from the unwhitened noise-floor calibration -- the
    # whitened solve converges with it in ~9 steps with every quality gate
    # intact, and the extra resolution the recipe would demand was verified
    # (unwhitened) to move every quality metric only in the sixth
    # significant digit.
    gtol: float = 1e-11
    # ridge_init=None solves at the fixed ridge above; a value enables ridge
    # continuation annealed from ridge_init down to `ridge` (the floor) by
    # factors of `ridge_decrease` on inner stationarity.
    ridge_init: float | None = 1e-7
    ridge_decrease: float = 0.01
    # Relative stationarity threshold that triggers the next ridge level:
    # looser anneals faster (fewer steps per level), tighter solves each
    # level out more completely before descending.
    ridge_grad_rtol: float = 1e-3


def make_train_grid(data_set):
    if data_set.train_points_list is not None:
        train_data = jnp.sort(jnp.asarray(data_set.train_points_list))
    elif data_set.sampling == "uniform":
        train_data = jnp.linspace(0.0, data_set.train_T, data_set.train_points)
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
        dense_data = jnp.linspace(0.0, data_set.dense_T, dense_points)
        tail_data = jnp.linspace(
            data_set.dense_T,
            data_set.train_T,
            data_set.tail_points + 1,
        )[1:]
        train_data = jnp.concatenate([dense_data, tail_data])
    else:
        raise ValueError("sampling must be 'front_loaded' or 'uniform'")
    if len(train_data) < 3 or not bool(jnp.all(jnp.diff(train_data) > 0.0)):
        raise ValueError("training points must be strictly increasing")
    if float(train_data[0]) != 0.0:
        raise ValueError("training points must start at zero")
    return train_data


def multicountry_growth_kernel(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    num_countries = parameters.num_countries
    if k_0 is None:
        k_0 = (1.0,) * num_countries
    if a_0 is None:
        a_0 = (0.0,) * num_countries
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive capital stock per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one NFA position per country")
    if lm_set.iterative_tol < 0.0 or lm_set.iterative_maxiter <= 0:
        raise ValueError("iterative_tol must be nonnegative and maxiter positive")
    supported_linear_solvers = ("auto", "qr", "lsmr")
    if lm_set.linear_solver not in supported_linear_solvers:
        raise ValueError(f"linear_solver must be one of {supported_linear_solvers}")

    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_train_grid(data_set).astype(dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.benchmark_T < max(data_set.test_T, float(train_data[-1])):
        raise ValueError("benchmark_T must cover the train and test horizons")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    benchmark_grid = np.linspace(0.0, data_set.benchmark_T, data_set.benchmark_points)
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=k_0_array.dtype)
    output_scale = multicountry_output_scale(parameters, k_0_array.dtype)
    initial_asset_gap = jnp.sum(populations * a_0_array) / output_scale
    if abs(float(initial_asset_gap)) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")
    q_reference = jnp.full(
        num_countries,
        1.0 / parameters.installation_elasticity,
        dtype=k_0_array.dtype,
    )
    marginal_wealth_reference = multicountry_reference_marginal_wealth(
        parameters, k_0_array.dtype
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

    def path_coordinates(theta, K_eval, K_tilde_eval):
        alpha_k, alpha_a, alpha_q, alpha_m, alpha_r, initial = theta
        q_offsets = initial[:num_countries]
        marginal_wealth_offsets = initial[num_countries : 2 * num_countries]
        interest_rate_offset = initial[-1]
        log_k = jnp.log(k_0_array) + K_tilde_eval @ alpha_k.T
        a = a_0_array + output_scale * (K_tilde_eval @ alpha_a.T)
        log_q = jnp.log(q_reference) + q_offsets + K_tilde_eval @ alpha_q.T
        log_marginal_wealth = (
            jnp.log(marginal_wealth_reference)
            + marginal_wealth_offsets
            + K_tilde_eval @ alpha_m.T
        )
        dlog_k_dt = K_eval @ alpha_k.T
        da_dt = output_scale * (K_eval @ alpha_a.T)
        dlog_q_dt = K_eval @ alpha_q.T
        dlog_marginal_wealth_dt = K_eval @ alpha_m.T
        interest_rate = jnp.exp(
            jnp.log(parameters.rho) + interest_rate_offset + K_tilde_eval @ alpha_r
        )
        return {
            "k": jnp.exp(log_k),
            "a": a,
            "tobin_q": jnp.exp(log_q),
            "marginal_wealth": jnp.exp(log_marginal_wealth),
            "interest_rate": interest_rate,
            "dlog_k_dt": dlog_k_dt,
            "da_dt": da_dt,
            "dlog_q_dt": dlog_q_dt,
            "dlog_marginal_wealth_dt": dlog_marginal_wealth_dt,
        }

    def evaluate_coordinates(
        theta,
        K_eval,
        K_tilde_eval,
    ):
        coordinates = path_coordinates(theta, K_eval, K_tilde_eval)
        return multicountry_path_values(**coordinates, parameters=parameters)

    residual_count = (4 * num_countries + 1) * n_train + 1
    residual_scale = 1.0 / jnp.sqrt(float(residual_count))

    def residual_fn(theta):
        values = evaluate_coordinates(theta, K, K_tilde)
        # The level-clearing equation at t=0 is fixed by a_0 and has a zero
        # Jacobian. G(0) = 0 replaces that row, while G_dot(0) = 0 supplies the
        # index-consistency equation that identifies the initial algebraic rate.
        global_constraints = jnp.concatenate(
            [
                values["resource"][:1],
                values["resource_drift"][:1],
                values["asset_clearing"][1:],
            ]
        )
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
                global_constraints,
            ]
        )

    num_dynamic_functions = 4 * num_countries
    num_rkhs_functions = num_dynamic_functions + 1
    num_initial_values = 2 * num_countries + 1
    # The Whitener variant makes every linear solver work on the whitened
    # subproblem (the initial values stay unpenalized and untransformed).
    penalty = repeated_block_whitener(
        K_metric,
        repeats=num_rkhs_functions,
        zero_pad_size=num_initial_values,
    )
    theta_0 = (
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(num_initial_values, dtype=dtype),
    )
    # CLI string -> typed solver config; identity_right_preconditioner()
    # opts out of right preconditioning on the matrix-free path.
    if lm_set.linear_solver == "lsmr":
        linear_solver = LSMR(
            identity_right_preconditioner(),
            tol=lm_set.iterative_tol,
            maxiter=lm_set.iterative_maxiter,
        )
    elif lm_set.linear_solver == "qr":
        linear_solver = QR()
    else:
        linear_solver = Auto()
    if lm_set.ridge_init is None:
        initial_ridge = lm_set.ridge
        continuation_options = {}
    else:
        initial_ridge = lm_set.ridge_init
        callback, user_state0 = ridge_continuation(
            ridge_floor=lm_set.ridge,
            decrease=lm_set.ridge_decrease,
            grad_rtol=lm_set.ridge_grad_rtol,
        )
        continuation_options = {"callback": callback, "user_state": user_state0}

    solver = RidgeLevenbergMarquardt(
        residual_fn,
        penalty=penalty,
        ridge=initial_ridge,
        linear_solver=linear_solver,
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
    dynamic_residual_names = (
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth_euler",
    )
    train_global_constraints = jnp.concatenate(
        [
            train_eval["resource"][:1],
            train_eval["resource_drift"][:1],
            train_eval["asset_clearing"][1:],
        ]
    )
    max_train_residual = max(
        max(
            float(jnp.max(jnp.abs(train_eval[name]))) for name in dynamic_residual_names
        ),
        float(jnp.max(jnp.abs(train_global_constraints))),
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name])))
        for name in (*dynamic_residual_names, "resource")
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"]))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"]))
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "x", "marginal_wealth")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account", "interest_rate")
        )
    )
    rejection_reasons = []
    if int(result.status) != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    if max_validation_residual > tolerance_set.validation_residual:
        rejection_reasons.append("validation_residual")
    if max_asset_clearing_residual > tolerance_set.accounting:
        rejection_reasons.append("asset_clearing")
    if max_rate_consistency_residual > tolerance_set.accounting:
        rejection_reasons.append("rate_consistency")
    solution_accepted = not rejection_reasons

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
        a_scaled_error = (
            jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
        )
        interest_rate_scaled_error = (
            jnp.abs(
                test_eval["interest_rate"]
                - jnp.asarray(benchmark_values["interest_rate"])
            )
            / parameters.rho
        )
        max_benchmark_error = max(
            float(jnp.max(k_relative_error)),
            float(jnp.max(q_relative_error)),
            float(jnp.max(c_relative_error)),
            float(jnp.max(a_scaled_error)),
            float(jnp.max(interest_rate_scaled_error)),
        )
        benchmark_match = max_benchmark_error <= tolerance_set.benchmark_match
    else:
        k_relative_error = jnp.asarray(jnp.nan)
        q_relative_error = jnp.asarray(jnp.nan)
        c_relative_error = jnp.asarray(jnp.nan)
        a_scaled_error = jnp.asarray(jnp.nan)
        interest_rate_scaled_error = jnp.asarray(jnp.nan)
        max_benchmark_error = float("nan")
        benchmark_match = False

    resolved_steady = multicountry_steady_state(
        validation_eval["marginal_wealth"][-1], parameters
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
    }
    rkhs_norms = {
        f"{name}_{country + 1}": rkhs_norm_squared(coefficients[country], K_metric)
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_rate = rkhs_norm_squared(theta[4], K_metric)
    rkhs_norm_total = sum(rkhs_norms.values()) + rkhs_norm_rate

    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "rho": parameters.rho,
        "installation_elasticity": parameters.installation_elasticity,
        "num_countries": num_countries,
        "residual_dtype": str(jnp.dtype(dtype)),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "max_train_residual": max_train_residual,
        "initial_resource_residual": float(train_eval["resource"][0]),
        "initial_resource_drift_residual": float(train_eval["resource_drift"][0]),
        "max_validation_residual": max_validation_residual,
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_rate_consistency_residual": max_rate_consistency_residual,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_available": benchmark_available,
        "benchmark_match": benchmark_match,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "interest_rate_scaled_error_max": float(jnp.max(interest_rate_scaled_error)),
        "terminal_residual_count": 0,
        "dae_residual_count": residual_count,
        "unknown_count": (4 * num_countries + 1) * n_train + 2 * num_countries + 1,
        "dynamic_rkhs_function_count": num_dynamic_functions,
        "algebraic_rkhs_function_count": 1,
        "free_initial_value_count": num_initial_values,
        "metric_selection_dimension": 2 * num_countries,
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
        "rkhs_norm_total": float(rkhs_norm_total),
        "rkhs_norm_interest_rate": float(rkhs_norm_rate),
        "solve_time": solve_time,
        "wall_time": solve_time,
        "solver": "nlls-gram",
        "linear_solver": lm_set.linear_solver,
        "residual_atol": lm_set.residual_atol,
        "train_residual_tol": tolerance_set.train_residual,
        "iterative_tol": lm_set.iterative_tol,
        "iterative_maxiter": lm_set.iterative_maxiter,
        "iterative_preconditioner": "none",
        "geodesic_acceleration": lm_set.geodesic_acceleration,
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted" if solution_accepted else ",".join(rejection_reasons)
        ),
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
                f"q_0_{suffix}": float(test_eval["q"][0, country]),
                f"m_0_{suffix}": float(test_eval["marginal_wealth"][0, country]),
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
            "resource_drift_residual": np.asarray(values["resource_drift"]),
            "asset_clearing_residual": np.asarray(values["asset_clearing"]),
            "interest_rate_country_gap": np.asarray(
                jnp.max(
                    jnp.abs(values["interest_rate_country_gap"]),
                    axis=1,
                )
            ),
        }
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "x",
                "output",
                "resources",
                "current_account",
                "capital",
                "assets",
                "tobin_q",
                "marginal_wealth",
                "marginal_wealth_euler",
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
            for name in (
                "k",
                "a",
                "q",
                "c",
                "x",
                "output",
                "current_account",
            ):
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
    jsonargparse.CLI(multicountry_growth_kernel)
