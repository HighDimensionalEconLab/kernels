from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_with_compute_dtype,
    repeated_blockdiag_metric,
)

from kernels import integrated_kernel_matrices, normalize_kernel
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
class NLLSGramSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 80
    residual_atol: float = 2e-8
    linear_solver: str = "qr"
    metric_eps: float = 1e-7


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


def multicountry_growth_kernel_nlls(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
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
    if lm_set.metric_eps <= 0.0:
        raise ValueError("metric_eps must be positive")
    if lm_set.linear_solver not in ("cholesky", "qr"):
        raise ValueError("linear_solver must be 'cholesky' or 'qr'")

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
    eps = lm_set.metric_eps

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
    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K_metric + eps * jnp.eye(n_train, dtype=jnp.float64))
    )

    def path_coordinates(theta, K_eval, K_tilde_eval, rate_coordinates=None):
        alpha_k, alpha_a, alpha_q, alpha_m, algebraic = theta
        q_offsets = algebraic[n_train : n_train + num_countries]
        marginal_wealth_offsets = algebraic[n_train + num_countries :]
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
        if rate_coordinates is None:
            interest_rate = parameters.rho - jnp.mean(dlog_marginal_wealth_dt, axis=1)
        else:
            interest_rate = parameters.rho * (1.0 + rate_coordinates)
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
        rate_coordinates=None,
    ):
        coordinates = path_coordinates(
            theta,
            K_eval,
            K_tilde_eval,
            rate_coordinates,
        )
        return multicountry_path_values(**coordinates, parameters=parameters)

    residual_count = (4 * num_countries + 1) * n_train + 1
    residual_scale = 1.0 / jnp.sqrt(float(residual_count))

    def residual_fn(theta):
        values = evaluate_coordinates(
            theta,
            K,
            K_tilde,
            theta[-1][:n_train],
        )
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
    num_algebraic_values = n_train + 2 * num_countries
    metric = metric_with_compute_dtype(
        repeated_blockdiag_metric(
            alpha_metric,
            n_train,
            num_dynamic_functions,
            additional=(
                metric_from_diagonal(
                    jnp.full(num_algebraic_values, eps, dtype=jnp.float64)
                ),
                num_algebraic_values,
            ),
        ),
        jnp.float64,
    )
    theta_0 = (
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros(num_algebraic_values, dtype=dtype),
    )
    solver = LevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        dual_solve_dtype=(jnp.float64 if lm_set.linear_solver == "cholesky" else None),
        metric=metric,
    )

    compile_start = time.perf_counter()
    compile_result = solver.solve(theta_0, max_steps=1, atol=lm_set.residual_atol)
    jax.block_until_ready(compile_result)
    compile_time = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        atol=lm_set.residual_atol,
    )
    jax.block_until_ready(result)
    solve_time = time.perf_counter() - solve_start
    theta = result.x

    def evaluate_grid(points, use_algebraic_rate=False):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        rate_coordinates = theta[-1][:n_train] if use_algebraic_rate else None
        return evaluate_coordinates(
            theta,
            K_eval,
            K_tilde_eval,
            rate_coordinates,
        )

    train_eval = evaluate_grid(train_data, use_algebraic_rate=True)
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
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    if max_asset_clearing_residual > ACCOUNTING_RESIDUAL_TOL:
        rejection_reasons.append("asset_clearing")
    if max_rate_consistency_residual > ACCOUNTING_RESIDUAL_TOL:
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
        benchmark_match = max_benchmark_error <= BENCHMARK_MATCH_TOL
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
        f"{name}_{country + 1}": alpha_metric.norm(coefficients[country]) ** 2
        - eps * (coefficients[country] @ coefficients[country])
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_total = sum(rkhs_norms.values())
    shifted_metric_norm_squared = rkhs_norm_total + eps * (
        sum(
            coefficients @ coefficients for group in theta[:4] for coefficients in group
        )
        + theta[-1] @ theta[-1]
    )

    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "rho": parameters.rho,
        "installation_elasticity": parameters.installation_elasticity,
        "num_countries": num_countries,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": "float64",
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
        "unknown_count": (4 * num_countries + 1) * n_train + 2 * num_countries,
        "dynamic_rkhs_function_count": num_dynamic_functions,
        "algebraic_rate_value_count": n_train,
        "free_initial_value_count": 2 * num_countries,
        "metric_selection_dimension": 2 * num_countries - 1,
        "selection": "minimum_shifted_rkhs_metric",
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
        "shifted_metric_norm_squared": float(shifted_metric_norm_squared),
        "solve_time": solve_time,
        "wall_time": solve_time,
        "compile_time": compile_time,
        "solver": "nlls-gram",
        "linear_solver_resolved": lm_set.linear_solver,
        "metric_eps": eps,
        "train_loss": float(result.info.loss),
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
    jsonargparse.CLI(multicountry_growth_kernel_nlls)
