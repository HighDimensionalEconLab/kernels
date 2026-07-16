from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from jax import config
from nlls_gram import (
    LMStatus,
    UnderdeterminedLevenbergMarquardt,
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
)

from kernels import integrated_kernel_matrices, normalize_kernel
from multicountry_growth import (
    MulticountryParameters,
    multicountry_output_scale,
    multicountry_path_values,
    multicountry_steady_state,
    multicountry_tail_diagnostics,
    multicountry_tvc_diagnostics,
    two_country_full_linearization,
)
from multicountry_growth_benchmark import multicountry_growth_benchmark

config.update("jax_enable_x64", True)

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
IDENTITY_RESIDUAL_TOL = 1e-8
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
    k_0: tuple[float, float] = (1.0, 1.0),
    a_0: tuple[float] = (0.0,),
    parameters: MulticountryParameters = MulticountryParameters(),
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
):
    if parameters.num_countries != 2:
        raise ValueError("the reduced kernel system currently requires two countries")
    if len(k_0) != 2 or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain two positive capital stocks")
    if len(a_0) != 1:
        raise ValueError("a_0 must contain the first country's NFA position")
    if lm_set.metric_eps <= 0.0:
        raise ValueError("metric_eps must be positive")
    if lm_set.linear_solver not in ("cholesky", "qr"):
        raise ValueError("linear_solver must be 'cholesky' or 'qr'")

    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_train_grid(data_set)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.benchmark_T < max(data_set.test_T, float(train_data[-1])):
        raise ValueError("benchmark_T must cover the train and test horizons")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points
    )
    benchmark_grid = np.linspace(
        0.0, data_set.benchmark_T, data_set.benchmark_points
    )
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0)
    a_0_array = jnp.asarray(a_0)
    q_reference = jnp.asarray(1.0 / parameters.installation_elasticity)
    output_scale = multicountry_output_scale(parameters, k_0_array.dtype)
    eps = lm_set.metric_eps

    K, K_tilde = integrated_kernel_matrices(
        train_data,
        train_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K = (K + K.T) / 2.0
    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K + eps * jnp.eye(n_train))
    )

    def path_coordinates(theta, K_eval, K_tilde_eval):
        alpha_k_1, alpha_k_2, alpha_a_1, alpha_q_1, alpha_q_2, initial = theta
        q_1_offset, q_2_offset, log_relative_weight = initial
        log_k = jnp.stack(
            [
                jnp.log(k_0_array[0]) + K_tilde_eval @ alpha_k_1,
                jnp.log(k_0_array[1]) + K_tilde_eval @ alpha_k_2,
            ],
            axis=1,
        )
        a_independent = (
            a_0_array[0] + output_scale * (K_tilde_eval @ alpha_a_1)
        )[:, None]
        log_q = jnp.stack(
            [
                jnp.log(q_reference) + q_1_offset + K_tilde_eval @ alpha_q_1,
                jnp.log(q_reference) + q_2_offset + K_tilde_eval @ alpha_q_2,
            ],
            axis=1,
        )
        dlog_k_dt = jnp.stack(
            [K_eval @ alpha_k_1, K_eval @ alpha_k_2], axis=1
        )
        da_independent_dt = (output_scale * (K_eval @ alpha_a_1))[:, None]
        dlog_q_dt = jnp.stack(
            [K_eval @ alpha_q_1, K_eval @ alpha_q_2], axis=1
        )
        return {
            "k": jnp.exp(log_k),
            "a_independent": a_independent,
            "tobin_q": jnp.exp(log_q),
            "log_relative_weights": jnp.atleast_1d(log_relative_weight),
            "dlog_k_dt": dlog_k_dt,
            "da_independent_dt": da_independent_dt,
            "dlog_q_dt": dlog_q_dt,
        }

    def evaluate_coordinates(theta, K_eval, K_tilde_eval):
        coordinates = path_coordinates(theta, K_eval, K_tilde_eval)
        return multicountry_path_values(**coordinates, parameters=parameters)

    residual_scale = 1.0 / jnp.sqrt(5.0 * n_train)

    def residual_fn(theta):
        values = evaluate_coordinates(theta, K, K_tilde)
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["assets"].reshape(-1),
            ]
        )

    metric = blockdiag_metric(
        [
            (alpha_metric, n_train),
            (alpha_metric, n_train),
            (alpha_metric, n_train),
            (alpha_metric, n_train),
            (alpha_metric, n_train),
            (metric_from_diagonal(jnp.full(3, eps)), 3),
        ]
    )
    theta_0 = (
        jnp.zeros(n_train),
        jnp.zeros(n_train),
        jnp.zeros(n_train),
        jnp.zeros(n_train),
        jnp.zeros(n_train),
        jnp.zeros(3),
    )
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
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
    residual_names = ("capital", "tobin_q", "assets")
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )
    identity_names = (
        "resource",
        "asset_clearing",
        "interest_rate_index_gap",
    )
    max_identity_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in identity_names
    )
    max_omitted_budget_residual = float(
        jnp.max(jnp.abs(validation_eval["omitted_asset"]))
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "x", "rate_denominator")
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
    if max_omitted_budget_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("omitted_budget_residual")
    if max_identity_residual > IDENTITY_RESIDUAL_TOL:
        rejection_reasons.append("accounting_identity")
    solution_accepted = not rejection_reasons

    benchmark, benchmark_diagnostics = multicountry_growth_benchmark(
        k_0,
        a_0,
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
    a_scaled_error = jnp.abs(
        test_eval["a"] - jnp.asarray(benchmark_values["a"])
    ) / output_scale
    interest_rate_scaled_error = jnp.abs(
        test_eval["interest_rate"]
        - jnp.asarray(benchmark_values["interest_rate"])
    ) / parameters.rho
    max_benchmark_error = max(
        float(jnp.max(k_relative_error)),
        float(jnp.max(q_relative_error)),
        float(jnp.max(c_relative_error)),
        float(jnp.max(a_scaled_error)),
        float(jnp.max(interest_rate_scaled_error)),
    )
    benchmark_match = max_benchmark_error <= BENCHMARK_MATCH_TOL

    log_relative_weight = theta[-1][2:3]
    resolved_steady = multicountry_steady_state(
        log_relative_weight, parameters
    )
    tail = multicountry_tail_diagnostics(
        validation_eval, log_relative_weight, parameters
    )
    tvc = multicountry_tvc_diagnostics(validation_data, validation_eval, parameters)
    eigenvalues = jnp.linalg.eigvals(
        two_country_full_linearization(log_relative_weight[0], parameters)
    )
    stable_roots = int(jnp.sum(eigenvalues.real < -1e-8))
    unstable_roots = int(jnp.sum(eigenvalues.real > 1e-8))
    center_roots = int(jnp.sum(jnp.abs(eigenvalues.real) <= 1e-8))

    coefficient_names = ("k_1", "k_2", "a_1", "q_1", "q_2")
    coefficient_vectors = theta[:5]
    rkhs_norms = {
        name: alpha_metric.norm(coefficient) ** 2
        - eps * (coefficient @ coefficient)
        for name, coefficient in zip(
            coefficient_names, coefficient_vectors, strict=True
        )
    }
    rkhs_norm_total = sum(rkhs_norms.values())
    shifted_metric_norm_squared = rkhs_norm_total + eps * sum(
        coefficient @ coefficient for coefficient in (*coefficient_vectors, theta[-1])
    )

    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "rho": parameters.rho,
        "installation_elasticity": parameters.installation_elasticity,
        "population_1": parameters.populations[0],
        "population_2": parameters.populations[1],
        "productivity_1": parameters.productivities[0],
        "productivity_2": parameters.productivities[1],
        "k_0_1": k_0[0],
        "k_0_2": k_0[1],
        "a_0_1": a_0[0],
        "a_0_2": -parameters.populations[0] * a_0[0] / parameters.populations[1],
        "a_0_1_over_output_ss": a_0[0] / float(output_scale),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_identity_residual": max_identity_residual,
        "max_omitted_budget_residual": max_omitted_budget_residual,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_match": benchmark_match,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "interest_rate_scaled_error_max": float(
            jnp.max(interest_rate_scaled_error)
        ),
        "terminal_residual_count": 0,
        "dae_residual_count": 5 * n_train,
        "unknown_count": 5 * n_train + 3,
        "selection": "minimum_shifted_rkhs_metric",
        "initialization": "zero_coordinate",
        "log_relative_weight": float(log_relative_weight[0]),
        "relative_weight": float(jnp.exp(log_relative_weight[0])),
        "q_0_1": float(test_eval["q"][0, 0]),
        "q_0_2": float(test_eval["q"][0, 1]),
        "terminal_a_1": float(validation_eval["a"][-1, 0]),
        "terminal_a_2": float(validation_eval["a"][-1, 1]),
        "long_run_a_1": float(resolved_steady["a"][0]),
        "long_run_a_2": float(resolved_steady["a"][1]),
        "minimum_rate_denominator": float(
            jnp.min(validation_eval["rate_denominator"])
        ),
        "ex_post_k_log_gap": float(jnp.max(tail["k_log_gap"])),
        "ex_post_q_log_gap": float(jnp.max(tail["q_log_gap"])),
        "ex_post_asset_scaled_gap": float(
            jnp.max(tail["asset_scaled_gap"])
        ),
        "ex_post_interest_rate_gap": float(tail["interest_rate_gap"]),
        "ex_post_discounted_assets": float(
            jnp.max(tvc["discounted_assets"])
        ),
        "ex_post_asset_tvc": float(jnp.max(tvc["asset_tvc"])),
        "ex_post_capital_tvc": float(jnp.max(tvc["capital_tvc"])),
        "stable_roots": stable_roots,
        "unstable_roots": unstable_roots,
        "center_roots": center_roots,
        "benchmark_log_relative_weight": benchmark_diagnostics[
            "log_relative_weight"
        ],
        "benchmark_boundary_residual": benchmark_diagnostics[
            "boundary_residual_max"
        ],
        "rkhs_norm_k_1": float(rkhs_norms["k_1"]),
        "rkhs_norm_k_2": float(rkhs_norms["k_2"]),
        "rkhs_norm_a_1": float(rkhs_norms["a_1"]),
        "rkhs_norm_q_1": float(rkhs_norms["q_1"]),
        "rkhs_norm_q_2": float(rkhs_norms["q_2"]),
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

    def frame(values, points):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "marginal_wealth": np.asarray(values["marginal_wealth"]),
            "rate_denominator": np.asarray(values["rate_denominator"]),
            "resource_residual": np.asarray(values["resource"]),
            "asset_clearing_residual": np.asarray(values["asset_clearing"]),
            "omitted_asset_residual": np.asarray(values["omitted_asset"]),
            "interest_rate_index_gap": np.asarray(
                values["interest_rate_index_gap"]
            ),
        }
        for country in range(2):
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
                "tobin_q",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        data["asset_residual_1"] = np.asarray(values["assets"][:, 0])
        return pd.DataFrame(data)

    train_df = frame(train_eval, train_data)
    validation_df = frame(validation_eval, validation_data)
    test_df = frame(test_eval, test_data)
    for country in range(2):
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
    jsonargparse.CLI(multicountry_growth_kernel_nlls)
