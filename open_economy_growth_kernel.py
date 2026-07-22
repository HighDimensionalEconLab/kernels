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
from open_economy_growth import (
    ModelParameters,
    investment_expenditure,
    path_residuals,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from open_economy_growth_benchmark import open_economy_growth_benchmark

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
BENCHMARK_MATCH_TOL = 1e-3
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
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 50
    residual_atol: float = 2e-8
    linear_solver: str = "qr"
    metric_eps: float = 1e-7


def open_economy_growth_kernel(
    k_0: float = 1.0,
    a_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    if k_0 <= 0.0:
        raise ValueError("k_0 must be positive")
    if abs(parameters.interest_rate - parameters.rho) > 1e-12:
        raise ValueError("Stage 1 requires world_interest_rate == rho")
    if lm_set.metric_eps <= 0.0:
        raise ValueError("metric_eps must be positive")
    if lm_set.linear_solver not in ("gram_cholesky", "qr"):
        raise ValueError("linear_solver must be 'gram_cholesky' or 'qr'")

    kernel = normalize_kernel(kernel_set.kernel)
    if data_set.train_points_list is None:
        if data_set.sampling == "uniform":
            train_data = jnp.linspace(
                0.0, data_set.train_T, data_set.train_points, dtype=dtype
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
            dense_data = jnp.linspace(0.0, data_set.dense_T, dense_points, dtype=dtype)
            tail_data = jnp.linspace(
                data_set.dense_T,
                data_set.train_T,
                data_set.tail_points + 1,
                dtype=dtype,
            )[1:]
            train_data = jnp.concatenate([dense_data, tail_data])
        else:
            raise ValueError("sampling must be 'front_loaded' or 'uniform'")
    else:
        train_data = jnp.sort(jnp.asarray(data_set.train_points_list, dtype=dtype))
    if len(train_data) < 3 or not bool(jnp.all(jnp.diff(train_data) > 0.0)):
        raise ValueError("training points must be strictly increasing")
    if float(train_data[0]) != 0.0:
        raise ValueError("training points must start at zero")
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
    steady = steady_state(parameters)
    output_scale = steady["output"]
    consumption_scale = steady["g"]
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

    def path_values(theta, K_eval, K_tilde_eval):
        alpha_k, alpha_a, alpha_q, initial = theta
        q_offset, c_offset = initial
        log_k = jnp.log(k_0) + K_tilde_eval @ alpha_k
        a = a_0 + output_scale * (K_tilde_eval @ alpha_a)
        log_q = jnp.log(steady["q"]) + q_offset + K_tilde_eval @ alpha_q
        k = jnp.exp(log_k)
        tobin_q = jnp.exp(log_q)
        c = consumption_scale * jnp.exp(c_offset)
        dlog_k_dt = K_eval @ alpha_k
        da_dt = output_scale * (K_eval @ alpha_a)
        dlog_q_dt = K_eval @ alpha_q
        return k, a, tobin_q, c, dlog_k_dt, da_dt, dlog_q_dt

    residual_scale = 1.0 / jnp.sqrt(3.0 * n_train)

    def residual_fn(theta):
        k, a, tobin_q, c, dlog_k_dt, da_dt, dlog_q_dt = path_values(theta, K, K_tilde)
        residuals = path_residuals(
            k,
            a,
            tobin_q,
            c,
            jnp.ones_like(k),
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            parameters,
        )
        return residual_scale * jnp.concatenate(
            [
                residuals["capital"],
                residuals["assets"],
                residuals["tobin_q"],
            ]
        )

    # There are 3n DAE equations in 3n + 2 unknowns. No terminal equation is
    # appended: the metric supplies the two-dimensional selection.
    metric = metric_with_compute_dtype(
        repeated_blockdiag_metric(
            alpha_metric,
            n_train,
            3,
            additional=(
                metric_from_diagonal(jnp.full(2, eps, dtype=jnp.float64)),
                2,
            ),
        ),
        jnp.float64,
    )

    # The zero coordinate contains no saddle-path or present-value information:
    # it is flat in the derivative RKHS at the model's scale normalizations.
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(2, dtype=dtype),
    )

    solver = LevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        linear_solve_dtype=jnp.float64,
        metric=metric,
    )

    compile_start = time.perf_counter()
    compile_result = solver.solve(
        theta_0,
        max_steps=1,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )
    jax.block_until_ready(compile_result)
    compile_time = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
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
        k, a, tobin_q, c, dlog_k_dt, da_dt, dlog_q_dt = path_values(
            theta, K_eval, K_tilde_eval
        )
        z = jnp.ones_like(k)
        residuals = path_residuals(
            k,
            a,
            tobin_q,
            c,
            z,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            parameters,
        )
        x = investment_expenditure(k, tobin_q, parameters)
        output = production(k, z, parameters)
        current_account = da_dt
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": tobin_q,
            "c": jnp.full_like(k, c),
            "x": x,
            "output": output,
            "current_account": current_account,
            **residuals,
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    tail = tail_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        parameters,
    )
    residual_names = ("capital", "assets", "tobin_q")
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(
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
            for name in ("k", "q", "c", "x")
        )
        and all(
            jnp.all(jnp.isfinite(grid["a"]))
            for grid in (train_eval, validation_eval, test_eval)
        )
    )
    c = train_eval["c"][0]
    pv_gap = present_value_gap(
        validation_eval["t"],
        validation_eval["k"],
        validation_eval["q"],
        c,
        a_0,
        parameters,
    )
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        c,
        train_eval["t"][-1],
        parameters,
    )

    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    solution_accepted = not rejection_reasons
    benchmark_match = max_benchmark_error <= BENCHMARK_MATCH_TOL

    alpha_k, alpha_a, alpha_q, initial = theta
    q_offset, c_offset = initial
    rkhs_norm_k = alpha_metric.norm(alpha_k) ** 2 - eps * (alpha_k @ alpha_k)
    rkhs_norm_a = alpha_metric.norm(alpha_a) ** 2 - eps * (alpha_a @ alpha_a)
    rkhs_norm_q = alpha_metric.norm(alpha_q) ** 2 - eps * (alpha_q @ alpha_q)
    rkhs_norm_total = rkhs_norm_k + rkhs_norm_a + rkhs_norm_q
    shifted_metric_norm_squared = rkhs_norm_total + eps * (
        alpha_k @ alpha_k
        + alpha_a @ alpha_a
        + alpha_q @ alpha_q
        + q_offset**2
        + c_offset**2
    )

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        k, a, tobin_q, c_value, *_ = path_values(theta, K_eval, K_tilde_eval)
        return k, a, tobin_q, jnp.full_like(k, c_value)

    results = {
        **asdict(parameters),
        "world_interest_rate_resolved": parameters.interest_rate,
        "k_0": k_0,
        "a_0": a_0,
        "a_0_over_output_ss": a_0 / float(output_scale),
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": "float64",
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "k_ss": float(steady["k"]),
        "q_ss": float(steady["q"]),
        "output_ss": float(output_scale),
        "g_ss": float(consumption_scale),
        "c": float(c),
        "q_0": float(steady["q"] * jnp.exp(q_offset)),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_match": benchmark_match,
        "terminal_residual_count": 0,
        "dae_residual_count": 3 * n_train,
        "unknown_count": 3 * n_train + 2,
        "selection": "minimum_shifted_rkhs_metric",
        "initialization": "zero_coordinate",
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
        "benchmark_a_0_gap": abs(float(benchmark_diagnostics["a_0_gap"])),
        "rkhs_norm_k": float(rkhs_norm_k),
        "rkhs_norm_a": float(rkhs_norm_a),
        "rkhs_norm_q": float(rkhs_norm_q),
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
        "c_offset": float(c_offset),
    }

    frame_names = (
        "t",
        "k",
        "a",
        "q",
        "c",
        "x",
        "output",
        "current_account",
        "capital",
        "assets",
        "tobin_q",
    )
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_names}
    )
    test_df = pd.DataFrame({name: np.asarray(test_eval[name]) for name in frame_names})
    for name in ("k", "a", "q", "c", "x", "output", "current_account"):
        test_df[f"{name}_benchmark"] = np.asarray(benchmark_values[name])
    test_df["k_relative_error"] = np.asarray(k_relative_error)
    test_df["q_relative_error"] = np.asarray(q_relative_error)
    test_df["c_relative_error"] = np.asarray(c_relative_error)
    test_df["a_scaled_error"] = np.asarray(a_scaled_error)

    hyperparameters = {
        "parameters": asdict(parameters),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_kernel)
