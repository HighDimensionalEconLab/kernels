from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import unopy

from models.kernels import integrated_kernel_matrices, normalize_kernel
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
from models.open_economy_growth_kernel import DataSettings, KernelSettings

ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 1e-2


@dataclass(frozen=True)
class SolverSettings:
    preset: str = "ipopt"
    max_iterations: int | None = None
    time_limit: float | None = None
    primal_tolerance: float = 1e-4
    dual_tolerance: float = 1e-4


def open_economy_growth_kernel_uno(
    k_0: float = 1.0,
    a_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    kernel_set: KernelSettings = KernelSettings(ell=20.0),
    data_set: DataSettings = DataSettings(),
    solver_set: SolverSettings = SolverSettings(),
    verbose: bool = False,
):
    # IPOPT works in float64 C arrays, so this solver is float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    kernel = normalize_kernel(kernel_set.kernel)
    if data_set.train_points_list is None:
        if data_set.sampling == "uniform":
            train_data = jnp.linspace(
                0.0, data_set.train_T, data_set.train_points, dtype=jnp.float64
            )
        elif data_set.sampling == "front_loaded":
            dense_points = data_set.train_points - data_set.tail_points
            dense_data = jnp.linspace(
                0.0, data_set.dense_T, dense_points, dtype=jnp.float64
            )
            tail_data = jnp.linspace(
                data_set.dense_T,
                data_set.train_T,
                data_set.tail_points + 1,
                dtype=jnp.float64,
            )[1:]
            train_data = jnp.concatenate([dense_data, tail_data])
        else:
            raise ValueError("sampling must be 'front_loaded' or 'uniform'")
    else:
        train_data = jnp.sort(
            jnp.asarray(data_set.train_points_list, dtype=jnp.float64)
        )
    if len(train_data) < 3 or not bool(jnp.all(jnp.diff(train_data) > 0.0)):
        raise ValueError("training points must be strictly increasing")
    if float(train_data[0]) != 0.0:
        raise ValueError("training points must start at zero")

    test_data = jnp.linspace(
        0.0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=jnp.float64
    )
    n_train = len(train_data)
    steady = steady_state(parameters)
    output_scale = steady["output"]
    consumption_scale = steady["g"]

    K, K_tilde = integrated_kernel_matrices(
        train_data,
        train_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K = (K + K.T) / 2.0

    n_variables = 5 * n_train + 3
    n_constraints = 5 * n_train

    def unpack(x):
        alpha_mu = x[:n_train]
        alpha_c = x[n_train : 2 * n_train]
        alpha_k = x[2 * n_train : 3 * n_train]
        alpha_a = x[3 * n_train : 4 * n_train]
        alpha_q = x[4 * n_train : 5 * n_train]
        offsets = x[5 * n_train :]
        return alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, offsets

    def path_values(x, K_eval, K_tilde_eval):
        alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, offsets = unpack(x)
        c_offset, mu_offset, q_offset = offsets
        log_mu = (
            -parameters.gamma * jnp.log(consumption_scale)
            + mu_offset
            + K_tilde_eval @ alpha_mu
        )
        log_c = jnp.log(consumption_scale) + c_offset + K_tilde_eval @ alpha_c
        log_k = jnp.log(k_0) + K_tilde_eval @ alpha_k
        a = a_0 + output_scale * (K_tilde_eval @ alpha_a)
        log_q = jnp.log(steady["q"]) + q_offset + K_tilde_eval @ alpha_q
        mu = jnp.exp(log_mu)
        c = jnp.exp(log_c)
        k = jnp.exp(log_k)
        q = jnp.exp(log_q)
        dlog_mu_dt = K_eval @ alpha_mu
        dlog_k_dt = K_eval @ alpha_k
        da_dt = output_scale * (K_eval @ alpha_a)
        dlog_q_dt = K_eval @ alpha_q
        return (
            k,
            a,
            q,
            c,
            mu,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
        )

    def objective(x):
        alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, _ = unpack(x)
        return (
            alpha_mu @ K @ alpha_mu
            + alpha_c @ K @ alpha_c
            + alpha_k @ K @ alpha_k
            + alpha_a @ K @ alpha_a
            + alpha_q @ K @ alpha_q
        )

    residual_scale = 1.0 / jnp.sqrt(5.0 * n_train)

    def constraints(x):
        (
            k,
            a,
            q,
            c,
            mu,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
        ) = path_values(x, K, K_tilde)
        residuals = path_residuals(
            k,
            a,
            q,
            c,
            mu,
            jnp.ones_like(k),
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
            parameters,
        )
        return residual_scale * jnp.concatenate(
            [
                residuals["capital"],
                residuals["assets"],
                residuals["tobin_q"],
                residuals["marginal_wealth"],
                residuals["consumption_foc"],
            ]
        )

    def lagrangian(x, objective_multiplier, multipliers):
        return objective_multiplier * objective(x) + jnp.dot(
            multipliers, constraints(x)
        )

    objective_value = jax.jit(objective)
    objective_gradient = jax.jit(jax.grad(objective))
    constraint_values = jax.jit(constraints)
    constraint_jacobian = jax.jit(jax.jacfwd(constraints))
    lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial_jax = jnp.asarray(x_initial)
    zero_multipliers = jnp.zeros(n_constraints, dtype=jnp.float64)
    jax.block_until_ready(objective_value(x_initial_jax))
    jax.block_until_ready(objective_gradient(x_initial_jax))
    jax.block_until_ready(constraint_values(x_initial_jax))
    jax.block_until_ready(constraint_jacobian(x_initial_jax))
    jax.block_until_ready(lagrangian_hessian(x_initial_jax, 1.0, zero_multipliers))

    variable_lower_bounds = np.full(n_variables, -np.inf, dtype=np.float64)
    variable_upper_bounds = np.full(n_variables, np.inf, dtype=np.float64)
    constraint_lower_bounds = np.zeros(n_constraints, dtype=np.float64)
    constraint_upper_bounds = np.zeros(n_constraints, dtype=np.float64)
    model = unopy.Model(
        unopy.PROBLEM_NONLINEAR,
        n_variables,
        variable_lower_bounds,
        variable_upper_bounds,
        unopy.ZERO_BASED_INDEXING,
    )

    def objective_callback(x):
        return float(objective_value(jnp.asarray(x)))

    def objective_gradient_callback(x, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(x)))

    model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(x, constraint_output):
        constraint_output[:] = np.asarray(constraint_values(jnp.asarray(x)))

    def jacobian_callback(x, jacobian_output):
        jacobian_output[:] = np.asarray(constraint_jacobian(jnp.asarray(x))).reshape(-1)

    model.set_constraints(
        n_constraints,
        constraints_callback,
        constraint_lower_bounds,
        constraint_upper_bounds,
        len(jacobian_rows),
        jacobian_rows,
        jacobian_columns,
        jacobian_callback,
    )

    hessian_rows, hessian_columns = np.tril_indices(n_variables)
    hessian_rows = hessian_rows.astype(np.int32)
    hessian_columns = hessian_columns.astype(np.int32)

    def hessian_callback(x, objective_multiplier, multipliers, hessian_output):
        hessian = np.asarray(
            lagrangian_hessian(
                jnp.asarray(x),
                float(objective_multiplier),
                jnp.asarray(multipliers),
            )
        )
        hessian_output[:] = hessian[hessian_rows, hessian_columns]

    model.set_lagrangian_hessian(
        len(hessian_rows),
        unopy.LOWER_TRIANGLE,
        hessian_rows,
        hessian_columns,
        hessian_callback,
    )
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    model.set_initial_primal_iterate(x_initial)

    solver = unopy.UnoSolver()
    solver.set_preset(solver_set.preset)
    solver.set_option("primal_tolerance", solver_set.primal_tolerance)
    solver.set_option("dual_tolerance", solver_set.dual_tolerance)
    if not verbose:
        solver.set_option("logger", "SILENT")
        solver.set_option("print_solution", False)
    if solver_set.max_iterations is not None:
        solver.set_option("max_iterations", solver_set.max_iterations)
    if solver_set.time_limit is not None:
        solver.set_option("time_limit", solver_set.time_limit)

    start = time.perf_counter()
    result = solver.optimize(model)
    elapsed = time.perf_counter() - start
    x = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        (
            k,
            a,
            q,
            c,
            mu,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
        ) = path_values(x, K_eval, K_tilde_eval)
        z = jnp.ones_like(k)
        residuals = path_residuals(
            k,
            a,
            q,
            c,
            mu,
            z,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
            parameters,
        )
        investment = investment_expenditure(k, q, parameters)
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": q,
            "c": c,
            "mu": mu,
            "x": investment,
            "output": production(k, z, parameters),
            "current_account": da_dt,
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
    residual_names = (
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth",
        "consumption_foc",
    )
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(
        k_0, a_0, parameters=parameters
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
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        train_eval["t"][-1],
        parameters,
    )

    solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    rejection_reasons = []
    if solver_status != "SUCCESS":
        rejection_reasons.append("solver_status")
    if solution_status not in ACCEPTED_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    solution_accepted = not rejection_reasons

    alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, offsets = unpack(x)
    c_offset, mu_offset, q_offset = offsets
    rkhs_norm_mu = alpha_mu @ K @ alpha_mu
    rkhs_norm_c = alpha_c @ K @ alpha_c
    rkhs_norm_k = alpha_k @ K @ alpha_k
    rkhs_norm_a = alpha_a @ K @ alpha_a
    rkhs_norm_q = alpha_q @ K @ alpha_q
    rkhs_norm_total = (
        rkhs_norm_mu + rkhs_norm_c + rkhs_norm_k + rkhs_norm_a + rkhs_norm_q
    )
    objective_norm_squared = objective_value(x)

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        k, a, q, c, *_ = path_values(x, K_eval, K_tilde_eval)
        return k, a, q, c

    results = {
        **asdict(parameters),
        "r_star_resolved": parameters.r_star,
        "k_0": k_0,
        "a_0": a_0,
        "a_0_over_output_ss": a_0 / float(output_scale),
        "residual_dtype": "float64",
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
        "c": float(train_eval["c"][0]),
        "q_0": float(steady["q"] * jnp.exp(q_offset)),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_benchmark_error": max_benchmark_error,
        "terminal_residual_count": 0,
        "differential_residual_count": 4 * n_train,
        "algebraic_residual_count": n_train,
        "dae_residual_count": 5 * n_train,
        "unknown_count": 5 * n_train + 3,
        "dynamic_rkhs_function_count": 4,
        "algebraic_rkhs_function_count": 1,
        "free_initial_value_count": 3,
        "metric_selection_dimension": 3,
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
        "max_consumption_variation": float(
            jnp.max(jnp.abs(train_eval["c"] - train_eval["c"][0]))
        ),
        "rkhs_norm_mu": float(rkhs_norm_mu),
        "rkhs_norm_c": float(rkhs_norm_c),
        "rkhs_norm_k": float(rkhs_norm_k),
        "rkhs_norm_a": float(rkhs_norm_a),
        "rkhs_norm_q": float(rkhs_norm_q),
        "rkhs_norm_total": float(rkhs_norm_total),
        "objective_value": float(objective_norm_squared),
        "solve_time": solve_time,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted" if solution_accepted else ",".join(rejection_reasons)
        ),
        "c_offset": float(c_offset),
        "mu_offset": float(mu_offset),
        "q_offset": float(q_offset),
    }

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
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth",
        "consumption_foc",
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
        "solver_set": asdict(solver_set),
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_kernel_uno)
