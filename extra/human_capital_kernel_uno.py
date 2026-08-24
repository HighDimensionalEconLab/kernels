from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import unopy
from nlls_gram import LevenbergMarquardt

from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    rkhs_norm_squared,
)
from models.utilities import Stopwatch, acceptance_fields

ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}


def unpack(x):
    n_train = (x.shape[0] - 5) // 7
    alpha_k = x[:n_train]
    alpha_h = x[n_train : 2 * n_train]
    alpha_i_k = x[2 * n_train : 3 * n_train]
    alpha_i_h = x[3 * n_train : 4 * n_train]
    alpha_c = x[4 * n_train : 5 * n_train]
    alpha_mu_k = x[5 * n_train : 6 * n_train]
    alpha_mu_h = x[6 * n_train : 7 * n_train]
    i_k_0 = x[7 * n_train]
    i_h_0 = x[7 * n_train + 1]
    c_0 = x[7 * n_train + 2]
    mu_k_0 = x[7 * n_train + 3]
    mu_h_0 = x[7 * n_train + 4]
    return (
        alpha_k,
        alpha_h,
        alpha_i_k,
        alpha_i_h,
        alpha_c,
        alpha_mu_k,
        alpha_mu_h,
        i_k_0,
        i_h_0,
        c_0,
        mu_k_0,
        mu_h_0,
    )


def production_terms(k, h, p):
    a_k, a_h = p["a_k"], p["a_h"]
    f = (k**a_k) * (h**a_h)
    f_k = a_k * (k ** (a_k - 1.0)) * (h**a_h)
    f_h = a_h * (k**a_k) * (h ** (a_h - 1.0))
    return f, f_k, f_h


def objective(x, K):
    (
        alpha_k,
        alpha_h,
        alpha_i_k,
        alpha_i_h,
        alpha_c,
        alpha_mu_k,
        alpha_mu_h,
        *_,
    ) = unpack(x)
    return (
        alpha_k @ K @ alpha_k
        + alpha_h @ K @ alpha_h
        + alpha_mu_k @ K @ alpha_mu_k
        + alpha_mu_h @ K @ alpha_mu_h
        + alpha_i_k @ K @ alpha_i_k
        + alpha_i_h @ K @ alpha_i_h
        + alpha_c @ K @ alpha_c
    )


def constraints(x, K, K_tilde, p):
    delta_k, delta_h, rho_hat = p["delta_k"], p["delta_h"], p["rho_hat"]
    (
        alpha_k,
        alpha_h,
        alpha_i_k,
        alpha_i_h,
        alpha_c,
        alpha_mu_k,
        alpha_mu_h,
        i_k_0,
        i_h_0,
        c_0,
        mu_k_0,
        mu_h_0,
    ) = unpack(x)
    paths = integrated_kernel_paths(
        {
            "k": IntegratedPath(alpha_k, p["k_0"]),
            "h": IntegratedPath(alpha_h, p["h_0"]),
            "i_k": IntegratedPath(alpha_i_k, i_k_0),
            "i_h": IntegratedPath(alpha_i_h, i_h_0),
            "c": IntegratedPath(alpha_c, c_0),
            "mu_k": IntegratedPath(alpha_mu_k, mu_k_0),
            "mu_h": IntegratedPath(alpha_mu_h, mu_h_0),
        },
        K,
        K_tilde,
    )
    k, h, i_k, i_h, c, mu_k, mu_h = (
        paths["k"].value,
        paths["h"].value,
        paths["i_k"].value,
        paths["i_h"].value,
        paths["c"].value,
        paths["mu_k"].value,
        paths["mu_h"].value,
    )
    dk_dt, dh_dt, dmu_k_dt, dmu_h_dt = (
        paths["k"].d_dt,
        paths["h"].d_dt,
        paths["mu_k"].d_dt,
        paths["mu_h"].d_dt,
    )
    f, f_k, f_h = production_terms(k, h, p)
    return jnp.concatenate(
        [
            dk_dt - (i_k - delta_k * k),
            dh_dt - (i_h - delta_h * h),
            dmu_k_dt + mu_k * (f_k - delta_k - rho_hat),
            dmu_h_dt + mu_h * (f_h - delta_h - rho_hat),
            c + i_h + i_k - f,
            mu_k * c - 1.0,
            mu_k - mu_h,
        ]
    )


def lagrangian(x, objective_multiplier, multipliers, K, K_tilde, p):
    return objective_multiplier * objective(x, K) + jnp.dot(
        multipliers, constraints(x, K, K_tilde, p)
    )


# Module level: per-call grad/jacfwd/hessian closures would retrace on
# every solve.
objective_value = jax.jit(objective)
objective_gradient = jax.jit(jax.grad(objective))
constraint_values = jax.jit(constraints)
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 80.0
    train_points: int = 81
    train_points_list: list[float] | None = None
    test_T: float = 100.0
    test_points: int = 100
    validation_points: int = 300


@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5


@dataclass(frozen=True)
class SolverSettings:
    preset: str = "filtersqp"
    max_iterations: int | None = None
    # Acceptance is wall-clock gated: filtersqp stops (and the solve is rejected)
    # when the limit is hit, so the budget must comfortably cover the NLP size.
    time_limit: float | None = 10.0


def human_capital_initial_residual(params, batch):
    a_k, a_h, delta_k, delta_h, k_0 = batch
    h_0 = jnp.exp(params["log_h"])
    c_0 = jnp.exp(params["log_c"])
    f = (k_0**a_k) * (h_0**a_h)
    f_k = a_k * (k_0 ** (a_k - 1.0)) * (h_0**a_h)
    f_h = a_h * (k_0**a_k) * (h_0 ** (a_h - 1.0))
    return jnp.array(
        [
            f_h - delta_h - (f_k - delta_k),
            c_0 + delta_h * h_0 + delta_k * k_0 - f,
        ],
        dtype=h_0.dtype,
    )


def human_capital_initial_conditions(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    k_0: float = 1.5,
    iterations: int = 20,
):
    dtype = jnp.float64
    batch = tuple(
        jnp.asarray(value, dtype=dtype) for value in (a_k, a_h, delta_k, delta_h, k_0)
    )
    k_0_jax = batch[-1]
    c_guess = (
        (k_0_jax ** batch[0]) * (k_0_jax ** batch[1])
        - batch[3] * k_0_jax
        - batch[2] * k_0_jax
    )
    c_guess = jnp.maximum(c_guess, jnp.asarray(1e-12, dtype=dtype))
    params = {
        "log_h": jnp.log(k_0_jax),
        "log_c": jnp.log(c_guess),
    }
    solver = LevenbergMarquardt(human_capital_initial_residual, init_damping=1e-4)
    result = solver.solve(
        params,
        batch,
        max_steps=iterations,
        max_steps_is_success=False,
    )

    h_0 = jnp.exp(result.x["log_h"])
    c_0 = jnp.exp(result.x["log_c"])
    residual = human_capital_initial_residual(result.x, batch)
    return h_0, c_0, residual, result.info


def human_capital_kernel_uno(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11,
    k_0: float = 1.5,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    solver_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(0, data_set.train_T, data_set.train_points)
    else:
        train_data = jnp.array(data_set.train_points_list)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points)
    validation_points = max(data_set.test_points, data_set.validation_points)
    validation_data = jnp.linspace(0, data_set.train_T, validation_points)

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    h_0_jax, c_0_init_jax, initial_residual, _ = human_capital_initial_conditions(
        a_k, a_h, delta_k, delta_h, k_0
    )
    h_0 = float(h_0_jax)
    c_0_init = float(c_0_init_jax)
    p = {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in {
            "a_k": a_k,
            "a_h": a_h,
            "delta_k": delta_k,
            "delta_h": delta_h,
            "rho_hat": rho_hat,
            "k_0": k_0,
            "h_0": h_0,
        }.items()
    }

    n_functions = 7
    n_scalars = 5
    n_variables = n_functions * n_train + n_scalars
    n_constraints = 7 * n_train
    x_initial = np.zeros(n_variables, dtype=np.float64)
    scalar_start = n_functions * n_train
    x_initial[scalar_start] = delta_k * k_0
    x_initial[scalar_start + 1] = delta_h * h_0
    x_initial[scalar_start + 2] = c_0_init
    x_initial[scalar_start + 3] = 1.0 / c_0_init
    x_initial[scalar_start + 4] = 1.0 / c_0_init

    x_initial_jax = jnp.asarray(x_initial)
    zero_multipliers = jnp.zeros(n_constraints, dtype=jnp.float64)
    jax.block_until_ready(objective_value(x_initial_jax, K_jax))
    jax.block_until_ready(objective_gradient(x_initial_jax, K_jax))
    jax.block_until_ready(constraint_values(x_initial_jax, K_jax, K_tilde_jax, p))
    jax.block_until_ready(constraint_jacobian(x_initial_jax, K_jax, K_tilde_jax, p))
    jax.block_until_ready(
        lagrangian_hessian(x_initial_jax, 1.0, zero_multipliers, K_jax, K_tilde_jax, p)
    )

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
        return float(objective_value(jnp.asarray(x), K_jax))

    def objective_gradient_callback(x, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(x), K_jax))

    model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(x, constraint_output):
        constraint_output[:] = np.asarray(
            constraint_values(jnp.asarray(x), K_jax, K_tilde_jax, p)
        )

    def jacobian_callback(x, jacobian_output):
        jacobian_output[:] = np.asarray(
            constraint_jacobian(jnp.asarray(x), K_jax, K_tilde_jax, p)
        ).reshape(-1)

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
                K_jax,
                K_tilde_jax,
                p,
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
    if not verbose:
        solver.set_option("logger", "SILENT")
        solver.set_option("print_solution", False)
    if solver_set.max_iterations is not None:
        solver.set_option("max_iterations", solver_set.max_iterations)
    if solver_set.time_limit is not None:
        solver.set_option("time_limit", solver_set.time_limit)

    timer = Stopwatch()
    result = solver.optimize(model)
    elapsed = timer.elapsed()

    # primal_solution appends solver-internal auxiliaries; only the leading
    # primal block is the solution.
    x = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))[:n_variables]
    (
        alpha_k,
        alpha_h,
        alpha_i_k,
        alpha_i_h,
        alpha_c,
        alpha_mu_k,
        alpha_mu_h,
        i_k_0,
        i_h_0,
        c_0,
        mu_k_0,
        mu_h_0,
    ) = unpack(x)
    rkhs_norms = {
        "k": float(rkhs_norm_squared(alpha_k, K)),
        "h": float(rkhs_norm_squared(alpha_h, K)),
        "i_k": float(rkhs_norm_squared(alpha_i_k, K)),
        "i_h": float(rkhs_norm_squared(alpha_i_h, K)),
        "c": float(rkhs_norm_squared(alpha_c, K)),
        "mu_k": float(rkhs_norm_squared(alpha_mu_k, K)),
        "mu_h": float(rkhs_norm_squared(alpha_mu_h, K)),
    }

    def evaluate_grid(points_data):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points_data, train_data, kernel, sigma, ell
        )
        paths = integrated_kernel_paths(
            {
                "k": IntegratedPath(alpha_k, p["k_0"]),
                "h": IntegratedPath(alpha_h, p["h_0"]),
                "i_k": IntegratedPath(alpha_i_k, i_k_0),
                "i_h": IntegratedPath(alpha_i_h, i_h_0),
                "c": IntegratedPath(alpha_c, c_0),
                "mu_k": IntegratedPath(alpha_mu_k, mu_k_0),
                "mu_h": IntegratedPath(alpha_mu_h, mu_h_0),
            },
            K_eval,
            K_tilde_eval,
        )
        k_values, h_values, i_k_values, i_h_values = (
            paths["k"].value,
            paths["h"].value,
            paths["i_k"].value,
            paths["i_h"].value,
        )
        c_values, mu_k_values, mu_h_values = (
            paths["c"].value,
            paths["mu_k"].value,
            paths["mu_h"].value,
        )
        dk_values, dh_values, dmu_k_values, dmu_h_values = (
            paths["k"].d_dt,
            paths["h"].d_dt,
            paths["mu_k"].d_dt,
            paths["mu_h"].d_dt,
        )
        f_values, f_k_values, f_h_values = production_terms(k_values, h_values, p)
        return {
            "t": points_data,
            "k": k_values,
            "h": h_values,
            "i_k": i_k_values,
            "i_h": i_h_values,
            "c": c_values,
            "mu_k": mu_k_values,
            "mu_h": mu_h_values,
            "physical_accumulation": dk_values - (i_k_values - delta_k * k_values),
            "human_accumulation": dh_values - (i_h_values - delta_h * h_values),
            "physical_euler": dmu_k_values
            + mu_k_values * (f_k_values - delta_k - rho_hat),
            "human_euler": dmu_h_values
            + mu_h_values * (f_h_values - delta_h - rho_hat),
            "feasibility": c_values + i_h_values + i_k_values - f_values,
            "shadow_price": mu_k_values * c_values - 1.0,
            "costate_gap": mu_k_values - mu_h_values,
            "hidden_dae_residual": (f_k_values - delta_k) - (f_h_values - delta_h),
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    residual_names = [
        "physical_accumulation",
        "human_accumulation",
        "physical_euler",
        "human_euler",
        "feasibility",
        "shadow_price",
        "costate_gap",
    ]
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )
    positive_names = ["k", "h", "i_k", "i_h", "c", "mu_k", "mu_h"]
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(test_eval[name]))
            and jnp.all(jnp.isfinite(validation_eval[name]))
            and jnp.min(test_eval[name]) > 0.0
            and jnp.min(validation_eval[name]) > 0.0
            for name in positive_names
        )
    )

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, kernel, sigma, ell
        )
        paths = integrated_kernel_paths(
            {
                "k": IntegratedPath(alpha_k, p["k_0"]),
                "h": IntegratedPath(alpha_h, p["h_0"]),
                "i_k": IntegratedPath(alpha_i_k, i_k_0),
                "i_h": IntegratedPath(alpha_i_h, i_h_0),
                "c": IntegratedPath(alpha_c, c_0),
                "mu_k": IntegratedPath(alpha_mu_k, mu_k_0),
                "mu_h": IntegratedPath(alpha_mu_h, mu_h_0),
            },
            K_test,
            K_tilde_test,
        )
        return (
            paths["k"].value,
            paths["h"].value,
            paths["c"].value,
            paths["i_k"].value,
            paths["i_h"].value,
            paths["mu_k"].value,
            paths["mu_h"].value,
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
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")

    results = {
        "a_k": a_k,
        "a_h": a_h,
        "delta_k": delta_k,
        "delta_h": delta_h,
        "rho_hat": rho_hat,
        "k_0": k_0,
        "h_0": h_0,
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "i_k_0": float(i_k_0),
        "i_h_0": float(i_h_0),
        "mu_k_0": float(mu_k_0),
        "mu_h_0": float(mu_h_0),
        "initial_condition_residual": float(
            jnp.linalg.norm(initial_residual, ord=jnp.inf)
        ),
        **{f"rkhs_norm_{name}": value for name, value in rkhs_norms.items()},
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_test_feasibility": float(jnp.max(jnp.abs(test_eval["feasibility"]))),
        "max_test_hidden_dae_residual": float(
            jnp.max(jnp.abs(test_eval["hidden_dae_residual"]))
        ),
        "solve_time": solve_time,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        **acceptance_fields(rejection_reasons),
    }
    print(f"solve_time(s) = {solve_time}")

    frame_columns = [
        "t",
        "k",
        "h",
        "i_k",
        "i_h",
        "c",
        "mu_k",
        "mu_h",
        "feasibility",
        "hidden_dae_residual",
    ]
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_columns}
    )
    test_df = pd.DataFrame(
        {name: np.asarray(test_eval[name]) for name in frame_columns}
    )
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "tolerance_set": asdict(tolerance_set),
        "parameters": {
            "a_k": a_k,
            "a_h": a_h,
            "delta_k": delta_k,
            "delta_h": delta_h,
            "rho_hat": rho_hat,
        },
        "initial_conditions": {"k_0": k_0, "h_0": h_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_kernel_uno)
