from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import unopy

from models.kernels import integrated_kernel_matrices, rkhs_norm_squared

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
TRAIN_RESIDUAL_TOL = 1e-8
DOMAIN_EPS = 1e-8


@dataclass(frozen=True)
class KernelSettings:
    nu: float = 0.5  # math.inf selects the Gaussian/RBF kernel
    ell: float = 15.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100


@dataclass(frozen=True)
class SolverSettings:
    preset: str = "ipopt"
    max_iterations: int | None = None
    time_limit: float | None = None


def optimal_advertising_kernel_uno(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    solver_set: SolverSettings = SolverSettings(),
    verbose: bool = False,
):
    nu, ell, sigma = kernel_set.nu, kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(0, data_set.train_T, data_set.train_points)
    else:
        train_data = jnp.array(data_set.train_points_list)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points)

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, nu, sigma, ell)
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    n_variables = 3 * n_train + 2
    n_constraints = 4 * n_train
    gamma = (beta + rho_hat) / c
    control_power = (1.0 - kappa) / kappa
    u_0_guess = max(beta * x_0 / max(1.0 - x_0, DOMAIN_EPS), DOMAIN_EPS)
    mu_0_guess = max(gamma / (rho_hat + beta + u_0_guess), DOMAIN_EPS)
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[3 * n_train] = mu_0_guess
    x_initial[3 * n_train + 1] = u_0_guess

    def unpack(z):
        alpha_x = z[:n_train]
        alpha_mu = z[n_train : 2 * n_train]
        alpha_u = z[2 * n_train : 3 * n_train]
        mu_0 = z[3 * n_train]
        u_0 = z[3 * n_train + 1]
        return alpha_x, alpha_mu, alpha_u, mu_0, u_0

    def path_values(z, K_eval, K_tilde_eval):
        alpha_x, alpha_mu, alpha_u, mu_0, u_0 = unpack(z)
        x = x_0 + K_tilde_eval @ alpha_x
        mu = mu_0 + K_tilde_eval @ alpha_mu
        u = u_0 + K_tilde_eval @ alpha_u
        dx_dt = K_eval @ alpha_x
        dmu_dt = K_eval @ alpha_mu
        return x, mu, u, dx_dt, dmu_dt

    def objective(z):
        alpha_x, alpha_mu, alpha_u, _, _ = unpack(z)
        return (
            alpha_x @ K_jax @ alpha_x
            + alpha_mu @ K_jax @ alpha_mu
            + alpha_u @ K_jax @ alpha_u
        )

    def residuals(x, mu, u, dx_dt, dmu_dt):
        state = dx_dt - ((1.0 - x) * u - beta * x)
        costate = dmu_dt - (-gamma + (rho_hat + beta) * mu + mu * u)
        shadow_price = u**control_power - kappa * mu * (1.0 - x)
        return state, costate, shadow_price

    def constraints(z):
        x, mu, u, dx_dt, dmu_dt = path_values(z, K_jax, K_tilde_jax)
        state, costate, shadow_price = residuals(x, mu, u, dx_dt, dmu_dt)
        return jnp.concatenate([state, costate, shadow_price, u])

    def lagrangian(z, objective_multiplier, multipliers):
        return objective_multiplier * objective(z) + jnp.dot(
            multipliers, constraints(z)
        )

    objective_value = jax.jit(objective)
    objective_gradient = jax.jit(jax.grad(objective))
    constraint_values = jax.jit(constraints)
    constraint_jacobian = jax.jit(jax.jacfwd(constraints))
    lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

    x_initial_jax = jnp.asarray(x_initial)
    zero_multipliers = jnp.zeros(n_constraints, dtype=jnp.float64)
    jax.block_until_ready(objective_value(x_initial_jax))
    jax.block_until_ready(objective_gradient(x_initial_jax))
    jax.block_until_ready(constraint_values(x_initial_jax))
    jax.block_until_ready(constraint_jacobian(x_initial_jax))
    jax.block_until_ready(lagrangian_hessian(x_initial_jax, 1.0, zero_multipliers))

    variable_lower_bounds = np.full(n_variables, -np.inf, dtype=np.float64)
    variable_upper_bounds = np.full(n_variables, np.inf, dtype=np.float64)
    variable_lower_bounds[3 * n_train] = 0.0
    variable_lower_bounds[3 * n_train + 1] = DOMAIN_EPS
    constraint_lower_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(n_train, DOMAIN_EPS, dtype=np.float64),
        ]
    )
    constraint_upper_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(n_train, np.inf, dtype=np.float64),
        ]
    )

    model = unopy.Model(
        unopy.PROBLEM_NONLINEAR,
        n_variables,
        variable_lower_bounds,
        variable_upper_bounds,
        unopy.ZERO_BASED_INDEXING,
    )

    def objective_callback(z):
        return float(objective_value(jnp.asarray(z)))

    def objective_gradient_callback(z, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(z)))

    model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(z, constraint_output):
        constraint_output[:] = np.asarray(constraint_values(jnp.asarray(z)))

    def jacobian_callback(z, jacobian_output):
        jacobian_output[:] = np.asarray(constraint_jacobian(jnp.asarray(z))).reshape(-1)

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

    def hessian_callback(z, objective_multiplier, multipliers, hessian_output):
        hessian = np.asarray(
            lagrangian_hessian(
                jnp.asarray(z),
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

    z = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))
    alpha_x, alpha_mu, alpha_u, mu_0, u_0 = unpack(z)

    x_train, mu_train, u_train, dx_train, dmu_train = path_values(z, K_jax, K_tilde_jax)
    state_res, costate_res, shadow_res = residuals(
        x_train, mu_train, u_train, dx_train, dmu_train
    )
    max_train_residual = max(
        float(jnp.max(jnp.abs(residual)))
        for residual in (state_res, costate_res, shadow_res)
    )

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, nu, sigma, ell
        )
        x_test, mu_test, u_test, _, _ = path_values(z, K_test, K_tilde_test)
        return x_test, mu_test, u_test

    x_test, mu_test, u_test = kernel_solution(test_data)

    solve_time = result.cpu_time
    if solve_time is None:
        solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    finite_valid = bool(
        jnp.all(jnp.isfinite(x_test))
        and jnp.all(jnp.isfinite(mu_test))
        and jnp.all(jnp.isfinite(u_test))
        and jnp.min(u_test) > 0.0
    )
    rejection_reasons = []
    if solver_status != "SUCCESS":
        rejection_reasons.append("solver_status")
    if solution_status not in ACCEPTED_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite_valid:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    solution_accepted = not rejection_reasons

    results = {
        "rho_hat": rho_hat,
        "c": c,
        "beta": beta,
        "kappa": kappa,
        "x_0": x_0,
        "gamma": gamma,
        "nu": nu,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "mu_0": float(mu_0),
        "u_0": float(u_0),
        "rkhs_norm_x": rkhs_norm_squared(alpha_x, K),
        "rkhs_norm_mu": rkhs_norm_squared(alpha_mu, K),
        "rkhs_norm_u": rkhs_norm_squared(alpha_u, K),
        "max_train_residual": max_train_residual,
        "solve_time": solve_time,
        "wall_time": elapsed,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        "solution_accepted": solution_accepted,
        "rejection_reason": "accepted"
        if solution_accepted
        else ",".join(rejection_reasons),
    }
    print(f"solve_time(s) = {solve_time}")

    train_df = pd.DataFrame(
        {
            "t": np.asarray(train_data),
            "x": np.asarray(x_train),
            "mu": np.asarray(mu_train),
            "u": np.asarray(u_train),
            "state": np.asarray(state_res),
            "costate": np.asarray(costate_res),
            "shadow_price": np.asarray(shadow_res),
        }
    )
    test_df = pd.DataFrame(
        {
            "t": np.asarray(test_data),
            "x": np.asarray(x_test),
            "mu": np.asarray(mu_test),
            "u": np.asarray(u_test),
        }
    )
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "parameters": {
            "rho_hat": rho_hat,
            "c": c,
            "beta": beta,
            "kappa": kappa,
            "gamma": gamma,
        },
        "initial_conditions": {"x_0": x_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(optimal_advertising_kernel_uno)
