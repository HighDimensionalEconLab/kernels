from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import unopy

from models.asset_pricing_benchmark import mu_f_array
from models.kernels import integrated_kernel_matrices, rkhs_norm_squared

ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
TRAIN_RESIDUAL_TOL = 1e-8


def unpack(z):
    n_train = z.shape[0] - 1
    alpha = z[:n_train]
    p_0 = z[n_train]
    return alpha, p_0


def path_values(z, K_eval, K_tilde_eval):
    alpha, p_0 = unpack(z)
    price = p_0 + K_tilde_eval @ alpha
    dprice_dt = K_eval @ alpha
    return price, dprice_dt


def dividends(t, p):
    return (p["x_0"] + p["c"] / p["g"]) * jnp.exp(p["g"] * t) - p["c"] / p["g"]


@jax.jit
def objective(z, K):
    alpha, _ = unpack(z)
    return alpha @ K @ alpha


# No-arbitrage residual dp/dt = r p - x along the dividend path.
@jax.jit
def constraints(z, K, K_tilde, t, p):
    price, dprice_dt = path_values(z, K, K_tilde)
    return dprice_dt - (p["r"] * price - dividends(t, p))


def lagrangian(z, objective_multiplier, multipliers, K, K_tilde, t, p):
    return objective_multiplier * objective(z, K) + jnp.dot(
        multipliers, constraints(z, K, K_tilde, t, p)
    )


# Module level: per-call grad/jacfwd/hessian closures would retrace on
# every solve.
objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))


@dataclass(frozen=True)
class KernelSettings:
    nu: float = 0.5  # math.inf selects the Gaussian/RBF kernel
    ell: float = 10.0
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


def asset_pricing_kernel_uno(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    solver_set: SolverSettings = SolverSettings(),
    verbose: bool = False,
):
    # IPOPT works in float64 C arrays, so this solver is float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    nu, ell, sigma = kernel_set.nu, kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(
            0, data_set.train_T, data_set.train_points, dtype=jnp.float64
        )
    else:
        train_data = jnp.array(data_set.train_points_list, dtype=jnp.float64)
    test_data = jnp.linspace(
        0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, nu, sigma, ell)
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    p = {"r": r, "c": c, "g": g, "x_0": x_0}

    n_variables = n_train + 1
    n_constraints = n_train
    p_0_guess = float(mu_f_array(jnp.array([0.0]), c, g, r, x_0)[0])
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[n_train] = max(p_0_guess, 0.0)

    x_initial_jax = jnp.asarray(x_initial)
    zero_multipliers = jnp.zeros(n_constraints, dtype=jnp.float64)
    jax.block_until_ready(objective(x_initial_jax, K_jax))
    jax.block_until_ready(objective_gradient(x_initial_jax, K_jax))
    jax.block_until_ready(constraints(x_initial_jax, K_jax, K_tilde_jax, train_data, p))
    jax.block_until_ready(
        constraint_jacobian(x_initial_jax, K_jax, K_tilde_jax, train_data, p)
    )
    jax.block_until_ready(
        lagrangian_hessian(
            x_initial_jax, 1.0, zero_multipliers, K_jax, K_tilde_jax, train_data, p
        )
    )

    variable_lower_bounds = np.full(n_variables, -np.inf, dtype=np.float64)
    variable_upper_bounds = np.full(n_variables, np.inf, dtype=np.float64)
    variable_lower_bounds[n_train] = 0.0
    constraint_lower_bounds = np.zeros(n_constraints, dtype=np.float64)
    constraint_upper_bounds = np.zeros(n_constraints, dtype=np.float64)

    model = unopy.Model(
        unopy.PROBLEM_QUADRATIC,
        n_variables,
        variable_lower_bounds,
        variable_upper_bounds,
        unopy.ZERO_BASED_INDEXING,
    )

    def objective_callback(z):
        return float(objective(jnp.asarray(z), K_jax))

    def objective_gradient_callback(z, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(z), K_jax))

    model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(z, constraint_output):
        constraint_output[:] = np.asarray(
            constraints(jnp.asarray(z), K_jax, K_tilde_jax, train_data, p)
        )

    def jacobian_callback(z, jacobian_output):
        jacobian_output[:] = np.asarray(
            constraint_jacobian(jnp.asarray(z), K_jax, K_tilde_jax, train_data, p)
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

    def hessian_callback(z, objective_multiplier, multipliers, hessian_output):
        hessian = np.asarray(
            lagrangian_hessian(
                jnp.asarray(z),
                float(objective_multiplier),
                jnp.asarray(multipliers),
                K_jax,
                K_tilde_jax,
                train_data,
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

    start = time.perf_counter()
    result = solver.optimize(model)
    elapsed = time.perf_counter() - start

    # primal_solution appends solver-internal auxiliaries; only the leading
    # n + 1 block is ours.
    z = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))[:n_variables]
    alpha, p_0 = unpack(z)
    train_residual = constraints(z, K_jax, K_tilde_jax, train_data, p)
    max_train_residual = float(jnp.max(jnp.abs(train_residual)))

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, nu, sigma, ell
        )
        p_test, dp_test = path_values(z, K_test, K_tilde_test)
        return p_test, dp_test

    p_test, dp_test = kernel_solution(test_data)
    test_residual = dp_test - (r * p_test - dividends(test_data, p))
    p_benchmark = mu_f_array(test_data, c, g, r, x_0)
    p_rel_error = jnp.abs(p_benchmark - p_test) / p_benchmark

    solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    finite = bool(jnp.all(jnp.isfinite(p_test)) and jnp.all(jnp.isfinite(p_benchmark)))
    rejection_reasons = []
    if solver_status != "SUCCESS":
        rejection_reasons.append("solver_status")
    if solution_status not in ACCEPTED_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite:
        rejection_reasons.append("nonfinite")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    solution_accepted = not rejection_reasons

    results = {
        "r": r,
        "c": c,
        "g": g,
        "x_0": x_0,
        "nu": nu,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(p_0),
        "rkhs_norm_p": float(rkhs_norm_squared(alpha, K)),
        "max_train_residual": max_train_residual,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
        "solve_time": solve_time,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        "solution_accepted": solution_accepted,
        "rejection_reason": "accepted"
        if solution_accepted
        else ",".join(rejection_reasons),
    }
    print(
        f"solve_time(s) = {solve_time}, "
        f"E(|rel_error(p)|) = {results['p_rel_error_mean']:.3e}"
    )

    p_train, dp_train = path_values(z, K_jax, K_tilde_jax)
    train_df = pd.DataFrame(
        {
            "t": np.asarray(train_data),
            "p": np.asarray(p_train),
            "residual": np.asarray(train_residual),
        }
    )
    test_df = pd.DataFrame(
        {
            "t": np.asarray(test_data),
            "p": np.asarray(p_test),
            "p_benchmark": np.asarray(p_benchmark),
            "p_rel_error": np.asarray(p_rel_error),
            "residual": np.asarray(test_residual),
        }
    )
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "parameters": {"r": r, "c": c, "g": g},
        "initial_conditions": {"x_0": x_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_kernel_uno)
