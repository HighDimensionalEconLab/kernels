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
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)

ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
MPK_BOUND_TOL = 1e-3
DOMAIN_EPS = 1e-6


@dataclass(frozen=True)
class KernelSettings:
    nu: float = 0.5  # math.inf selects the Gaussian/RBF kernel
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = CONSTANTS.train_T
    train_points: int = 101
    train_points_list: list[float] | None = None
    test_T: float = CONSTANTS.sequence_test_T
    test_points: int = 100
    benchmark_T: float = CONSTANTS.benchmark_T
    benchmark_points: int = CONSTANTS.benchmark_points


@dataclass(frozen=True)
class SolverSettings:
    preset: str = "ipopt"
    max_iterations: int | None = None
    time_limit: float | None = None


# Kinked production needs the fail-fast budget: near the basin threshold the
# NLP can wander, and the sweep classifies a slow solve as rejected. Both
# limits are calibrated to the kinked experiments' 41-point training design
# (the replication drivers pass it; the smooth-model default is 101 points,
# where 25 iterations alone cost ~1 s and the wall fires spuriously).
KINKED_SOLVER_SETTINGS = SolverSettings(max_iterations=25, time_limit=0.15)


# The shared NEOCLASSICAL_GROWTH_CONSTANTS (utilities.py) maps the Aruoba,
# Fernandez-Villaverde & Rubio-Ramirez (2006 JEDC) benchmark values.
def neoclassical_growth_kernel_uno(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    k_0: float = 1.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    solver_set: SolverSettings | None = None,
    c_0_guess: float | None = None,  # overrides the flat steady-flow warm start
    verbose: bool = False,
):
    # IPOPT works in float64 C arrays, so this solver is float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if sigma_crra <= 0.0:
        raise ValueError("sigma_crra must be positive")
    kinked_parameter_count = sum(parameter is not None for parameter in (A, b_1, b_2))
    if kinked_parameter_count not in {0, 3}:
        raise ValueError("Set all of A, b_1, and b_2, or set none of them.")
    use_kinked_production = kinked_parameter_count == 3
    A_value = 1.0 if A is None else float(A)
    b_1_value = 1.0 if b_1 is None else float(b_1)
    b_2_value = 0.0 if b_2 is None else float(b_2)
    if solver_set is None:
        solver_set = (
            KINKED_SOLVER_SETTINGS if use_kinked_production else SolverSettings()
        )

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
    validation_points = max(data_set.test_points, data_set.benchmark_points)
    validation_data = jnp.linspace(
        0, data_set.train_T, validation_points, dtype=jnp.float64
    )
    benchmark_grid = jnp.linspace(
        0, data_set.benchmark_T, data_set.benchmark_points, dtype=jnp.float64
    )

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, nu, sigma, ell)
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    n_variables = 3 * n_train + 2
    n_constraints = 6 * n_train

    def initial_production(k_value):
        if use_kinked_production:
            return A_value * max(k_value**a, b_1_value * k_value**a - b_2_value)
        return k_value**a

    c_flow_guess = max(initial_production(k_0) - delta * k_0, DOMAIN_EPS)
    if c_0_guess is not None:
        c_flow_guess = max(float(c_0_guess), DOMAIN_EPS)
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[3 * n_train] = c_flow_guess
    x_initial[3 * n_train + 1] = c_flow_guess**-sigma_crra

    def unpack(x):
        alpha_mu = x[:n_train]
        alpha_c = x[n_train : 2 * n_train]
        alpha_k = x[2 * n_train : 3 * n_train]
        c_0 = x[3 * n_train]
        mu_0 = x[3 * n_train + 1]
        return alpha_mu, alpha_c, alpha_k, c_0, mu_0

    def path_values(x, K_eval, K_tilde_eval):
        alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)
        mu = mu_0 + K_tilde_eval @ alpha_mu
        c = c_0 + K_tilde_eval @ alpha_c
        k = k_0 + K_tilde_eval @ alpha_k
        dmu_dt = K_eval @ alpha_mu
        dk_dt = K_eval @ alpha_k
        return k, c, mu, dk_dt, dmu_dt

    def production_scalar(k_scalar):
        k_positive = jnp.maximum(k_scalar, DOMAIN_EPS)
        z = k_positive**a
        if use_kinked_production:
            return A_value * jnp.maximum(z, b_1_value * z - b_2_value)
        return z

    production = jax.vmap(production_scalar)
    marginal_product = jax.vmap(jax.grad(production_scalar))

    def objective(x):
        alpha_mu, alpha_c, alpha_k, _, _ = unpack(x)
        return (
            alpha_mu @ K_jax @ alpha_mu
            + alpha_c @ K_jax @ alpha_c
            + alpha_k @ K_jax @ alpha_k
        )

    # CRRA first-order condition u'(c) = mu, i.e. mu c^sigma_crra = 1; the
    # costate and resource residuals are utility-free.
    def constraints(x):
        k, c, mu, dk_dt, dmu_dt = path_values(x, K_jax, K_tilde_jax)
        output = production(k)
        mpk = marginal_product(k)
        resource = dk_dt - (output - delta * k - c)
        euler = dmu_dt + mu * (mpk - delta - rho_hat)
        shadow_price = mu * c**sigma_crra - 1.0
        return jnp.concatenate([resource, euler, shadow_price, k, c, mu])

    def lagrangian(x, objective_multiplier, multipliers):
        return objective_multiplier * objective(x) + jnp.dot(
            multipliers, constraints(x)
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
    variable_lower_bounds[3 * n_train :] = DOMAIN_EPS
    constraint_lower_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(3 * n_train, DOMAIN_EPS, dtype=np.float64),
        ]
    )
    constraint_upper_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(3 * n_train, np.inf, dtype=np.float64),
        ]
    )

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
    alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)
    rkhs_norms = {
        "k": float(rkhs_norm_squared(alpha_k, K)),
        "c": float(rkhs_norm_squared(alpha_c, K)),
        "mu": float(rkhs_norm_squared(alpha_mu, K)),
    }

    def evaluate_grid(points_data):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points_data, train_data, nu, sigma, ell
        )
        k_values, c_values, mu_values, dk_values, dmu_values = path_values(
            x, K_eval, K_tilde_eval
        )
        output_values = production(k_values)
        mpk_values = marginal_product(k_values)
        k_positive = jnp.maximum(k_values, DOMAIN_EPS)
        if use_kinked_production:
            m1_values = A_value * a * k_positive ** (a - 1.0)
            m2_values = b_1_value * m1_values
        else:
            m1_values = mpk_values
            m2_values = mpk_values
        implied_mpk = delta + rho_hat - dmu_values / mu_values
        mpk_lower_bound = jnp.minimum(m1_values, m2_values)
        mpk_upper_bound = jnp.maximum(m1_values, m2_values)
        return {
            "t": points_data,
            "k": k_values,
            "c": c_values,
            "mu": mu_values,
            "Y": output_values,
            "resource": dk_values - (output_values - delta * k_values - c_values),
            "euler": dmu_values + mu_values * (mpk_values - delta - rho_hat),
            "shadow_price": mu_values * c_values**sigma_crra - 1.0,
            "mpk_lower_violation": jnp.maximum(mpk_lower_bound - implied_mpk, 0.0),
            "mpk_upper_violation": jnp.maximum(implied_mpk - mpk_upper_bound, 0.0),
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    residual_names = ["resource", "euler", "shadow_price"]
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )
    p_lower_violation = max(
        float(jnp.max(train_eval["mpk_lower_violation"])),
        float(jnp.max(validation_eval["mpk_lower_violation"])),
    )
    p_upper_violation = max(
        float(jnp.max(train_eval["mpk_upper_violation"])),
        float(jnp.max(validation_eval["mpk_upper_violation"])),
    )

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, nu, sigma, ell
        )
        k_test, c_test, _, _, _ = path_values(x, K_test, K_tilde_test)
        return k_test, c_test

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name]))
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "c", "mu")
        )
        and all(
            jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "c", "mu")
        )
    )

    solve_time = result.cpu_time
    if solve_time is None:
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
    if use_kinked_production and max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    if use_kinked_production and (
        p_lower_violation > MPK_BOUND_TOL or p_upper_violation > MPK_BOUND_TOL
    ):
        rejection_reasons.append("p_bound")
    solution_accepted = not rejection_reasons

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "k_0": k_0,
        "A": A_value,
        "b_1": b_1_value,
        "b_2": b_2_value,
        "use_kinked_production": use_kinked_production,
        "nu": nu,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "rkhs_norm_k": rkhs_norms["k"],
        "rkhs_norm_c": rkhs_norms["c"],
        "rkhs_norm_mu": rkhs_norms["mu"],
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_eval["euler"]))),
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

    train_df = pd.DataFrame(
        {
            name: np.asarray(train_eval[name])
            for name in ("t", "k", "c", "mu", "resource", "euler", "shadow_price")
        }
    )
    test_df = pd.DataFrame(
        {
            name: np.asarray(test_eval[name])
            for name in ("t", "k", "c", "mu", "Y", "resource", "euler", "shadow_price")
        }
    )

    if not use_kinked_production:
        benchmark_solution = neoclassical_growth_benchmark(
            a, delta, rho_hat, sigma_crra, k_0, benchmark_grid
        )
        k_benchmark, c_benchmark = benchmark_solution(test_data)
        k_rel_error = jnp.abs(k_benchmark - test_eval["k"]) / k_benchmark
        c_rel_error = jnp.abs(c_benchmark - test_eval["c"]) / c_benchmark
        test_df["k_benchmark"] = np.asarray(k_benchmark)
        test_df["c_benchmark"] = np.asarray(c_benchmark)
        test_df["k_rel_error"] = np.asarray(k_rel_error)
        test_df["c_rel_error"] = np.asarray(c_rel_error)
        # min/max error statistics exclude the pinned t = 0 point ([1:]): its
        # near-zero error is an artifact of the initial condition, not accuracy.
        results.update(
            k_rel_error_mean=float(k_rel_error.mean()),
            c_rel_error_mean=float(c_rel_error.mean()),
            k_rel_error_max=float(k_rel_error[1:].max()),
            c_rel_error_max=float(c_rel_error[1:].max()),
            k_rel_error_min=float(k_rel_error[1:].min()),
            c_rel_error_min=float(c_rel_error[1:].min()),
        )
        print(
            f"solve_time(s) = {solve_time}, "
            f"E(|rel_error(k)|) = {results['k_rel_error_mean']:.3e}, "
            f"E(|rel_error(c)|) = {results['c_rel_error_mean']:.3e}"
        )
    else:
        print(f"solve_time(s) = {solve_time}")

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "A": A_value,
            "b_1": b_1_value,
            "b_2": b_2_value,
        },
        "initial_conditions": {"k_0": k_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_uno)
