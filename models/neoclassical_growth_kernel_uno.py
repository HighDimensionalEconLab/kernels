from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import unopy

from models.kernels import (
    integrated_kernel_matrices,
    normalize_kernel,
)
from models.neoclassical_growth import (
    StructuralParameters,
    mpk,
    production,
    steady_flow_consumption_guess,
)
from models.neoclassical_growth_benchmark import benchmark_df
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.utilities import (
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)

ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
# IPOPT's interior point evaluates infeasible trial iterates: the model
# functions clamp k at this floor and the primal bounds keep (c, mu, k) above
# it.
DOMAIN_EPS = 1e-6


def unpack(x):
    n_train = (x.shape[0] - 2) // 3
    alpha_mu = x[:n_train]
    alpha_c = x[n_train : 2 * n_train]
    alpha_k = x[2 * n_train : 3 * n_train]
    c_0 = x[3 * n_train]
    mu_0 = x[3 * n_train + 1]
    return alpha_mu, alpha_c, alpha_k, c_0, mu_0


def path_values(x, K_eval, K_tilde_eval, k_0):
    alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)
    mu = mu_0 + K_tilde_eval @ alpha_mu
    c = c_0 + K_tilde_eval @ alpha_c
    k = k_0 + K_tilde_eval @ alpha_k
    dmu_dt = K_eval @ alpha_mu
    dk_dt = K_eval @ alpha_k
    return k, c, mu, dk_dt, dmu_dt


@jax.jit
def objective(x, K):
    alpha_mu, alpha_c, alpha_k, _, _ = unpack(x)
    return alpha_mu @ K @ alpha_mu + alpha_c @ K @ alpha_c + alpha_k @ K @ alpha_k


# CRRA first-order condition u'(c) = mu, i.e. mu c^sigma_crra = 1; the
# costate and resource residuals are utility-free.
@jax.jit
def constraints(x, K, K_tilde, p):
    k, c, mu, dk_dt, dmu_dt = path_values(x, K, K_tilde, p.k_0)
    k_positive = jnp.maximum(k, DOMAIN_EPS)
    resource = dk_dt - (production(1.0, k_positive, p) - p.delta * k - c)
    euler = dmu_dt + mu * (mpk(1.0, k_positive, p) - p.delta - p.rho_hat)
    shadow_price = mu * c**p.sigma_crra - 1.0
    return jnp.concatenate([resource, euler, shadow_price, k, c, mu])


def lagrangian(x, objective_multiplier, multipliers, K, K_tilde, p):
    return objective_multiplier * objective(x, K) + jnp.dot(
        multipliers, constraints(x, K, K_tilde, p)
    )


# Module level: per-call grad/jacfwd/hessian closures would retrace on
# every solve.
objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern12"  # matern12 | matern32 | matern52 | rbf
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = CONSTANTS.train_T
    train_points: int = 101
    train_points_list: list[float] | None = None
    test_T: float = CONSTANTS.sequence_test_T
    test_points: int = 100


# Acceptance gates, CLI-reachable; the mpk bound applies only under kinked
# production.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 5e-3
    mpk_bound: float = 1e-3


@dataclass(frozen=True)
class SolverSettings:
    preset: str = "ipopt"
    max_iterations: int | None = None
    time_limit: float | None = None


# Fail-fast budget for kinked production: near the basin threshold the NLP
# can wander, and the sweep classifies a slow solve as rejected. Calibrated
# to the 41-point kinked training design the replication drivers pass.
KINKED_SOLVER_SETTINGS = SolverSettings(max_iterations=25, time_limit=0.15)


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
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    c_0_guess: float | None = None,  # overrides the flat steady-flow warm start
    verbose: bool = False,
):
    # Model setup. IPOPT works in float64 C arrays, so this solver is
    # float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    # Kinked production is enabled exactly when all of A, b_1, b_2 are given.
    use_kinked_production = not (A is None or b_1 is None or b_2 is None)
    if use_kinked_production:
        A, b_1, b_2 = float(A), float(b_1), float(b_2)
        k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a)
    else:
        A, b_1, b_2, k_thr = 1.0, 1.0, 0.0, float("inf")
    if solver_set is None:
        solver_set = (
            KINKED_SOLVER_SETTINGS if use_kinked_production else SolverSettings()
        )

    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, jnp.float64)
    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    n_variables = 3 * n_train + 2
    n_constraints = 6 * n_train

    # Value container only: no p rides through IPOPT for implicit
    # differentiation.
    p = StructuralParameters.create(
        jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        k_thr=k_thr,
    )

    c_flow_guess = steady_flow_consumption_guess(k_0, a, delta, A, b_1, b_2)
    if c_0_guess is not None:
        c_flow_guess = max(float(c_0_guess), DOMAIN_EPS)
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[3 * n_train] = c_flow_guess
    x_initial[3 * n_train + 1] = c_flow_guess**-sigma_crra

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

    # IPOPT model wiring.
    model = unopy.Model(
        unopy.PROBLEM_NONLINEAR,
        n_variables,
        variable_lower_bounds,
        variable_upper_bounds,
        unopy.ZERO_BASED_INDEXING,
    )

    def objective_callback(x):
        return float(objective(jnp.asarray(x), K_jax))

    def objective_gradient_callback(x, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(x), K_jax))

    model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(x, constraint_output):
        constraint_output[:] = np.asarray(
            constraints(jnp.asarray(x), K_jax, K_tilde_jax, p)
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
    # 3n + 2 block is ours.
    x = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))[:n_variables]
    c_0, mu_0 = unpack(x)[3], unpack(x)[4]

    # Diagnostics: solved paths and residuals on the train, dense validation,
    # and test grids.
    def snapshot(t_grid):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            t_grid, train_data, kernel, sigma, ell
        )
        k, c, mu, dk_dt, dmu_dt = path_values(x, K_eval, K_tilde_eval, k_0)
        k_positive = jnp.maximum(k, DOMAIN_EPS)
        return pd.DataFrame(
            {
                "t": np.asarray(t_grid),
                "k": np.asarray(k),
                "c": np.asarray(c),
                "mu": np.asarray(mu),
                "resource": np.asarray(
                    dk_dt - (production(1.0, k_positive, p) - delta * k - c)
                ),
                "euler": np.asarray(
                    dmu_dt + mu * (mpk(1.0, k_positive, p) - delta - rho_hat)
                ),
                "shadow_price": np.asarray(mu * c**sigma_crra - 1.0),
            }
        )

    test_data = jnp.linspace(
        0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )
    # Dense off-grid residual check between the training points.
    validation_data = jnp.linspace(
        0, data_set.train_T, max(data_set.test_points, 801), dtype=jnp.float64
    )
    train_df = snapshot(train_data)
    validation_df = snapshot(validation_data)
    test_df = snapshot(test_data)

    residual_names = ("resource", "euler", "shadow_price")
    max_train_residual = max(
        float(train_df[name].abs().max()) for name in residual_names
    )
    max_validation_residual = max(
        float(validation_df[name].abs().max()) for name in residual_names
    )

    # The costate-implied MPK must sit between the two branch slopes up to
    # kink slack; violations gate acceptance under kinked production.
    def mpk_violations(df):
        k = np.maximum(df["k"].to_numpy(), DOMAIN_EPS)
        m1 = A * a * k ** (a - 1.0)
        m2 = b_1 * m1
        implied = (
            np.where(k > k_thr, m2, m1) - df["euler"].to_numpy() / df["mu"].to_numpy()
        )
        lower = np.minimum(m1, m2)
        upper = np.maximum(m1, m2)
        return (
            float(np.maximum(lower - implied, 0.0).max()),
            float(np.maximum(implied - upper, 0.0).max()),
        )

    train_lower, train_upper = mpk_violations(train_df)
    validation_lower, validation_upper = mpk_violations(validation_df)
    p_lower_violation = max(train_lower, validation_lower)
    p_upper_violation = max(train_upper, validation_upper)

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, kernel, sigma, ell
        )
        k_test, c_test, _, _, _ = path_values(x, K_test, K_tilde_test, k_0)
        return k_test, c_test

    finite_positive = bool(
        all(
            np.isfinite(df[name]).all() and (df[name] > 0.0).all()
            for df in (train_df, validation_df, test_df)
            for name in ("k", "c", "mu")
        )
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
    if (
        use_kinked_production
        and max_validation_residual > tolerance_set.validation_residual
    ):
        rejection_reasons.append("validation_residual")
    if use_kinked_production and (
        p_lower_violation > tolerance_set.mpk_bound
        or p_upper_violation > tolerance_set.mpk_bound
    ):
        rejection_reasons.append("p_bound")

    # Published results.
    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "k_0": k_0,
        "A": A,
        "b_1": b_1,
        "b_2": b_2,
        "use_kinked_production": use_kinked_production,
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "solve_time": solve_time,
        "solver_status": solver_status,
        "solution_status": solution_status,
        **acceptance_fields(rejection_reasons),
    }

    if not use_kinked_production:
        benchmark = benchmark_df(
            [[1.0, k_0]], np.asarray(test_data), a, delta, rho_hat, sigma_crra
        )
        k_benchmark = benchmark["k"].to_numpy()
        c_benchmark = benchmark["c"].to_numpy()
        k_rel_error = np.abs(k_benchmark - test_df["k"].to_numpy()) / k_benchmark
        c_rel_error = np.abs(c_benchmark - test_df["c"].to_numpy()) / c_benchmark
        test_df["k_benchmark"] = k_benchmark
        test_df["c_benchmark"] = c_benchmark
        test_df["k_rel_error"] = k_rel_error
        test_df["c_rel_error"] = c_rel_error
        # min/max error statistics exclude the pinned t = 0 point ([1:]): its
        # near-zero error is an artifact of the initial condition, not accuracy.
        results.update(
            test_k_rel_error_mean=float(k_rel_error.mean()),
            test_c_rel_error_mean=float(c_rel_error.mean()),
            test_k_rel_error_max=float(k_rel_error[1:].max()),
            test_c_rel_error_max=float(c_rel_error[1:].max()),
            test_k_rel_error_min=float(k_rel_error[1:].min()),
            test_c_rel_error_min=float(c_rel_error[1:].min()),
        )
        print(
            f"solve_time(s) = {solve_time}, "
            f"E(|rel_error(k)|) = {results['test_k_rel_error_mean']:.3e}, "
            f"E(|rel_error(c)|) = {results['test_c_rel_error_mean']:.3e}"
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
            "A": A,
            "b_1": b_1,
            "b_2": b_2,
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
