from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import unopy
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

from models.kernels import (
    KernelSettings,
    integrated_kernel_matrices,
    rkhs_norm_squared,
)
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    DataSettings,
    StructuralParameters,
    Theta,
    evaluate_paths,
    residual_fn,
    snapshot,
    steady_flow_consumption_guess,
)
from models.utilities import Stopwatch, acceptance_fields, make_train_t_grid


@jax.jit
def objective(x, args):
    theta = args["unravel"](x)
    return (
        rkhs_norm_squared(theta.alpha_mu, args["K"])
        + rkhs_norm_squared(theta.alpha_c, args["K"])
        + rkhs_norm_squared(theta.alpha_k, args["K"])
    )


@jax.jit
def constraints(x, args, p):
    theta = args["unravel"](x)
    return residual_fn(theta, args, p)


def lagrangian(x, objective_multiplier, multipliers, args, p):
    return objective_multiplier * objective(x, args) + jnp.dot(
        multipliers, constraints(x, args, p)
    )


# Module level: per-call grad/jacfwd/hessian closures would retrace on
# every solve.
objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))


# CLI-reachable bars; all but log_tvc_threshold gate acceptance, and the mpk
# bound applies only under kinked production.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 5e-3
    mpk_bound: float = 1e-3
    log_tvc_threshold: float = 0.0


@dataclass(frozen=True)
class SolverSettings:
    preset: str = "ipopt"
    max_iterations: int | None = None
    time_limit: float | None = None


# Fail-fast budget for kinked production: near the basin threshold the NLP
# can wander, and the sweep classifies a slow solve as rejected.
KINKED_SOLVER_SETTINGS = SolverSettings(max_iterations=25, time_limit=0.15)


def neoclassical_growth_kernel_uno(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    k_0: float = 1.0,
    # No trend machinery in this driver: technology is the constant z_0.
    z_0: float = 1.0,
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
    # IPOPT works in float64 C arrays, so this solver is float64-only.
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

    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, jnp.float64)
    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)
    K_jax = (K + K.T) / 2
    K_tilde_jax = jnp.asarray(K_tilde)

    # Value container only: no p rides through IPOPT for implicit
    # differentiation.
    p = StructuralParameters.create(
        jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=z_0,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        k_thr=k_thr,
    )

    c_flow_guess = steady_flow_consumption_guess(z_0, k_0, a, delta, A, b_1, b_2)
    if c_0_guess is not None:
        c_flow_guess = float(c_0_guess)
    theta_0 = Theta(
        alpha_mu=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_c=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_k=jnp.zeros(n_train, dtype=jnp.float64),
        c_0=jnp.asarray(c_flow_guess),
        mu_0=jnp.asarray(c_flow_guess**-sigma_crra),
    )
    x_initial, unravel = ravel_pytree(theta_0)
    args = {
        "K": K_jax,
        "K_tilde": K_tilde_jax,
        "t": None,
        "z_trend": 1.0,
        "g_hat": None,
        "unravel": Partial(unravel),
    }

    n_variables = x_initial.size
    n_constraints = 3 * n_train
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
        return float(objective(jnp.asarray(x), args))

    def objective_gradient_callback(x, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(x), args))

    model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(x, constraint_output):
        constraint_output[:] = np.asarray(constraints(jnp.asarray(x), args, p))

    def jacobian_callback(x, jacobian_output):
        jacobian_output[:] = np.asarray(
            constraint_jacobian(jnp.asarray(x), args, p)
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
                args,
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
    model.set_initial_primal_iterate(np.asarray(x_initial))

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
    x = np.array(result.primal_solution, dtype=np.float64)[:n_variables]
    theta = unravel(jnp.asarray(x))
    c_0, mu_0 = theta.c_0, theta.mu_0

    test_data = jnp.linspace(
        0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )
    # Dense off-grid residual check between the training points.
    validation_data = jnp.linspace(
        0, data_set.train_T, max(data_set.test_points, 801), dtype=jnp.float64
    )
    test_benchmark = (
        None
        if use_kinked_production
        else neoclassical_growth_benchmark(
            a, delta, rho_hat, sigma_crra, g=0.0, z_0=z_0, k_0=k_0
        )
    )
    train_df = snapshot(theta, train_data, p, train_data, kernel_set, z=z_0)
    validation_df = snapshot(theta, validation_data, p, train_data, kernel_set, z=z_0)
    test_df = snapshot(
        theta,
        test_data,
        p,
        train_data,
        kernel_set,
        benchmark=test_benchmark,
        z=z_0,
    )

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
        k = df["k"].to_numpy()
        m1 = z_0 * A * a * k ** (a - 1.0)
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
    def kernel_solution(t_grid):
        paths = evaluate_paths(theta, t_grid, train_data, kernel_set, p)
        return (
            paths["k"].value,
            paths["c"].value,
            paths["mu"].value,
            paths["k"].d_dt,
            paths["mu"].d_dt,
        )

    tvc_t = jnp.asarray([data_set.test_tvc_T], dtype=jnp.float64)
    paths_T = evaluate_paths(theta, tvc_t, train_data, kernel_set, p)
    k_T, c_T = paths_T["k"].value, paths_T["c"].value
    log_tvc = float(
        -rho_hat * data_set.test_tvc_T - sigma_crra * jnp.log(c_T[0]) + jnp.log(k_T[0])
    )
    transversality_ok = bool(log_tvc < tolerance_set.log_tvc_threshold)

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
    if solution_status not in {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}:
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

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "z_0": z_0,
        "k_0": k_0,
        "A": A,
        "b_1": b_1,
        "b_2": b_2,
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
        "test_tvc_T": data_set.test_tvc_T,
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "solve_time": solve_time,
        **acceptance_fields(rejection_reasons),
    }

    if not use_kinked_production:
        k_rel_error = test_df["k_rel_error"].to_numpy()
        c_rel_error = test_df["c_rel_error"].to_numpy()
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
        "tolerance_set": asdict(tolerance_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "A": A,
            "b_1": b_1,
            "b_2": b_2,
        },
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_uno)
