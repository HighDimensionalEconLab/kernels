from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import unopy
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

from models.human_capital_shared import (
    RESIDUAL_NAMES,
    DataSettings,
    StructuralParameters,
    ToleranceSettings,
    human_capital_initial_conditions,
    production_terms,
)
from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    rkhs_norm_squared,
)
from models.utilities import (
    ACCEPTED_UNO_SOLUTION_STATUSES,
    KernelResidualArgs,
    Stopwatch,
    UnoOptimizerSettings,
    acceptance_fields,
    make_train_t_grid,
)


# Alpha blocks first (the metric penalizes the leading flat coordinates).
class Theta(NamedTuple):
    alpha_k: jax.Array
    alpha_h: jax.Array
    alpha_i_k: jax.Array
    alpha_i_h: jax.Array
    alpha_c: jax.Array
    alpha_mu_k: jax.Array
    alpha_mu_h: jax.Array
    i_k_0: jax.Array
    i_h_0: jax.Array
    c_0: jax.Array
    mu_k_0: jax.Array
    mu_h_0: jax.Array


@jax.jit
def objective(x, args):
    theta = args.unravel(x)
    return (
        rkhs_norm_squared(theta.alpha_k, args.K)
        + rkhs_norm_squared(theta.alpha_h, args.K)
        + rkhs_norm_squared(theta.alpha_mu_k, args.K)
        + rkhs_norm_squared(theta.alpha_mu_h, args.K)
        + rkhs_norm_squared(theta.alpha_i_k, args.K)
        + rkhs_norm_squared(theta.alpha_i_h, args.K)
        + rkhs_norm_squared(theta.alpha_c, args.K)
    )


# The 1/sqrt(count)-scaled residual rows are the IPOPT constraints, the same
# stack the NLLS driver solves.
def residual_fn(theta, args, p):
    delta_k, delta_h, rho_hat = p.delta_k, p.delta_h, p.rho_hat
    paths = integrated_kernel_paths(
        {
            "k": IntegratedPath(theta.alpha_k, p.k_0),
            "h": IntegratedPath(theta.alpha_h, p.h_0),
            "i_k": IntegratedPath(theta.alpha_i_k, theta.i_k_0),
            "i_h": IntegratedPath(theta.alpha_i_h, theta.i_h_0),
            "c": IntegratedPath(theta.alpha_c, theta.c_0),
            "mu_k": IntegratedPath(theta.alpha_mu_k, theta.mu_k_0),
            "mu_h": IntegratedPath(theta.alpha_mu_h, theta.mu_h_0),
        },
        args.K,
        args.K_tilde,
    )
    k, dk_dt = paths["k"]
    h, dh_dt = paths["h"]
    i_k, i_h, c = paths["i_k"].value, paths["i_h"].value, paths["c"].value
    mu_k, dmu_k_dt = paths["mu_k"]
    mu_h, dmu_h_dt = paths["mu_h"]
    output, f_k, f_h = production_terms(k, h, p)
    values = {
        "k": k,
        "h": h,
        "i_k": i_k,
        "i_h": i_h,
        "c": c,
        "mu_k": mu_k,
        "mu_h": mu_h,
        "physical_accumulation": dk_dt - (i_k - delta_k * k),
        "human_accumulation": dh_dt - (i_h - delta_h * h),
        "physical_euler": dmu_k_dt + mu_k * (f_k - delta_k - rho_hat),
        "human_euler": dmu_h_dt + mu_h * (f_h - delta_h - rho_hat),
        "feasibility": c + i_h + i_k - output,
        "shadow_price": mu_k * c - 1.0,
        "costate_gap": mu_k - mu_h,
        "hidden_dae_residual": (f_k - delta_k) - (f_h - delta_h),
        "Y": output,
    }
    residual = jnp.concatenate([values[name] for name in RESIDUAL_NAMES])
    return residual / jnp.sqrt(residual.size), values


@jax.jit
def constraints(x, args, p):
    return residual_fn(args.unravel(x), args, p)[0]


def lagrangian(x, objective_multiplier, multipliers, args, p):
    return objective_multiplier * objective(x, args) + jnp.dot(
        multipliers, constraints(x, args, p)
    )


# Module level: per-call grad/jacfwd/hessian closures would retrace on
# every solve.
objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))


def human_capital_kernel_uno(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11,
    k_0: float = 1.5,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    # Acceptance is wall-clock gated: filtersqp stops (and the solve is
    # rejected) when the limit is hit, so the budget must comfortably cover
    # the NLP size.
    solver_set: UnoOptimizerSettings = UnoOptimizerSettings(
        preset="filtersqp", time_limit=10.0
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    # IPOPT works in float64 C arrays, so this solver is float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, jnp.float64)
    test_data = jnp.linspace(
        0.0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )
    validation_data = jnp.linspace(
        0.0,
        data_set.train_T,
        data_set.validation_points,
        dtype=jnp.float64,
    )

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)

    h_0, c_0_init, initial_residual, _ = human_capital_initial_conditions(
        a_k, a_h, delta_k, delta_h, k_0
    )
    # Value container only: no p rides through IPOPT for implicit
    # differentiation.
    p = StructuralParameters.create(
        dtype=jnp.float64,
        a_k=a_k,
        a_h=a_h,
        delta_k=delta_k,
        delta_h=delta_h,
        rho_hat=rho_hat,
        k_0=k_0,
        h_0=h_0,
    )

    theta_0 = Theta(
        alpha_k=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_h=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_i_k=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_i_h=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_c=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_mu_k=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_mu_h=jnp.zeros(n_train, dtype=jnp.float64),
        i_k_0=jnp.asarray(delta_k * k_0, dtype=jnp.float64),
        i_h_0=jnp.asarray(delta_h * h_0, dtype=jnp.float64),
        c_0=jnp.asarray(c_0_init, dtype=jnp.float64),
        mu_k_0=jnp.asarray(1.0 / c_0_init, dtype=jnp.float64),
        mu_h_0=jnp.asarray(1.0 / c_0_init, dtype=jnp.float64),
    )
    x_initial, unravel = ravel_pytree(theta_0)
    args = KernelResidualArgs(K=K, K_tilde=K_tilde, unravel=Partial(unravel))

    n_variables = x_initial.size
    n_constraints = 7 * n_train
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
    if solver_set.primal_tolerance is not None:
        solver.set_option("primal_tolerance", solver_set.primal_tolerance)
    if solver_set.dual_tolerance is not None:
        solver.set_option("dual_tolerance", solver_set.dual_tolerance)

    timer = Stopwatch()
    result = solver.optimize(model)
    elapsed = timer.elapsed()

    # primal_solution appends solver-internal auxiliaries; only the leading
    # 7n + 5 block is ours.
    x = np.array(result.primal_solution, dtype=np.float64)[:n_variables]
    theta = unravel(jnp.asarray(x))
    rkhs_norms = {
        "k": float(rkhs_norm_squared(theta.alpha_k, K)),
        "h": float(rkhs_norm_squared(theta.alpha_h, K)),
        "i_k": float(rkhs_norm_squared(theta.alpha_i_k, K)),
        "i_h": float(rkhs_norm_squared(theta.alpha_i_h, K)),
        "c": float(rkhs_norm_squared(theta.alpha_c, K)),
        "mu_k": float(rkhs_norm_squared(theta.alpha_mu_k, K)),
        "mu_h": float(rkhs_norm_squared(theta.alpha_mu_h, K)),
    }

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        values = residual_fn(
            theta, KernelResidualArgs(K=K_eval, K_tilde=K_tilde_eval), p
        )[1]
        return {"t": points, **values}

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in RESIDUAL_NAMES
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in RESIDUAL_NAMES
    )
    positive_names = ("k", "h", "i_k", "i_h", "c", "mu_k", "mu_h")
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name])) and jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in positive_names
        )
    )

    kernel_solution = jax.jit(evaluate_grid)

    solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    rejection_reasons = []
    if solver_status != "SUCCESS":
        rejection_reasons.append("solver_status")
    if solution_status not in ACCEPTED_UNO_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")

    results = {
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "tolerance_set": asdict(tolerance_set),
        "c_0": float(theta.c_0),
        "i_k_0": float(theta.i_k_0),
        "i_h_0": float(theta.i_h_0),
        "mu_k_0": float(theta.mu_k_0),
        "mu_h_0": float(theta.mu_h_0),
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
    if verbose:
        print(f"solve_time(s) = {solve_time}")

    frame_columns = (
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
    )
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_columns}
    )
    test_df = pd.DataFrame(
        {name: np.asarray(test_eval[name]) for name in frame_columns}
    )
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_kernel_uno)
