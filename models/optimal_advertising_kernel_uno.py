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

from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    rkhs_norm_squared,
)
from models.optimal_advertising_shared import (
    DataSettings,
    StructuralParameters,
    ToleranceSettings,
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
    alpha_x: jax.Array
    alpha_mu: jax.Array
    alpha_u: jax.Array
    mu_0: jax.Array
    u_0: jax.Array


@jax.jit
def objective(x, args):
    theta = args.unravel(x)
    return (
        rkhs_norm_squared(theta.alpha_x, args.K)
        + rkhs_norm_squared(theta.alpha_mu, args.K)
        + rkhs_norm_squared(theta.alpha_u, args.K)
    )


# The 1/sqrt(count)-scaled residual rows are the IPOPT constraints, the same
# stack the NLLS driver solves.
RESIDUAL_NAMES = ("state", "costate", "shadow_price")


def residual_fn(theta, args, p):
    rho_hat, c, beta, kappa = p.rho_hat, p.c, p.beta, p.kappa
    gamma = (beta + rho_hat) / c
    control_power = (1.0 - kappa) / kappa
    paths = integrated_kernel_paths(
        {
            "x": IntegratedPath(theta.alpha_x, p.x_0),
            "mu": IntegratedPath(theta.alpha_mu, theta.mu_0),
            "u": IntegratedPath(theta.alpha_u, theta.u_0),
        },
        args.K,
        args.K_tilde,
    )
    x, dx_dt = paths["x"]
    mu, dmu_dt = paths["mu"]
    u = paths["u"].value
    values = {
        "x": x,
        "mu": mu,
        "u": u,
        "state": dx_dt - ((1.0 - x) * u - beta * x),
        "costate": dmu_dt - (-gamma + (rho_hat + beta) * mu + mu * u),
        "shadow_price": u**control_power - kappa * mu * (1.0 - x),
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


def optimal_advertising_kernel_uno(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    kernel_set: KernelSettings = KernelSettings(ell=15.0),
    data_set: DataSettings = DataSettings(),
    solver_set: UnoOptimizerSettings = UnoOptimizerSettings(),
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

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)

    gamma = (beta + rho_hat) / c
    p = StructuralParameters.create(
        dtype=jnp.float64, rho_hat=rho_hat, c=c, beta=beta, kappa=kappa, x_0=x_0
    )
    u_0_guess = beta * x_0 / (1.0 - x_0)
    mu_0_guess = gamma / (rho_hat + beta + u_0_guess)
    theta_0 = Theta(
        alpha_x=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_mu=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_u=jnp.zeros(n_train, dtype=jnp.float64),
        mu_0=jnp.asarray(mu_0_guess, dtype=jnp.float64),
        u_0=jnp.asarray(u_0_guess, dtype=jnp.float64),
    )
    x_initial, unravel = ravel_pytree(theta_0)
    args = KernelResidualArgs(K=K, K_tilde=K_tilde, unravel=Partial(unravel))

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
    if solver_set.primal_tolerance is not None:
        solver.set_option("primal_tolerance", solver_set.primal_tolerance)
    if solver_set.dual_tolerance is not None:
        solver.set_option("dual_tolerance", solver_set.dual_tolerance)

    timer = Stopwatch()
    result = solver.optimize(model)
    elapsed = timer.elapsed()

    # primal_solution appends solver-internal auxiliaries; only the leading
    # 3n + 2 block is ours.
    x = np.array(result.primal_solution, dtype=np.float64)[:n_variables]
    theta = unravel(jnp.asarray(x))

    train_values = residual_fn(theta, args, p)[1]
    x_train = train_values["x"]
    mu_train = train_values["mu"]
    u_train = train_values["u"]
    state_res = train_values["state"]
    costate_res = train_values["costate"]
    shadow_res = train_values["shadow_price"]
    max_train_residual = max(
        float(jnp.max(jnp.abs(residual)))
        for residual in (state_res, costate_res, shadow_res)
    )

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        values = residual_fn(
            theta, KernelResidualArgs(K=K_eval, K_tilde=K_tilde_eval), p
        )[1]
        return {"t": points, **values}

    test_values = kernel_solution(test_data)
    x_test, mu_test, u_test = test_values["x"], test_values["mu"], test_values["u"]

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
    if solution_status not in ACCEPTED_UNO_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite_valid:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")

    results = {
        "parameters": {**p.to_dict(), "gamma": gamma},
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "tolerance_set": asdict(tolerance_set),
        "mu_0": float(theta.mu_0),
        "u_0": float(theta.u_0),
        "rkhs_norm_x": float(rkhs_norm_squared(theta.alpha_x, K)),
        "rkhs_norm_mu": float(rkhs_norm_squared(theta.alpha_mu, K)),
        "rkhs_norm_u": float(rkhs_norm_squared(theta.alpha_u, K)),
        "max_train_residual": max_train_residual,
        "solve_time": solve_time,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
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
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    jsonargparse.CLI(optimal_advertising_kernel_uno)
