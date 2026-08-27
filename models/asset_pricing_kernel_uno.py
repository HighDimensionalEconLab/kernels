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

from models.asset_pricing_shared import (
    DataSettings,
    StructuralParameters,
    ToleranceSettings,
    dividends,
    mu_f_array,
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


# Alpha first (the metric penalizes the leading flat coordinates).
class Theta(NamedTuple):
    alpha: jax.Array
    p_0: jax.Array


# No-arbitrage residual dp/dt = r p - x along the dividend path.
def residual_fn(theta, args, p):
    paths = integrated_kernel_paths(
        {"price": IntegratedPath(theta.alpha, theta.p_0)}, args.K, args.K_tilde
    )
    price, dprice_dt = paths["price"]
    residual = dprice_dt - (p.r * price - dividends(args.batch["t"], p))
    return residual, {"p": price, "dp_dt": dprice_dt, "residual": residual}


@jax.jit
def objective(x, args):
    theta = args.unravel(x)
    return rkhs_norm_squared(theta.alpha, args.K)


@jax.jit
def constraints(x, args, p):
    theta = args.unravel(x)
    return residual_fn(theta, args, p)[0]


def lagrangian(x, objective_multiplier, multipliers, args, p):
    return objective_multiplier * objective(x, args) + jnp.dot(
        multipliers, constraints(x, args, p)
    )


# Module level: per-call grad/jacfwd/hessian closures would retrace on
# every solve.
objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))


def asset_pricing_kernel_uno(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    kernel_set: KernelSettings = KernelSettings(),
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
        0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )

    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)
    K_jax = (K + K.T) / 2
    K_tilde_jax = jnp.asarray(K_tilde)

    p = StructuralParameters.create(dtype=jnp.float64, r=r, c=c, g=g, x_0=x_0)

    theta_0 = Theta(
        alpha=jnp.zeros(n_train, dtype=jnp.float64),
        p_0=jnp.zeros((), dtype=jnp.float64),
    )
    x_initial, unravel = ravel_pytree(theta_0)
    args = KernelResidualArgs(
        batch={"t": train_data},
        K=K_jax,
        K_tilde=K_tilde_jax,
        unravel=Partial(unravel),
    )

    n_variables = x_initial.size
    n_constraints = n_train
    variable_lower_bounds = np.full(n_variables, -np.inf, dtype=np.float64)
    variable_upper_bounds = np.full(n_variables, np.inf, dtype=np.float64)
    constraint_lower_bounds = np.zeros(n_constraints, dtype=np.float64)
    constraint_upper_bounds = np.zeros(n_constraints, dtype=np.float64)

    model = unopy.Model(
        unopy.PROBLEM_QUADRATIC,
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
    # n + 1 block is ours.
    x = np.array(result.primal_solution, dtype=np.float64)[:n_variables]
    theta = unravel(jnp.asarray(x))
    alpha, p_0 = theta.alpha, theta.p_0
    train_values = residual_fn(theta, args, p)[1]
    p_train = train_values["p"]
    train_residual = train_values["residual"]
    max_train_residual = float(jnp.max(jnp.abs(train_residual)))

    # Paths and the no-arbitrage residual on an evaluation grid through fresh
    # cross-kernel matrices against the training grid.
    @jax.jit
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        return residual_fn(
            theta,
            KernelResidualArgs(batch={"t": points}, K=K_eval, K_tilde=K_tilde_eval),
            p,
        )[1]

    test_values = evaluate_grid(test_data)
    p_test = test_values["p"]
    test_residual = test_values["residual"]
    p_benchmark = mu_f_array(test_data, p)
    p_rel_error = jnp.abs(p_benchmark - p_test) / p_benchmark

    solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    finite = bool(jnp.all(jnp.isfinite(p_test)) and jnp.all(jnp.isfinite(p_benchmark)))
    rejection_reasons = []
    if solver_status != "SUCCESS":
        rejection_reasons.append("solver_status")
    if solution_status not in ACCEPTED_UNO_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite:
        rejection_reasons.append("nonfinite")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    if float(p_0) < 0.0:
        rejection_reasons.append("negative_p_0")

    results = {
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "tolerance_set": asdict(tolerance_set),
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "p_0": float(p_0),
        "rkhs_norm_p": float(rkhs_norm_squared(alpha, K_jax)),
        "max_train_residual": max_train_residual,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
        "solve_time": solve_time,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        print(
            f"solve_time(s) = {solve_time}, "
            f"E(|rel_error(p)|) = {results['p_rel_error_mean']:.3e}"
        )

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
    return results, {
        "model": evaluate_grid,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_kernel_uno)
