from __future__ import annotations

from dataclasses import asdict, dataclass
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
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    DataSettings,
    StructuralParameters,
    mpk,
    production,
)
from models.utilities import (
    KernelResidualArgs,
    Stopwatch,
    UnoOptimizerSettings,
    acceptance_fields,
    gate_rejections,
    make_train_t_grid,
)


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# g_hat is the trend rate of the approximation class, None unless learned.
class Theta(NamedTuple):
    alpha_mu: jax.Array
    alpha_c: jax.Array
    alpha_k: jax.Array
    c_0: jax.Array
    mu_0: jax.Array
    g_hat: jax.Array | None = None


# CRRA first-order condition u'(c) = mu; the costate and resource residuals
# are utility-free. batch["g_hat"] is the fixed trend rate (0 without
# growth); a learned theta.g_hat overrides it.
def residual_fn(theta, args, p):
    g_hat = theta.g_hat if theta.g_hat is not None else args.batch["g_hat"]
    trend_mu = -p.sigma_crra * g_hat
    paths = integrated_kernel_paths(
        {
            "mu": IntegratedPath(theta.alpha_mu, theta.mu_0, trend_mu),
            "c": IntegratedPath(theta.alpha_c, theta.c_0, g_hat),
            "k": IntegratedPath(theta.alpha_k, p.k_0, g_hat),
        },
        args.K,
        args.K_tilde,
        args.batch["t"],
    )
    k, dk_dt = paths["k"]
    mu, dmu_dt = paths["mu"]
    c = paths["c"].value
    z = p.z_0 * args.batch["z_trend"]
    values = {
        "k": k,
        "c": c,
        "mu": mu,
        "dk_dt": dk_dt,
        "dmu_dt": dmu_dt,
        "resource": dk_dt - (production(z, k, p) - p.delta * k - c),
        "euler": dmu_dt + mu * (mpk(z, k, p) - p.delta - p.rho_hat),
        "shadow_price": mu * c**p.sigma_crra - 1.0,
    }
    residual = jnp.concatenate(
        [values["resource"], values["euler"], values["shadow_price"]]
    )
    return residual / jnp.sqrt(residual.size), values


@jax.jit
def objective(x, args):
    theta = args.unravel(x)
    return (
        rkhs_norm_squared(theta.alpha_mu, args.K)
        + rkhs_norm_squared(theta.alpha_c, args.K)
        + rkhs_norm_squared(theta.alpha_k, args.K)
    )


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


# CLI-reachable bars; all but log_tvc_threshold gate acceptance, and the mpk
# bound applies only under butterfly production.
@dataclass(frozen=True)
class ToleranceSettings:
    gate_train: bool = True
    train_residual: float = 1e-5
    gate_validation: bool = True
    validation_residual: float = 2e-4
    mpk_bound: float = 1e-3
    log_tvc_threshold: float = 0.0


# Fail-fast budget for butterfly production: near the basin threshold the NLP
# can wander, and the sweep classifies a slow solve as rejected.
BUTTERFLY_SOLVER_SETTINGS = UnoOptimizerSettings(max_iterations=25, time_limit=0.15)


def neoclassical_growth_kernel_uno(
    a: float = CONSTANTS["a"],
    delta: float = CONSTANTS["delta"],
    rho_hat: float = CONSTANTS["rho_hat"],
    sigma_crra: float = CONSTANTS["sigma_crra"],
    k_0: float = 1.0,
    # No trend machinery in this driver: technology is the constant z_0.
    z_0: float = 1.0,
    A: float = 1.0,
    b_1: float = 1.0,
    b_2: float = 0.0,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    solver_set: UnoOptimizerSettings | None = None,
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    # IPOPT works in float64 C arrays, so this solver is float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    use_butterfly_production = b_1 > 1.0
    k_thr = (
        (b_2 / (b_1 - 1.0)) ** (1.0 / a) if use_butterfly_production else float("inf")
    )
    if solver_set is None:
        solver_set = (
            BUTTERFLY_SOLVER_SETTINGS
            if use_butterfly_production
            else UnoOptimizerSettings()
        )

    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, jnp.float64)
    n_train = len(train_data)
    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)

    # Value container only: no p rides through IPOPT for implicit
    # differentiation.
    p = StructuralParameters.create(
        dtype=jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=z_0,
        k_0=k_0,
        g=0.0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        k_thr=k_thr,
    )

    c_0_init = production(p.z_0, p.k_0, p) - p.delta * p.k_0
    theta_0 = Theta(
        alpha_mu=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_c=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_k=jnp.zeros(n_train, dtype=jnp.float64),
        c_0=c_0_init,
        mu_0=c_0_init**-sigma_crra,
    )
    x_initial, unravel = ravel_pytree(theta_0)
    args = KernelResidualArgs(
        batch={
            "t": train_data,
            "z_trend": jnp.ones_like(train_data),
            "g_hat": jnp.zeros((), dtype=jnp.float64),
        },
        K=K,
        K_tilde=K_tilde,
        unravel=Partial(unravel),
    )

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
    c_0, mu_0 = theta.c_0, theta.mu_0

    # Paths and residual rows on an evaluation grid through fresh
    # cross-kernel matrices against the training grid.
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        values = residual_fn(
            theta,
            KernelResidualArgs(
                batch={
                    "t": points,
                    "z_trend": jnp.ones_like(points),
                    "g_hat": jnp.zeros((), dtype=jnp.float64),
                },
                K=K_eval,
                K_tilde=K_tilde_eval,
            ),
            p,
        )[1]
        return {"t": points, **values}

    test_data = jnp.linspace(
        0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )
    # Dense off-grid residual check between the training points.
    validation_data = jnp.linspace(
        0, data_set.train_T, data_set.validation_points, dtype=jnp.float64
    )
    test_benchmark = neoclassical_growth_benchmark(p)
    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    frame_columns = ("t", "k", "c", "mu", "resource", "euler", "shadow_price")
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_columns}
    )
    validation_df = pd.DataFrame(
        {name: np.asarray(validation_eval[name]) for name in frame_columns}
    )
    test_df = pd.DataFrame(
        {name: np.asarray(test_eval[name]) for name in frame_columns}
    )
    k_b, c_b = test_benchmark(test_data)
    test_df["k_benchmark"] = np.asarray(k_b)
    test_df["c_benchmark"] = np.asarray(c_b)
    test_df["k_rel_error"] = np.abs(test_df["k"] - test_df["k_benchmark"]) / np.abs(
        test_df["k_benchmark"]
    )
    test_df["c_rel_error"] = np.abs(test_df["c"] - test_df["c_benchmark"]) / np.abs(
        test_df["c_benchmark"]
    )

    max_train_residual = max(
        float(train_df["resource"].abs().max()),
        float(train_df["euler"].abs().max()),
        float(train_df["shadow_price"].abs().max()),
    )
    max_validation_residual = max(
        float(validation_df["resource"].abs().max()),
        float(validation_df["euler"].abs().max()),
        float(validation_df["shadow_price"].abs().max()),
    )
    mean_validation_residual = float(
        validation_df[list(("resource", "euler", "shadow_price"))]
        .abs()
        .to_numpy()
        .mean()
    )

    # The costate-implied MPK must sit between the two branch slopes up to
    # branch slack; violations gate acceptance under butterfly production.
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
        values = evaluate_grid(t_grid)
        return (
            values["k"],
            values["c"],
            values["mu"],
            values["dk_dt"],
            values["dmu_dt"],
        )

    tvc_t = jnp.asarray([data_set.test_tvc_T], dtype=jnp.float64)
    tvc_eval = evaluate_grid(tvc_t)
    k_T, c_T = tvc_eval["k"], tvc_eval["c"]
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
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    rejection_reasons += gate_rejections(
        tolerance_set, max_train_residual, mean_validation_residual
    )
    if use_butterfly_production and (
        p_lower_violation > tolerance_set.mpk_bound
        or p_upper_violation > tolerance_set.mpk_bound
    ):
        rejection_reasons.append("p_bound")

    results = {
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "tolerance_set": asdict(tolerance_set),
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "mean_validation_residual": mean_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "test_k_rel_error_abs": float(test_df["k_rel_error"].mean()),
        "test_c_rel_error_abs": float(test_df["c_rel_error"].mean()),
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "solve_time": solve_time,
        "solver_status": solver_status,
        "solution_status": solution_status,
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        print(
            f"solve_time(s) = {solve_time}, "
            f"E(|rel_error(k)|) = {results['test_k_rel_error_abs']:.3e}, "
            f"E(|rel_error(c)|) = {results['test_c_rel_error_abs']:.3e}"
        )
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_uno)
