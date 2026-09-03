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
from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.open_economy_growth_shared import (
    OPEN_ECONOMY_GROWTH_CONSTANTS as CONSTANTS,
)
from models.open_economy_growth_shared import (
    DataSettings,
    StructuralParameters,
    ToleranceSettings,
    capital_drift,
    current_account,
    investment_expenditure,
    present_value_gap,
    production,
    q_drift,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.utilities import (
    KernelResidualArgs,
    Stopwatch,
    UnoOptimizerSettings,
    acceptance_fields,
    gate_rejections,
    make_sampled_train_grid,
)


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# the trailing scalars are the free level anchors of the jump variables.
class Theta(NamedTuple):
    alpha_mu: jax.Array
    alpha_c: jax.Array
    alpha_k: jax.Array
    alpha_a: jax.Array
    alpha_q: jax.Array
    c_0: jax.Array
    mu_0: jax.Array
    q_0: jax.Array


# CRRA first-order condition u'(c) = mu; assets are the one signed path.
def residual_fn(theta, args, p):
    paths = integrated_kernel_paths(
        {
            "mu": IntegratedPath(theta.alpha_mu, theta.mu_0),
            "c": IntegratedPath(theta.alpha_c, theta.c_0),
            "k": IntegratedPath(theta.alpha_k, p.k_0),
            "a": IntegratedPath(theta.alpha_a, p.a_0),
            "q": IntegratedPath(theta.alpha_q, theta.q_0),
        },
        args.K,
        args.K_tilde,
    )
    k, dk_dt = paths["k"]
    a, da_dt = paths["a"]
    q, dq_dt = paths["q"]
    mu, dmu_dt = paths["mu"]
    c = paths["c"].value
    z = jnp.ones_like(k)
    values = {
        "k": k,
        "a": a,
        "q": q,
        "c": c,
        "mu": mu,
        "current_account": da_dt,
        "capital": dk_dt - capital_drift(k, q, p),
        "assets": da_dt - current_account(k, a, q, c, z, p),
        "tobin_q": dq_dt - q_drift(k, q, z, p),
        "marginal_wealth": dmu_dt - mu * (p.r - p.r_star),
        "consumption_foc": mu * c**p.gamma - 1.0,
    }
    residual = jnp.concatenate(
        [
            values["capital"],
            values["assets"],
            values["tobin_q"],
            values["marginal_wealth"],
            values["consumption_foc"],
        ]
    )
    return residual / jnp.sqrt(residual.size), values


@jax.jit
def objective(x, args):
    theta = args.unravel(x)
    return (
        rkhs_norm_squared(theta.alpha_mu, args.K)
        + rkhs_norm_squared(theta.alpha_c, args.K)
        + rkhs_norm_squared(theta.alpha_k, args.K)
        + rkhs_norm_squared(theta.alpha_a, args.K)
        + rkhs_norm_squared(theta.alpha_q, args.K)
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


def open_economy_growth_kernel_uno(
    alpha: float = CONSTANTS["alpha"],
    gamma: float = CONSTANTS["gamma"],
    delta: float = CONSTANTS["delta"],
    r: float = CONSTANTS["r"],
    eta: float = CONSTANTS["eta"],
    r_star: float = CONSTANTS["r_star"],
    k_0: float = 1.0,
    a_0: float = 0.0,
    kernel_set: KernelSettings = KernelSettings(ell=20.0),
    data_set: DataSettings = DataSettings(),
    solver_set: UnoOptimizerSettings = UnoOptimizerSettings(
        primal_tolerance=1e-4, dual_tolerance=1e-4
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    # IPOPT works in float64 C arrays, so this solver is float64-only.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_sampled_train_grid(data_set, jnp.float64)

    test_data = jnp.linspace(
        0.0, data_set.test_T, data_set.test_points, dtype=jnp.float64
    )
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=jnp.float64
    )
    n_train = len(train_data)
    # Value container only: no p rides through IPOPT for implicit
    # differentiation.
    p = StructuralParameters.create(
        dtype=jnp.float64,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        r=r,
        eta=eta,
        r_star=r_star,
        k_0=k_0,
        a_0=a_0,
    )
    steady = steady_state(p)
    output_scale = steady["output"]
    consumption_scale = steady["g"]

    K, K_tilde = integrated_kernel_matrices(train_data, train_data, kernel, sigma, ell)

    # Seeds zero the constant-path residual rows: at q_0 = 1/eta the
    # chi-normalized investment rate equals delta, and c = domestic
    # resources + bond income then zeros the seed current account.
    c_seed = production(p.k_0, 1.0, p) - p.delta * p.k_0 + p.r_star * p.a_0
    theta_0 = Theta(
        alpha_mu=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_c=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_k=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_a=jnp.zeros(n_train, dtype=jnp.float64),
        alpha_q=jnp.zeros(n_train, dtype=jnp.float64),
        c_0=c_seed,
        mu_0=c_seed**-p.gamma,
        q_0=1.0 / p.eta,
    )
    x_initial, unravel = ravel_pytree(theta_0)
    args = KernelResidualArgs(K=K, K_tilde=K_tilde, unravel=Partial(unravel))

    n_variables = x_initial.size
    n_constraints = 5 * n_train
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
    # primal block is the solution.
    x = np.array(result.primal_solution, dtype=np.float64)[:n_variables]
    theta = unravel(jnp.asarray(x))

    # Paths, model quantities, and DAE residuals on an evaluation grid
    # through fresh cross-kernel matrices against the training grid.
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        values = residual_fn(
            theta, KernelResidualArgs(K=K_eval, K_tilde=K_tilde_eval), p
        )[1]
        return {
            "t": points,
            **values,
            "x": investment_expenditure(values["k"], values["q"], p),
            "output": production(values["k"], jnp.ones_like(values["k"]), p),
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    tail = tail_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        p,
    )
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval["capital"]))),
        float(jnp.max(jnp.abs(train_eval["assets"]))),
        float(jnp.max(jnp.abs(train_eval["tobin_q"]))),
        float(jnp.max(jnp.abs(train_eval["marginal_wealth"]))),
        float(jnp.max(jnp.abs(train_eval["consumption_foc"]))),
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval["capital"]))),
        float(jnp.max(jnp.abs(validation_eval["assets"]))),
        float(jnp.max(jnp.abs(validation_eval["tobin_q"]))),
        float(jnp.max(jnp.abs(validation_eval["marginal_wealth"]))),
        float(jnp.max(jnp.abs(validation_eval["consumption_foc"]))),
    )
    mean_validation_residual = float(
        jnp.mean(
            jnp.abs(
                jnp.concatenate(
                    [
                        jnp.ravel(validation_eval[name])
                        for name in (
                            "capital",
                            "assets",
                            "tobin_q",
                            "marginal_wealth",
                            "consumption_foc",
                        )
                    ]
                )
            )
        )
    )

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(p)
    benchmark_values = benchmark(np.asarray(test_data))
    k_relative_error = jnp.abs(
        test_eval["k"] - jnp.asarray(benchmark_values["k"])
    ) / jnp.asarray(benchmark_values["k"])
    q_relative_error = jnp.abs(
        test_eval["q"] - jnp.asarray(benchmark_values["q"])
    ) / jnp.asarray(benchmark_values["q"])
    c_relative_error = jnp.abs(
        test_eval["c"] - jnp.asarray(benchmark_values["c"])
    ) / jnp.asarray(benchmark_values["c"])
    a_scaled_error = (
        jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
    )
    max_benchmark_error = max(
        float(jnp.max(k_relative_error)),
        float(jnp.max(q_relative_error)),
        float(jnp.max(c_relative_error)),
        float(jnp.max(a_scaled_error)),
    )

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name])) and jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "mu", "x")
        )
        and all(
            jnp.all(jnp.isfinite(grid["a"]))
            for grid in (train_eval, validation_eval, test_eval)
        )
    )
    pv_gap = present_value_gap(
        validation_eval["t"],
        validation_eval["k"],
        validation_eval["q"],
        validation_eval["c"],
        p,
    )
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        train_eval["t"][-1],
        p,
    )

    solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    validation_pass = max_validation_residual <= tolerance_set.validation_residual
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    rejection_reasons += gate_rejections(
        tolerance_set, max_train_residual, mean_validation_residual
    )

    rkhs_norm_mu = rkhs_norm_squared(theta.alpha_mu, K)
    rkhs_norm_c = rkhs_norm_squared(theta.alpha_c, K)
    rkhs_norm_k = rkhs_norm_squared(theta.alpha_k, K)
    rkhs_norm_a = rkhs_norm_squared(theta.alpha_a, K)
    rkhs_norm_q = rkhs_norm_squared(theta.alpha_q, K)
    rkhs_norm_total = (
        rkhs_norm_mu + rkhs_norm_c + rkhs_norm_k + rkhs_norm_a + rkhs_norm_q
    )

    @jax.jit
    def kernel_solution(points):
        values = evaluate_grid(points)
        return values["k"], values["a"], values["q"], values["c"]

    results = {
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "solver_set": asdict(solver_set),
        "tolerance_set": asdict(tolerance_set),
        "k_ss": float(steady["k"]),
        "q_ss": float(steady["q"]),
        "output_ss": float(output_scale),
        "g_ss": float(consumption_scale),
        "c_0": float(theta.c_0),
        "mu_0": float(theta.mu_0),
        "q_0": float(theta.q_0),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "mean_validation_residual": mean_validation_residual,
        "validation_pass": validation_pass,
        "max_benchmark_error": max_benchmark_error,
        "differential_residual_count": 4 * n_train,
        "algebraic_residual_count": n_train,
        "dae_residual_count": 5 * n_train,
        "unknown_count": 5 * n_train + 3,
        "dynamic_rkhs_function_count": 4,
        "algebraic_rkhs_function_count": 1,
        "free_initial_value_count": 3,
        "metric_selection_dimension": 3,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "ex_post_present_value_gap": float(jnp.abs(pv_gap)),
        "ex_post_current_account": float(jnp.abs(train_eval["current_account"][-1])),
        "ex_post_stable_gap": abs(float(tail["stable"])),
        "ex_post_present_value_tail_gap": abs(float(tail["present_value"])),
        "ex_post_discounted_assets": float(tvc["discounted_assets"]),
        "ex_post_asset_tvc": float(tvc["asset_tvc"]),
        "ex_post_capital_tvc": float(tvc["capital_tvc"]),
        "benchmark_present_value_residual": abs(
            float(benchmark_diagnostics["present_value_residual"])
        ),
        "max_marginal_wealth_residual": float(
            jnp.max(jnp.abs(train_eval["marginal_wealth"]))
        ),
        "max_consumption_foc_residual": float(
            jnp.max(jnp.abs(train_eval["consumption_foc"]))
        ),
        "max_consumption_variation": float(
            jnp.max(jnp.abs(train_eval["c"] - train_eval["c"][0]))
        ),
        "rkhs_norm_mu": float(rkhs_norm_mu),
        "rkhs_norm_c": float(rkhs_norm_c),
        "rkhs_norm_k": float(rkhs_norm_k),
        "rkhs_norm_a": float(rkhs_norm_a),
        "rkhs_norm_q": float(rkhs_norm_q),
        "rkhs_norm_total": float(rkhs_norm_total),
        "solve_time": solve_time,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        **acceptance_fields(rejection_reasons),
    }

    frame_names = (
        "t",
        "k",
        "a",
        "q",
        "c",
        "mu",
        "x",
        "output",
        "current_account",
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth",
        "consumption_foc",
    )
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_names}
    )
    test_df = pd.DataFrame({name: np.asarray(test_eval[name]) for name in frame_names})
    for name in ("k", "a", "q", "c", "x", "output", "current_account"):
        test_df[f"{name}_benchmark"] = np.asarray(benchmark_values[name])
    test_df["k_relative_error"] = np.asarray(k_relative_error)
    test_df["q_relative_error"] = np.asarray(q_relative_error)
    test_df["c_relative_error"] = np.asarray(c_relative_error)
    test_df["a_scaled_error"] = np.asarray(a_scaled_error)
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_kernel_uno)
