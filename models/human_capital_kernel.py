from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import RidgeLevenbergMarquardt

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
    integrated_metric_factorization,
    rkhs_norm_squared,
)
from models.utilities import (
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
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


def human_capital_kernel(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11,
    k_0: float = 1.5,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    # Convergence requires gtol AND atol; the converged residual floors at
    # O(ridge * ||dual||), so atol sits above it.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.1,
        damping_increase=4.0,
        max_steps=100,
        atol=1e-9,
        gtol=1e-10,
    ),
    ridge_set: RidgeSettings = RidgeSettings(ridge=3e-11),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, dtype)
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0,
        data_set.train_T,
        data_set.validation_points,
        dtype=dtype,
    )
    n_train = len(train_data)

    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=7
    )
    h_0, c_0_init, initial_residual, _ = human_capital_initial_conditions(
        a_k, a_h, delta_k, delta_h, k_0, dtype=dtype
    )

    # Everything in p -- h_0 as its own leaf included -- is implicitly
    # differentiable through the solve (nlls-gram's implicit JVP/VJP at the
    # ridge stationary point).
    p = StructuralParameters.create(
        dtype=dtype,
        a_k=a_k,
        a_h=a_h,
        delta_k=delta_k,
        delta_h=delta_h,
        rho_hat=rho_hat,
        k_0=k_0,
        h_0=h_0,
    )
    residual_args = KernelResidualArgs(K=K, K_tilde=K_tilde)

    theta_0 = Theta(
        alpha_k=jnp.zeros(n_train, dtype=dtype),
        alpha_h=jnp.zeros(n_train, dtype=dtype),
        alpha_i_k=jnp.zeros(n_train, dtype=dtype),
        alpha_i_h=jnp.zeros(n_train, dtype=dtype),
        alpha_c=jnp.zeros(n_train, dtype=dtype),
        alpha_mu_k=jnp.zeros(n_train, dtype=dtype),
        alpha_mu_h=jnp.zeros(n_train, dtype=dtype),
        i_k_0=jnp.asarray(delta_k * k_0, dtype=dtype),
        i_h_0=jnp.asarray(delta_h * h_0, dtype=dtype),
        c_0=jnp.asarray(c_0_init, dtype=dtype),
        mu_k_0=jnp.asarray(1.0 / c_0_init, dtype=dtype),
        mu_h_0=jnp.asarray(1.0 / c_0_init, dtype=dtype),
    )
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        has_aux=True,
        metric=metric,
        ridge=ridge_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )

    solve_options = dict(
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.atol,
        max_steps_is_success=False,
    )

    timer = Stopwatch()
    result = solver.solve(theta_0, residual_args, p=p, **solve_options)
    elapsed = timer.elapsed(result)
    theta = result.x

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
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")

    kernel_solution = jax.jit(evaluate_grid)

    alpha_by_name = {
        "k": theta.alpha_k,
        "h": theta.alpha_h,
        "i_k": theta.alpha_i_k,
        "i_h": theta.alpha_i_h,
        "c": theta.alpha_c,
        "mu_k": theta.alpha_mu_k,
        "mu_h": theta.alpha_mu_h,
    }
    rkhs_norms = {
        name: float(rkhs_norm_squared(alpha, K))
        for name, alpha in alpha_by_name.items()
    }
    results = {
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
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
        "solve_time": elapsed,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        print(f"solve_time(s) = {elapsed}")

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
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": p,
            "residual_args": residual_args,
            "multi_start": None,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_kernel)
