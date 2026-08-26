from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    RidgeLevenbergMarquardt,
)

from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    integrated_metric_factorization,
    rkhs_norm_squared,
)
from models.optimal_advertising_shared import (
    DataSettings,
    StructuralParameters,
    ToleranceSettings,
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
    alpha_x: jax.Array
    alpha_mu: jax.Array
    alpha_u: jax.Array
    mu_0: jax.Array
    u_0: jax.Array


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
    residual = jnp.concatenate(
        [
            dx_dt - ((1.0 - x) * u - beta * x),
            dmu_dt - (-gamma + (rho_hat + beta) * mu + mu * u),
            u**control_power - kappa * mu * (1.0 - x),
        ]
    )
    return residual / jnp.sqrt(residual.size)


def optimal_advertising_kernel(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(ell=15.0),
    data_set: DataSettings = DataSettings(),
    # Conjunctive stopping: gtol bounds the ridge stationarity
    # ||J'r + ridge K alpha|| (calibrated per the
    # ~1e-3 * ridge * info.penalty_grad_norm recipe: ridge*||L'L theta|| is
    # ~6e-12 at convergence), atol additionally demands the scaled residual
    # norm solved. The converged residual floor is O(ridge * ||dual||), so
    # atol must sit above it.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.1,
        damping_increase=4.0,
        max_steps=50,
        atol=1e-9,
        gtol=1e-10,
    ),
    # The ridge keeps the minimizer O(ridge) from the pure RKHS-seminorm
    # interpolant with the free (mu_0, u_0) unpenalized (no epsilon shift
    # anywhere).
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
        max(data_set.test_points, data_set.validation_points),
        dtype=dtype,
    )
    n_train = len(train_data)

    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=3
    )
    gamma = (beta + rho_hat) / c

    # Everything in p is implicitly differentiable through the solve
    # (nlls-gram's implicit JVP/VJP at the ridge stationary point).
    p = StructuralParameters.create(
        dtype=dtype, rho_hat=rho_hat, c=c, beta=beta, kappa=kappa, x_0=x_0
    )
    residual_args = KernelResidualArgs(K=K, K_tilde=K_tilde)

    u_0_guess = beta * x_0 / (1.0 - x_0)
    mu_0_guess = gamma / (rho_hat + beta + u_0_guess)
    theta_0 = Theta(
        alpha_x=jnp.zeros(n_train, dtype=dtype),
        alpha_mu=jnp.zeros(n_train, dtype=dtype),
        alpha_u=jnp.zeros(n_train, dtype=dtype),
        mu_0=jnp.asarray(mu_0_guess, dtype=dtype),
        u_0=jnp.asarray(u_0_guess, dtype=dtype),
    )
    solver = RidgeLevenbergMarquardt(
        residual_fn,
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
    alpha_x, alpha_mu, alpha_u = theta.alpha_x, theta.alpha_mu, theta.alpha_u
    mu_0, u_0 = theta.mu_0, theta.u_0

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        paths = integrated_kernel_paths(
            {
                "x": IntegratedPath(theta.alpha_x, p.x_0),
                "mu": IntegratedPath(theta.alpha_mu, theta.mu_0),
                "u": IntegratedPath(theta.alpha_u, theta.u_0),
            },
            K_eval,
            K_tilde_eval,
        )
        rho_hat, c, beta, kappa = p.rho_hat, p.c, p.beta, p.kappa
        gamma = (beta + rho_hat) / c
        control_power = (1.0 - kappa) / kappa
        x, dx_dt = paths["x"]
        mu, dmu_dt = paths["mu"]
        u = paths["u"].value
        return {
            "t": points,
            "x": x,
            "mu": mu,
            "u": u,
            "state": dx_dt - ((1.0 - x) * u - beta * x),
            "costate": dmu_dt - (-gamma + (rho_hat + beta) * mu + mu * u),
            "shadow_price": u**control_power - kappa * mu * (1.0 - x),
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    residual_names = ("state", "costate", "shadow_price")
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )
    finite_valid = bool(
        all(
            jnp.all(jnp.isfinite(grid[name]))
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("x", "mu", "u")
        )
        and all(
            jnp.min(grid["u"]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
        )
    )
    train_residual_tolerance = max(
        tolerance_set.train_residual,
        8.0 * float(jnp.finfo(dtype).eps),
    )
    rejection_reasons = []
    if not finite_valid:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > train_residual_tolerance:
        rejection_reasons.append("train_residual")

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        paths = integrated_kernel_paths(
            {
                "x": IntegratedPath(theta.alpha_x, p.x_0),
                "mu": IntegratedPath(theta.alpha_mu, theta.mu_0),
                "u": IntegratedPath(theta.alpha_u, theta.u_0),
            },
            K_eval,
            K_tilde_eval,
        )
        return paths["x"].value, paths["mu"].value, paths["u"].value

    rkhs_norms = {
        "x": rkhs_norm_squared(alpha_x, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
        "u": rkhs_norm_squared(alpha_u, K),
    }
    results = {
        "parameters": {**p.to_dict(), "gamma": gamma},
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
        "tolerance_set": asdict(tolerance_set),
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "mu_0": float(mu_0),
        "u_0": float(u_0),
        **{f"rkhs_norm_{name}": float(value) for name, value in rkhs_norms.items()},
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": train_residual_tolerance,
        "max_validation_residual": max_validation_residual,
        "solve_time": elapsed,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        print(f"solve_time(s) = {elapsed}")

    frame_columns = ("t", "x", "mu", "u", "state", "costate", "shadow_price")
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
    jsonargparse.CLI(optimal_advertising_kernel)
