from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    LevenbergMarquardt,
    LMStatus,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_with_compute_dtype,
    repeated_blockdiag_metric,
)

from kernels import integrated_kernel_matrices, normalize_kernel

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-8
DOMAIN_EPS = 1e-8
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern12"
    ell: float = 15.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100
    validation_points: int = 300


@dataclass(frozen=True)
class NLLSGramSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 50
    residual_atol: float = 1e-10
    linear_solver: str = "cholesky"
    metric_eps: float = 1e-7


def optimal_advertising_kernel_nlls(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
    verbose: bool = False,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(
            0.0, data_set.train_T, data_set.train_points, dtype=dtype
        )
    else:
        train_data = jnp.sort(jnp.asarray(data_set.train_points_list, dtype=dtype))
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0,
        data_set.train_T,
        max(data_set.test_points, data_set.validation_points),
        dtype=dtype,
    )
    n_train = len(train_data)
    eps = lm_set.metric_eps
    if eps <= 0.0:
        raise ValueError("metric_eps must be positive")

    metric_data = train_data.astype(jnp.float64)
    K_metric, K_tilde_metric = integrated_kernel_matrices(
        metric_data, metric_data, kernel, sigma, ell
    )
    K_metric = (K_metric + K_metric.T) / 2
    K = K_metric.astype(dtype)
    K_tilde = K_tilde_metric.astype(dtype)
    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K_metric + eps * jnp.eye(n_train, dtype=jnp.float64))
    )

    gamma = (beta + rho_hat) / c
    control_power = (1.0 - kappa) / kappa

    def path_values(theta, K_eval, K_tilde_eval):
        alpha_x, alpha_mu, alpha_u, initial = theta
        mu_0, u_0 = initial
        x = x_0 + K_tilde_eval @ alpha_x
        mu = mu_0 + K_tilde_eval @ alpha_mu
        u = u_0 + K_tilde_eval @ alpha_u
        dx_dt = K_eval @ alpha_x
        dmu_dt = K_eval @ alpha_mu
        return x, mu, u, dx_dt, dmu_dt

    def residuals(values):
        x, mu, u, dx_dt, dmu_dt = values
        state = dx_dt - ((1.0 - x) * u - beta * x)
        costate = dmu_dt - (-gamma + (rho_hat + beta) * mu + mu * u)
        shadow_price = u**control_power - kappa * mu * (1.0 - x)
        return state, costate, shadow_price

    residual_scale = 1.0 / jnp.sqrt(3.0 * n_train)

    def residual_fn(theta):
        return residual_scale * jnp.concatenate(
            residuals(path_values(theta, K, K_tilde))
        )

    metric = metric_with_compute_dtype(
        repeated_blockdiag_metric(
            alpha_metric,
            n_train,
            3,
            additional=(
                metric_from_diagonal(jnp.full(2, eps, dtype=jnp.float64)),
                2,
            ),
        ),
        jnp.float64,
    )
    u_0_guess = max(beta * x_0 / max(1.0 - x_0, DOMAIN_EPS), DOMAIN_EPS)
    mu_0_guess = max(gamma / (rho_hat + beta + u_0_guess), DOMAIN_EPS)
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.asarray([mu_0_guess, u_0_guess], dtype=dtype),
    )
    solver = LevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        dual_solve_dtype=(jnp.float64 if lm_set.linear_solver == "cholesky" else None),
        metric=metric,
    )

    compile_start = time.perf_counter()
    jax.block_until_ready(solver.solve(theta_0, max_steps=1, atol=lm_set.residual_atol))
    compile_time = time.perf_counter() - compile_start

    start = time.perf_counter()
    result = solver.solve(
        theta_0, max_steps=lm_set.max_steps, atol=lm_set.residual_atol
    )
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    theta = result.x
    alpha_x, alpha_mu, alpha_u, initial = theta
    mu_0, u_0 = initial

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        values = path_values(theta, K_eval, K_tilde_eval)
        state, costate, shadow_price = residuals(values)
        x, mu, u, *_ = values
        return {
            "t": points,
            "x": x,
            "mu": mu,
            "u": u,
            "state": state,
            "costate": costate,
            "shadow_price": shadow_price,
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
        TRAIN_RESIDUAL_TOL,
        8.0 * float(jnp.finfo(dtype).eps),
    )
    rejection_reasons = []
    if not finite_valid:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > train_residual_tolerance:
        rejection_reasons.append("train_residual")
    solution_accepted = not rejection_reasons

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        x, mu, u, *_ = path_values(theta, K_eval, K_tilde_eval)
        return x, mu, u

    rkhs_norms = {
        "x": alpha_metric.norm(alpha_x) ** 2 - eps * (alpha_x @ alpha_x),
        "mu": alpha_metric.norm(alpha_mu) ** 2 - eps * (alpha_mu @ alpha_mu),
        "u": alpha_metric.norm(alpha_u) ** 2 - eps * (alpha_u @ alpha_u),
    }
    results = {
        "rho_hat": rho_hat,
        "c": c,
        "beta": beta,
        "kappa": kappa,
        "x_0": x_0,
        "gamma": gamma,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": "float64",
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "mu_0": float(mu_0),
        "u_0": float(u_0),
        **{f"rkhs_norm_{name}": float(value) for name, value in rkhs_norms.items()},
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": train_residual_tolerance,
        "max_validation_residual": max_validation_residual,
        "solve_time": elapsed,
        "wall_time": elapsed,
        "solver": "nlls-gram",
        "linear_solver_resolved": lm_set.linear_solver,
        "metric_eps": eps,
        "train_loss": float(result.info.loss),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "compile_time": compile_time,
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted" if solution_accepted else ",".join(rejection_reasons)
        ),
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
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "parameters": {
            "rho_hat": rho_hat,
            "c": c,
            "beta": beta,
            "kappa": kappa,
            "gamma": gamma,
        },
        "initial_conditions": {"x_0": x_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(optimal_advertising_kernel_nlls)
