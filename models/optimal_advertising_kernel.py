from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    RidgeLevenbergMarquardt,
)

from models.kernels import (
    integrated_kernel_matrices,
    integrated_metric_factorization,
    normalize_kernel,
    rkhs_norm_squared,
)
from models.utilities import (
    STATUS_TO_REASON,
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)

DOMAIN_EPS = 1e-8


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


# Acceptance gates, CLI-reachable so a float32 caller can pass its own bar
# (the float32 replication driver does).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-8


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 50
    # Conjunctive stopping: gtol bounds the ridge stationarity
    # ||J'r + ridge K alpha|| (calibrated per the
    # ~1e-3 * ridge * info.penalty_grad_norm recipe: ridge*||L'L theta|| is
    # ~6e-12 at convergence), residual_atol additionally demands the scaled
    # residual norm solved. The converged residual floor is
    # O(ridge * ||dual||), so residual_atol must sit above it; the ridge
    # keeps the minimizer O(ridge) from the pure RKHS-seminorm interpolant
    # with the free (mu_0, u_0) unpenalized (no epsilon shift anywhere).
    residual_atol: float = 1e-9
    ridge: float = 3e-11
    gtol: float = 1e-10


def optimal_advertising_kernel(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = normalize_kernel(kernel_set.kernel)
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

    u_0_guess = max(beta * x_0 / max(1.0 - x_0, DOMAIN_EPS), DOMAIN_EPS)
    mu_0_guess = max(gamma / (rho_hat + beta + u_0_guess), DOMAIN_EPS)
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.asarray([mu_0_guess, u_0_guess], dtype=dtype),
    )
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=lm_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )

    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )
    elapsed = timer.elapsed(result)
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
        x, mu, u, *_ = path_values(theta, K_eval, K_tilde_eval)
        return x, mu, u

    rkhs_norms = {
        "x": rkhs_norm_squared(alpha_x, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
        "u": rkhs_norm_squared(alpha_u, K),
    }
    results = {
        "rho_hat": rho_hat,
        "c": c,
        "beta": beta,
        "kappa": kappa,
        "x_0": x_0,
        "gamma": gamma,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["state"].dtype),
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
        "linear_solver": "cholesky",
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
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
    jsonargparse.CLI(optimal_advertising_kernel)
