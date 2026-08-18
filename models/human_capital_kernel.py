from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    LevenbergMarquardt,
    RidgeLevenbergMarquardt,
)

from models.kernels import (
    integrated_kernel_matrices,
    integrated_metric_factorization,
    normalize_kernel,
)
from models.utilities import (
    STATUS_TO_REASON,
    SolveSpec,
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)


def human_capital_initial_residual(params, batch):
    a_k, a_h, delta_k, delta_h, k_0 = batch
    h_0 = jnp.exp(params["log_h"])
    c_0 = jnp.exp(params["log_c"])
    output = (k_0**a_k) * (h_0**a_h)
    marginal_k = a_k * (k_0 ** (a_k - 1.0)) * (h_0**a_h)
    marginal_h = a_h * (k_0**a_k) * (h_0 ** (a_h - 1.0))
    return jnp.array(
        [
            marginal_h - delta_h - (marginal_k - delta_k),
            c_0 + delta_h * h_0 + delta_k * k_0 - output,
        ],
        dtype=h_0.dtype,
    )


def human_capital_initial_conditions(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    k_0: float = 1.5,
    iterations: int = 20,
    dtype=jnp.float64,
):
    batch = tuple(
        jnp.asarray(value, dtype=dtype) for value in (a_k, a_h, delta_k, delta_h, k_0)
    )
    k_0_jax = batch[-1]
    c_guess = (
        (k_0_jax ** batch[0]) * (k_0_jax ** batch[1])
        - batch[3] * k_0_jax
        - batch[2] * k_0_jax
    )
    params = {
        "log_h": jnp.log(k_0_jax),
        "log_c": jnp.log(jnp.maximum(c_guess, jnp.asarray(1e-12, dtype=dtype))),
    }
    solver = LevenbergMarquardt(human_capital_initial_residual, init_damping=1e-4)
    result = solver.solve(
        params,
        batch,
        max_steps=iterations,
        max_steps_is_success=False,
    )
    h_0 = jnp.exp(result.x["log_h"])
    c_0 = jnp.exp(result.x["log_c"])
    residual = human_capital_initial_residual(result.x, batch)
    return h_0, c_0, residual, result


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern12"
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 80.0
    train_points: int = 81
    train_points_list: list[float] | None = None
    test_T: float = 100.0
    test_points: int = 100
    validation_points: int = 300


# Acceptance gates, CLI-reachable so a float32 caller can pass its own bar
# (the float32 replication driver does).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 100
    # Conjunctive stopping: gtol bounds the ridge stationarity
    # ||J'r + ridge K alpha|| (calibrated per the
    # ~1e-3 * ridge * info.penalty_grad_norm recipe: ridge*||L'L theta|| is
    # ~1.6e-11 at convergence), residual_atol additionally demands the
    # scaled residual norm solved. The converged residual floors at
    # O(ridge * ||dual||), so residual_atol sits above it; the ridge keeps
    # the minimizer O(ridge) from the pure RKHS-seminorm interpolant with
    # the free initial values unpenalized (no epsilon shift anywhere).
    residual_atol: float = 1e-9
    ridge: float = 3e-11
    gtol: float = 1e-10


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
        train_data, kernel, sigma, ell, repeats=7
    )
    h_0_jax, c_0_init_jax, initial_residual, _ = human_capital_initial_conditions(
        a_k, a_h, delta_k, delta_h, k_0, dtype=dtype
    )
    h_0 = float(h_0_jax)
    c_0_init = float(c_0_init_jax)

    def path_values(theta, K_eval, K_tilde_eval):
        (
            alpha_k,
            alpha_h,
            alpha_i_k,
            alpha_i_h,
            alpha_c,
            alpha_mu_k,
            alpha_mu_h,
            initial,
        ) = theta
        i_k_0, i_h_0, c_0, mu_k_0, mu_h_0 = initial
        k = k_0 + K_tilde_eval @ alpha_k
        h = h_0 + K_tilde_eval @ alpha_h
        i_k = i_k_0 + K_tilde_eval @ alpha_i_k
        i_h = i_h_0 + K_tilde_eval @ alpha_i_h
        c = c_0 + K_tilde_eval @ alpha_c
        mu_k = mu_k_0 + K_tilde_eval @ alpha_mu_k
        mu_h = mu_h_0 + K_tilde_eval @ alpha_mu_h
        dk_dt = K_eval @ alpha_k
        dh_dt = K_eval @ alpha_h
        dmu_k_dt = K_eval @ alpha_mu_k
        dmu_h_dt = K_eval @ alpha_mu_h
        return k, h, i_k, i_h, c, mu_k, mu_h, dk_dt, dh_dt, dmu_k_dt, dmu_h_dt

    def production_terms(k, h):
        output = (k**a_k) * (h**a_h)
        f_k = a_k * k ** (a_k - 1.0) * h**a_h
        f_h = a_h * k**a_k * h ** (a_h - 1.0)
        return output, f_k, f_h

    def residuals(values):
        (
            k,
            h,
            i_k,
            i_h,
            c,
            mu_k,
            mu_h,
            dk_dt,
            dh_dt,
            dmu_k_dt,
            dmu_h_dt,
        ) = values
        output, f_k, f_h = production_terms(k, h)
        return (
            dk_dt - (i_k - delta_k * k),
            dh_dt - (i_h - delta_h * h),
            dmu_k_dt + mu_k * (f_k - delta_k - rho_hat),
            dmu_h_dt + mu_h * (f_h - delta_h - rho_hat),
            c + i_h + i_k - output,
            mu_k * c - 1.0,
            mu_k - mu_h,
        )

    residual_scale = 1.0 / jnp.sqrt(7.0 * n_train)

    def residual_fn(theta):
        return residual_scale * jnp.concatenate(
            residuals(path_values(theta, K, K_tilde))
        )

    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.asarray(
            [
                delta_k * k_0,
                delta_h * h_0,
                c_0_init,
                1.0 / c_0_init,
                1.0 / c_0_init,
            ],
            dtype=dtype,
        ),
    )
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=lm_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )

    solve_options = dict(
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )

    timer = Stopwatch()
    result = solver.solve(theta_0, **solve_options)
    elapsed = timer.elapsed(result)
    theta = result.x
    (
        alpha_k,
        alpha_h,
        alpha_i_k,
        alpha_i_h,
        alpha_c,
        alpha_mu_k,
        alpha_mu_h,
        initial,
    ) = theta
    i_k_0, i_h_0, c_0, mu_k_0, mu_h_0 = initial

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        values = path_values(theta, K_eval, K_tilde_eval)
        residual_values = residuals(values)
        k, h, i_k, i_h, c, mu_k, mu_h, *_ = values
        output, f_k, f_h = production_terms(k, h)
        names = (
            "physical_accumulation",
            "human_accumulation",
            "physical_euler",
            "human_euler",
            "feasibility",
            "shadow_price",
            "costate_gap",
        )
        return {
            "t": points,
            "k": k,
            "h": h,
            "i_k": i_k,
            "i_h": i_h,
            "c": c,
            "mu_k": mu_k,
            "mu_h": mu_h,
            **dict(zip(names, residual_values, strict=True)),
            "hidden_dae_residual": (f_k - delta_k) - (f_h - delta_h),
            "Y": output,
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    residual_names = (
        "physical_accumulation",
        "human_accumulation",
        "physical_euler",
        "human_euler",
        "feasibility",
        "shadow_price",
        "costate_gap",
    )
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )
    positive_names = ("k", "h", "i_k", "i_h", "c", "mu_k", "mu_h")
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name])) and jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in positive_names
        )
    )
    train_residual_tolerance = tolerance_set.train_residual
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > train_residual_tolerance:
        rejection_reasons.append("train_residual")

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        k, h, i_k, i_h, c, mu_k, mu_h, *_ = path_values(theta, K_eval, K_tilde_eval)
        return k, h, c, i_k, i_h, mu_k, mu_h

    alpha_by_name = {
        "k": alpha_k,
        "h": alpha_h,
        "i_k": alpha_i_k,
        "i_h": alpha_i_h,
        "c": alpha_c,
        "mu_k": alpha_mu_k,
        "mu_h": alpha_mu_h,
    }
    rkhs_norms = {
        name: float(alpha @ K @ alpha) for name, alpha in alpha_by_name.items()
    }
    results = {
        "a_k": a_k,
        "a_h": a_h,
        "delta_k": delta_k,
        "delta_h": delta_h,
        "rho_hat": rho_hat,
        "k_0": k_0,
        "h_0": h_0,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["feasibility"].dtype),
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "i_k_0": float(i_k_0),
        "i_h_0": float(i_h_0),
        "mu_k_0": float(mu_k_0),
        "mu_h_0": float(mu_h_0),
        "initial_condition_residual": float(
            jnp.linalg.norm(initial_residual, ord=jnp.inf)
        ),
        **{f"rkhs_norm_{name}": value for name, value in rkhs_norms.items()},
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": train_residual_tolerance,
        "max_validation_residual": max_validation_residual,
        "max_test_feasibility": float(jnp.max(jnp.abs(test_eval["feasibility"]))),
        "max_test_hidden_dae_residual": float(
            jnp.max(jnp.abs(test_eval["hidden_dae_residual"]))
        ),
        "solve_time": elapsed,
        "wall_time": elapsed,
        "solver": "nlls-gram",
        "linear_solver": "cholesky",
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
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
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "parameters": {
            "a_k": a_k,
            "a_h": a_h,
            "delta_k": delta_k,
            "delta_h": delta_h,
            "rho_hat": rho_hat,
        },
        "initial_conditions": {"k_0": k_0, "h_0": h_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
        "solve": SolveSpec(
            solver=solver,
            theta_0=theta_0,
            p=None,
            residual_args=None,
            multi_start=None,
            options=solve_options,
        ),
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_kernel)
