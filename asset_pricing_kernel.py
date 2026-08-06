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

from asset_pricing_benchmark import mu_f_array
from kernels import (
    integrated_kernel_matrices,
    integrated_metric_factorization,
    normalize_kernel,
    rkhs_norm_squared,
)
from utilities import (
    STATUS_TO_REASON,
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern12"  # matern12 | matern32 | matern52 | rbf
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100


# Acceptance gates, CLI-reachable so a float32 caller can pass its own bar
# (the float32 replication driver does).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-8


# The affine residual rewards near-Gauss-Newton steps, so a tiny init_damping
# converges in 2 LM steps (a factor ~10 faster than the IPOPT baseline); the
# damping decrease/increase factors never bind and stay at library defaults.
@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.5
    damping_increase: float = 4.0
    max_steps: int = 100
    residual_atol: float = 1e-9  # unscaled residual norm; bounds max|residual|
    p0_init: float = 0.0
    # The fixed ridge weight of the objective ||r||^2 + ridge * a'Ka with the
    # free p_0 unpenalized (no epsilon shift anywhere): the minimizer is
    # O(ridge) from the pure RKHS-seminorm interpolant with p_0 free. The
    # converged residual norm is O(ridge * ||dual||), so residual_atol must
    # sit above that floor -- the ridge objective deliberately backs off
    # exact interpolation, unlike the metric formulation whose affine solve
    # reached float64 roundoff. gtol bounds the ridge stationarity
    # ||J'r + ridge K a||; convergence requires gtol AND residual_atol
    # (conjunctive). The one-dimensional null space runs along the weakly
    # penalized p_0 mixture, and gtol follows the
    # ~1e-3 * ridge * info.penalty_grad_norm calibration recipe exactly:
    # ridge*||K a|| is ~1e-10 at convergence, so 1e-13 resolves p_0 to ~0.1%
    # (measured: gtol = 1e-9 leaves p_0 off by O(1); 1e-13 matches the
    # metric selection). Tuning:
    # https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/
    ridge: float = 1e-10
    gtol: float = 1e-13


def asset_pricing_kernel(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, dtype)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    n_train = len(train_data)

    def dividends(t):
        return (x_0 + c / g) * jnp.exp(g * t) - c / g

    x_train = dividends(train_data)

    # theta is the tuple (alpha, p_0), which flattens to [alpha (n_train),
    # p_0 (1)] -- the block order of the metric below.
    def path_values(theta, K_eval, K_tilde_eval):
        alpha, p_0 = theta
        return p_0 + K_tilde_eval @ alpha, K_eval @ alpha

    # The residual is affine in theta, so each ridge-LM step is an exact
    # kernel ridge solve converging to the minimizer of ||r||^2 + ridge * a'Ka
    # with p_0 in the unpenalized free block -- the min-RKHS-norm selection
    # (the original a'Ka objective) up to an O(ridge) bias.
    K_metric, K_tilde_jax, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=1
    )
    K_jax = K_metric

    def train_path_values(theta):
        return path_values(theta, K_jax, K_tilde_jax)

    def residual_fn(theta):
        p, dp_dt = train_path_values(theta)
        return dp_dt - (r * p - x_train)

    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=lm_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.asarray(lm_set.p0_init, dtype=dtype),
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
    alpha, p_0 = theta
    steps = int(result.steps)
    status = int(result.status)
    p_train, dp_train = train_path_values(theta)
    train_residual = dp_train - (r * p_train - x_train)
    max_train_residual = float(jnp.max(jnp.abs(train_residual)))

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, kernel, sigma, ell
        )
        return path_values(theta, K_test, K_tilde_test)

    p_test, dp_test = kernel_solution(test_data)
    test_residual = dp_test - (r * p_test - dividends(test_data))
    p_benchmark = mu_f_array(test_data, c, g, r, x_0)
    p_rel_error = jnp.abs(p_benchmark - p_test) / p_benchmark

    # Analytic finite-horizon bias of the dense-grid limit of the min-norm
    # selection (Matern-1/2 only): the continuum criterion ||p'||^2_H over
    # [0, T] selects p*(t) + zeta_T e^{rt} with
    # zeta_T = -<p*', r e^{r.}>_H / ||r e^{r.}||^2_H, elementary exponential
    # integrals under the H-norm (1/2s^2)[ell int f'^2 + (1/ell) int f^2
    # + f(0)^2 + f(T)^2]. The p_0 bias is zeta_T; the bias at time t is
    # zeta_T e^{rt}. At ell = 1/r it collapses to -(A/r) e^{(g-r)T}. The
    # remaining p_0 error is the O(dt^2) grid-spacing selection bias.
    if kernel == "matern12":
        T = float(train_data[-1])
        A = g * (x_0 + c / g) / (r - g)  # p*'(t) = A e^{g t}
        E, F = np.exp((g + r) * T), np.exp(2 * r * T)
        I_1, I_2 = (E - 1) / (g + r), (F - 1) / (2 * r)
        horizon_bias_p0 = float(
            -(A / r)
            * ((ell * g * r + 1 / ell) * I_1 + 1 + E)
            / ((ell * r * r + 1 / ell) * I_2 + 1 + F)
        )
    else:
        horizon_bias_p0 = float("nan")

    # Acceptance certifies the solution, not the stopping code: a solve that
    # stalls at its residual floor below the train gate is feasible (the
    # IPOPT baseline's FEASIBLE_SMALL_STEP semantics) -- the smooth kernels
    # routinely end there because atol is unreachable for their
    # ill-conditioned Grams. stopping_reason still reports the code.
    train_residual_tolerance = max(
        tolerance_set.train_residual,
        8.0 * float(jnp.finfo(dtype).eps),
    )
    finite = bool(jnp.all(jnp.isfinite(p_test)) and jnp.all(jnp.isfinite(p_benchmark)))
    rejection_reasons = []
    if not finite:
        rejection_reasons.append("nonfinite")
    if max_train_residual > train_residual_tolerance:
        rejection_reasons.append("train_residual")
    if float(p_0) < 0.0:
        rejection_reasons.append("negative_p_0")

    results = {
        "r": r,
        "c": c,
        "g": g,
        "x_0": x_0,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_residual.dtype),
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(p_0),
        "horizon_bias_p0": horizon_bias_p0,
        # Unshifted alpha' K alpha (the metric norm includes the epsilon shift).
        "rkhs_norm_p": float(rkhs_norm_squared(alpha, K_metric)),
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": train_residual_tolerance,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
        "solve_time": elapsed,
        "wall_time": elapsed,
        "solver": "nlls-gram",
        "linear_solver": "cholesky",
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": steps,
        "stopping_reason": STATUS_TO_REASON.get(status, "unknown"),
        **acceptance_fields(rejection_reasons),
    }
    print(
        f"solve_time(s) = {elapsed}, "
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
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "parameters": {"r": r, "c": c, "g": g},
        "initial_conditions": {"x_0": x_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_kernel)
