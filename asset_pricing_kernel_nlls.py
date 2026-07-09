from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from jax import config
from nlls_gram import (
    LMStatus,
    Metric,
    UnderdeterminedLevenbergMarquardt,
    metric_from_cholesky,
    metric_from_tridiagonal_precision,
)

from asset_pricing_benchmark import mu_f_array
from kernels import (
    integrated_kernel_matrices,
    matern_0p5_tridiagonal_precision,
    matern_kernel_0p5_matvec_pair,
)

config.update("jax_enable_x64", True)

TRAIN_RESIDUAL_TOL = 1e-8
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class KernelSettings:
    nu: float = 0.5  # math.inf selects the Gaussian/RBF kernel
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100


# The affine residual rewards near-Gauss-Newton steps, so a tiny init_damping
# converges in 2 LM steps (a factor ~10 faster than the IPOPT baseline); the
# damping decrease/increase factors never bind and stay at library defaults.
@dataclass(frozen=True)
class NLLSGramSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.5
    damping_increase: float = 4.0
    max_steps: int = 100
    residual_atol: float = 1e-11  # unscaled residual norm; bounds max|residual|
    p0_metric_weight: float = 1e-6  # m_0: PD-completion of the semi-norm a'Ka
    p0_init: float = 0.0
    # "cholesky" or "cg"; the metric provides the solve/norm callbacks both
    # need. Unpreconditioned CG stalls at scale: metric.solve puts K^{-1}
    # inside the dual operator (cond ~ n^2), so the inner budget must grow
    # with grid refinement. With an "_sm" preconditioner the budget is flat in
    # n, and cg overtakes the dense path beyond n ~ 1000 with tridiagonal_sm
    # (matrix_free extends it to O(n)); cholesky_sm gains little because its
    # per-iteration cho_solve is O(n^2). Dense cholesky remains the right
    # default at small n.
    linear_solver: str = "cholesky"
    iterative_maxiter: int = 8  # CG inner iterations per LM step; ~20 with a
    # preconditioner reaches the exact Gauss-Newton step in 2 LM steps
    # "none" | "cholesky_sm" | "tridiagonal_sm": dual preconditioner
    # P = K + (r^2/m_0) 11' for linear_solver="cg", applied as a K-solve plus
    # a Sherman-Morrison rank-1 correction. "cholesky_sm" reuses the dense
    # metric factor; "tridiagonal_sm" also switches the metric itself to the
    # exact O(n) Matern-1/2 tridiagonal precision (no dense factorization) and
    # requires a strictly increasing grid. With linear_solver="cholesky" the
    # preconditioner part is inert and only the metric choice applies.
    preconditioner: str = "none"
    # Replace the dense K/K_tilde train matrices with O(n) Rybicki-Press
    # matvecs, making the whole solve O(n) time and memory (n = 100k trains in
    # ~75ms). Requires nu = 0.5, linear_solver="cg", and
    # preconditioner="tridiagonal_sm".
    matrix_free: bool = False


def asset_pricing_kernel_nlls(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 1.0,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
):
    nu, ell, sigma = kernel_set.nu, kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(0, data_set.train_T, data_set.train_points)
    else:
        train_data = jnp.array(data_set.train_points_list)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points)

    n_train = len(train_data)
    if lm_set.preconditioner not in ("none", "cholesky_sm", "tridiagonal_sm"):
        raise ValueError(f"unknown preconditioner: {lm_set.preconditioner!r}")
    if lm_set.preconditioner == "tridiagonal_sm":
        if nu != 0.5:
            raise ValueError('preconditioner="tridiagonal_sm" requires nu = 0.5')
        # The AR(1) precision and the Rybicki-Press matvecs are only the
        # Matern-1/2 kernel on a sorted grid; the dense path has no such
        # restriction, so guard here rather than silently solving the wrong
        # operator.
        if not bool(jnp.all(jnp.diff(train_data) > 0)):
            raise ValueError(
                'preconditioner="tridiagonal_sm" requires a strictly '
                "increasing train grid"
            )
    if lm_set.matrix_free and (
        lm_set.linear_solver != "cg" or lm_set.preconditioner != "tridiagonal_sm"
    ):
        raise ValueError(
            'matrix_free requires linear_solver="cg" and '
            'preconditioner="tridiagonal_sm"'
        )

    def dividends(t):
        return (x_0 + c / g) * jnp.exp(g * t) - c / g

    x_train = dividends(train_data)

    # theta is the tuple (alpha, p_0); ravel_pytree flattens it in tuple order
    # to [alpha (n_train), p_0 (1)], matching the n_train slicing in the metric
    # callbacks below.
    def path_values(theta, K_eval, K_tilde_eval):
        alpha, p_0 = theta
        p = p_0 + K_tilde_eval @ alpha
        dp_dt = K_eval @ alpha
        return p, dp_dt

    if lm_set.matrix_free:

        def train_path_values(theta):
            alpha, p_0 = theta
            dp_dt, k_tilde_alpha = matern_kernel_0p5_matvec_pair(
                train_data, alpha, sigma, ell
            )
            return p_0 + k_tilde_alpha, dp_dt
    else:
        K, K_tilde = integrated_kernel_matrices(train_data, train_data, nu, sigma, ell)
        K_jax = (K + K.T) / 2
        K_tilde_jax = K_tilde

        def train_path_values(theta):
            return path_values(theta, K_jax, K_tilde_jax)

    def residual_fn(theta):
        p, dp_dt = train_path_values(theta)
        return dp_dt - (r * p - x_train)

    # The residual is affine in theta, so LM from theta_0 = 0 converges to the
    # exact interpolant minimizing theta' M theta. With M = blockdiag(K, m_0)
    # this recovers the min-RKHS-norm selection (the original a'Ka objective)
    # up to a selection shift in p_0 that scales linearly with m_0 (~1e-7 at
    # the default m_0 = 1e-6).
    m_0 = lm_set.p0_metric_weight
    if lm_set.preconditioner == "tridiagonal_sm":
        # Matern-1/2 is Markov, so K^{-1} is exactly tridiagonal with
        # closed-form OU/AR(1) precision entries (nonuniform-grid-ready):
        # nothing dense is ever factored on this path.
        alpha_metric = metric_from_tridiagonal_precision(
            *matern_0p5_tridiagonal_precision(train_data, sigma, ell)
        )
    else:
        alpha_metric = metric_from_cholesky(jnp.linalg.cholesky(K_jax))
    K_solve = alpha_metric.solve

    def alpha_norm_sq(x):
        return alpha_metric.norm(x) ** 2

    def metric_solve(x):
        # x may be a (n+1,) vector or (n+1, m) matrix of columns to solve.
        return jnp.concatenate([K_solve(x[:n_train]), x[n_train:] / m_0])

    def metric_norm(x):
        return jnp.sqrt(alpha_norm_sq(x[:n_train]) + m_0 * jnp.sum(x[n_train:] ** 2))

    # Dual preconditioner P = K + (r^2/m_0) 11' for the CG path: the leading
    # term of J M^{-1} J' = B K^{-1} B' + (r^2/m_0) 11' plus the exact rank-1
    # spike, applied as a K-solve with a Sherman-Morrison correction. It is
    # spectrally equivalent to the dual operator uniformly in n, pinning the
    # inner iteration count; only used when linear_solver="cg".
    if lm_set.preconditioner == "none" or lm_set.linear_solver != "cg":
        dual_preconditioner = None
    else:
        ones = jnp.ones(n_train)
        K_inv_ones = K_solve(ones)
        sm_denominator = m_0 / r**2 + ones @ K_inv_ones

        def dual_preconditioner(v, damping):
            y = K_solve(v)
            return y - K_inv_ones * ((ones @ y) / sm_denominator)

    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        iterative_maxiter=lm_set.iterative_maxiter,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        dual_preconditioner=dual_preconditioner,
        metric=Metric(solve=metric_solve, norm=metric_norm),
    )

    theta_0 = (jnp.zeros(n_train), jnp.asarray(lm_set.p0_init))

    # The timed solve below reuses this solver instance, so its jit cache is
    # warm; max_steps/atol are dynamic and do not recompile.
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
    alpha, p_0 = theta
    steps = int(result.steps)
    status = int(result.status)
    stopping_reason = STATUS_TO_REASON.get(status, "unknown")
    train_loss = float(result.info.loss)
    p_train, dp_train = train_path_values(theta)
    train_residual = dp_train - (r * p_train - x_train)
    max_train_residual = float(jnp.max(jnp.abs(train_residual)))

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, nu, sigma, ell
        )
        p_test, dp_test = path_values(theta, K_test, K_tilde_test)
        return p_test, dp_test

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
    if nu == 0.5:
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

    finite = bool(jnp.all(jnp.isfinite(p_test)) and jnp.all(jnp.isfinite(p_benchmark)))
    rejection_reasons = []
    if status != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite:
        rejection_reasons.append("nonfinite")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if float(p_0) < 0.0:
        rejection_reasons.append("negative_p_0")
    solution_accepted = not rejection_reasons

    results = {
        "r": r,
        "c": c,
        "g": g,
        "x_0": x_0,
        "nu": nu,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(p_0),
        "horizon_bias_p0": horizon_bias_p0,
        "rkhs_norm_p": float(alpha_norm_sq(alpha)),
        "max_train_residual": max_train_residual,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
        "solve_time": elapsed,
        "wall_time": elapsed,
        "solver": "nlls-gram",
        "train_loss": train_loss,
        "steps": steps,
        "stopping_reason": stopping_reason,
        "compile_time": compile_time,
        "solution_accepted": solution_accepted,
        "rejection_reason": "accepted"
        if solution_accepted
        else ",".join(rejection_reasons),
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
    jsonargparse.CLI(asset_pricing_kernel_nlls)
