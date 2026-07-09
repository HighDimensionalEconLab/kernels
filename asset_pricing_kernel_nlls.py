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
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_tridiagonal_precision,
    sherman_morrison_preconditioner,
)

from asset_pricing_benchmark import mu_f_array
from kernels import (
    integrated_kernel_matrices,
    matern_0p5_tridiagonal_precision,
    matern_kernel_matvec_pair,
    matern_qsm_solve,
    normalize_kernel,
)

config.update("jax_enable_x64", True)

TRAIN_RESIDUAL_TOL = 1e-8
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}
# method="auto" thresholds: dense wins below MATRIX_FREE_MIN_POINTS; the
# structured methods win beyond it (measured on CPU; the gap widens with n).
# The smoother the kernel, the faster its nugget-free Gram becomes numerically
# singular under grid refinement, and past that wall the iterative dual solves
# stall where the direct dense factorization still works: matern32's iterative
# window closes around n ~ 5000 (its accuracy already saturates by n ~ 2000),
# and matern52/rbf are dense-only outright.
MATRIX_FREE_MIN_POINTS = 256
MATERN32_MATRIX_FREE_MAX_POINTS = 4000
PRECONDITIONED_CG_MIN_POINTS = 1500
# Per-kernel metric nugget (relative to sigma^2), applied wherever K is
# factored; matern12 stays exact (closed-form, well-conditioned tridiagonal
# precision).
DEFAULT_METRIC_JITTER = {
    "matern12": 0.0,
    "matern32": 1e-10,
    "matern52": 1e-9,
    "rbf": 1e-9,
}


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
    # "auto" picks by kernel, grid, and size. "dense" factors the dual system
    # directly -- the universal fallback (any kernel or grid; the only method
    # for matern52 and rbf, whose smooth Grams put the iterative dual solves
    # past the float64 conditioning wall). "matrix_free" is O(n) time and
    # memory end to end -- structured matvec residual plus an O(n) kernel
    # metric and Sherman-Morrison preconditioner -- for matern12 (any n) and
    # matern32 (up to ~5000 points) on strictly increasing grids.
    # "preconditioned_cg" keeps the dense metric factor but solves the dual
    # system by preconditioned CG with a budget flat in n; its niche is
    # matern12 on unsorted grids at scale.
    method: str = "auto"
    iterative_maxiter: int = 20  # CG inner budget per LM step (cg methods)
    # Absolute nugget added to K wherever the metric factors it; None picks
    # the per-kernel default (DEFAULT_METRIC_JITTER * sigma^2). A selection-
    # side regularization only: residuals are unaffected, and every method
    # uses the same value so their min-norm solutions agree.
    metric_jitter: float | None = None


def asset_pricing_kernel_nlls(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 1.0,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
):
    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(0, data_set.train_T, data_set.train_points)
    else:
        train_data = jnp.array(data_set.train_points_list)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points)
    n_train = len(train_data)

    increasing_grid = bool(jnp.all(jnp.diff(train_data) > 0))
    matrix_free_ok = increasing_grid and (
        kernel == "matern12"
        or (kernel == "matern32" and n_train <= MATERN32_MATRIX_FREE_MAX_POINTS)
    )
    method = lm_set.method
    if method == "auto":
        if matrix_free_ok and n_train >= MATRIX_FREE_MIN_POINTS:
            method = "matrix_free"
        elif kernel == "matern12" and n_train >= PRECONDITIONED_CG_MIN_POINTS:
            method = "preconditioned_cg"
        else:
            method = "dense"
    elif method not in ("dense", "preconditioned_cg", "matrix_free"):
        raise ValueError(f"unknown method: {lm_set.method!r}")
    if method == "matrix_free" and not (
        kernel in ("matern12", "matern32") and increasing_grid
    ):
        raise ValueError(
            'method="matrix_free" requires matern12 or matern32 on a strictly '
            'increasing train grid; use "dense" otherwise'
        )
    if method == "preconditioned_cg" and kernel != "matern12":
        raise ValueError(
            'method="preconditioned_cg" requires matern12; use "dense" otherwise'
        )
    jitter = lm_set.metric_jitter
    if jitter is None:
        jitter = DEFAULT_METRIC_JITTER[kernel] * sigma**2

    def dividends(t):
        return (x_0 + c / g) * jnp.exp(g * t) - c / g

    x_train = dividends(train_data)

    # theta is the tuple (alpha, p_0), which flattens to [alpha (n_train),
    # p_0 (1)] -- the block order of the metric below.
    def path_values(theta, K_eval, K_tilde_eval):
        alpha, p_0 = theta
        return p_0 + K_tilde_eval @ alpha, K_eval @ alpha

    if method == "matrix_free":

        def train_path_values(theta):
            alpha, p_0 = theta
            dp_dt, k_tilde_alpha = matern_kernel_matvec_pair(
                kernel, train_data, alpha, sigma, ell
            )
            return p_0 + k_tilde_alpha, dp_dt

        if kernel == "matern12":
            # The Matern-1/2 precision is exactly tridiagonal; no nugget.
            alpha_metric = metric_from_tridiagonal_precision(
                *matern_0p5_tridiagonal_precision(train_data, sigma, ell)
            )
        else:

            def alpha_norm(x):
                k_x, _ = matern_kernel_matvec_pair(kernel, train_data, x, sigma, ell)
                return jnp.sqrt(x @ k_x + jitter * (x @ x))

            alpha_metric = Metric(
                solve=matern_qsm_solve(kernel, train_data, sigma, ell, jitter),
                norm=alpha_norm,
            )
    else:
        K, K_tilde = integrated_kernel_matrices(
            train_data, train_data, kernel, sigma, ell
        )
        K_jax = (K + K.T) / 2
        K_tilde_jax = K_tilde

        def train_path_values(theta):
            return path_values(theta, K_jax, K_tilde_jax)

        alpha_metric = metric_from_cholesky(
            jnp.linalg.cholesky(K_jax + jitter * jnp.eye(n_train))
        )

    def residual_fn(theta):
        p, dp_dt = train_path_values(theta)
        return dp_dt - (r * p - x_train)

    # The residual is affine in theta, so LM from theta_0 = 0 converges to the
    # exact interpolant minimizing theta' M theta with M = blockdiag(K, m_0):
    # the min-RKHS-norm selection (the original a'Ka objective) up to a
    # selection shift in p_0 that scales linearly with m_0 (~1e-7 at the
    # default m_0 = 1e-6).
    m_0 = lm_set.p0_metric_weight
    metric = blockdiag_metric(
        [(alpha_metric, n_train), (metric_from_diagonal(jnp.full(1, m_0)), 1)]
    )

    # The p_0 residual column -r*1 injects the exactly known rank-1 spike
    # (r^2/m_0) 11' into the dual operator J M^{-1} J'; preconditioning with
    # K + (r^2/m_0) 11' is spectrally equivalent uniformly in n, pinning the
    # inner CG budget where the unpreconditioned budget grows with refinement.
    dual_preconditioner = None
    if method != "dense":
        dual_preconditioner = sherman_morrison_preconditioner(
            alpha_metric.solve, jnp.ones(n_train), r**2 / m_0
        )

    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver="cholesky" if method == "dense" else "cg",
        iterative_maxiter=lm_set.iterative_maxiter,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        dual_preconditioner=dual_preconditioner,
        metric=metric,
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
    # stalls at its float64 residual floor below TRAIN_RESIDUAL_TOL is
    # feasible (the IPOPT baseline's FEASIBLE_SMALL_STEP semantics) -- the
    # smooth kernels routinely end there because atol is unreachable for
    # their ill-conditioned Grams. stopping_reason still reports the code.
    finite = bool(jnp.all(jnp.isfinite(p_test)) and jnp.all(jnp.isfinite(p_benchmark)))
    rejection_reasons = []
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
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(p_0),
        "horizon_bias_p0": horizon_bias_p0,
        "rkhs_norm_p": float(alpha_metric.norm(alpha) ** 2),
        "max_train_residual": max_train_residual,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
        "solve_time": elapsed,
        "wall_time": elapsed,
        "solver": "nlls-gram",
        "method_resolved": method,
        "metric_jitter": jitter,
        "train_loss": float(result.info.loss),
        "steps": steps,
        "stopping_reason": STATUS_TO_REASON.get(status, "unknown"),
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
