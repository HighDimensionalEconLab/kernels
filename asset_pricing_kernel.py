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
)

from asset_pricing_benchmark import mu_f_array
from kernels import (
    integrated_kernel_matrices,
    normalize_kernel,
    rkhs_norm_squared,
    shifted_kernel_coefficient_metric,
)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-8
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
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
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.5
    damping_increase: float = 4.0
    max_steps: int = 100
    residual_atol: float = 1e-11  # unscaled residual norm; bounds max|residual|
    p0_init: float = 0.0
    # The single shift of the unified metric M = blockdiag(K, 0) + eps I:
    # the kernel block is factored as K + eps I and the free p_0 carries
    # weight eps, so ||M^{-1}|| = 1/eps uniformly in n and the min-M-norm
    # interpolant is O(eps) from the pure RKHS-seminorm solution with p_0
    # free. A selection-side regularization only: residuals are unaffected,
    # and every solver/representation combination uses the same eps so their
    # selections agree. The default balances the ends of the measured working
    # range: larger shifts visibly bias the near-singular smooth Grams (rbf),
    # while below ~1e-8
    # the (c^2/eps)-weighted dual spike outruns the inner CG budgets. Tuning:
    # https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/#the-metric
    metric_eps: float = 1e-7


def asset_pricing_kernel(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 1.0,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(
            0, data_set.train_T, data_set.train_points, dtype=dtype
        )
    else:
        train_data = jnp.sort(jnp.array(data_set.train_points_list, dtype=dtype))
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    n_train = len(train_data)

    eps = lm_set.metric_eps
    if eps <= 0:
        raise ValueError("metric_eps must be positive (the p_0 block needs PD)")

    def dividends(t):
        return (x_0 + c / g) * jnp.exp(g * t) - c / g

    x_train = dividends(train_data)

    # theta is the tuple (alpha, p_0), which flattens to [alpha (n_train),
    # p_0 (1)] -- the block order of the metric below.
    def path_values(theta, K_eval, K_tilde_eval):
        alpha, p_0 = theta
        return p_0 + K_tilde_eval @ alpha, K_eval @ alpha

    metric_data = train_data.astype(jnp.float64)
    K_metric, K_tilde_metric = integrated_kernel_matrices(
        metric_data, metric_data, kernel, sigma, ell
    )
    K_metric = (K_metric + K_metric.T) / 2
    K_jax = K_metric.astype(dtype)
    K_tilde_jax = K_tilde_metric.astype(dtype)

    def train_path_values(theta):
        return path_values(theta, K_jax, K_tilde_jax)

    def residual_fn(theta):
        p, dp_dt = train_path_values(theta)
        return dp_dt - (r * p - x_train)

    # The residual is affine in theta, so LM from theta_0 = 0 converges to the
    # exact interpolant minimizing theta' M theta with the unified shifted
    # metric M = blockdiag(K, 0) + eps I: the min-RKHS-norm selection (the
    # original a'Ka objective with p_0 free) up to an O(eps) selection shift
    # in all parameters.
    metric = shifted_kernel_coefficient_metric(
        K_metric,
        repeats=1,
        free_size=1,
        epsilon=eps,
    )

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        linear_solve_dtype=jnp.float64,
        metric_solve_dtype=jnp.float64,
        metric=metric,
    )
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.asarray(lm_set.p0_init, dtype=dtype),
    )

    start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
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
    train_residual_tolerance = max(
        TRAIN_RESIDUAL_TOL,
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
    solution_accepted = not rejection_reasons

    results = {
        "r": r,
        "c": c,
        "g": g,
        "x_0": x_0,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": "float64",
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "p_0": float(p_0),
        "horizon_bias_p0": horizon_bias_p0,
        # Jitter-free alpha' K alpha (the metric norm includes the nugget).
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
        "linear_solver": "auto",
        "metric_eps": eps,
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "steps": steps,
        "stopping_reason": STATUS_TO_REASON.get(status, "unknown"),
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
    jsonargparse.CLI(asset_pricing_kernel)
