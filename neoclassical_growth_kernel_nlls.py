# NLLS port of the smooth-production neoclassical growth collocation; the
# kinked-production variant (A, b_1, b_2 and its mpk-bound diagnostics) stays
# with the IPOPT version in neoclassical_growth_kernel.py.
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
    UnderdeterminedLevenbergMarquardt,
    blockdiag_metric,
    matern_state_space,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_from_state_space,
    metric_from_tridiagonal_precision,
)

from kernels import (
    integrated_kernel_matrices,
    matern_0p5_tridiagonal_precision,
    matern_kernel_matvec_pair,
    normalize_kernel,
)
from neoclassical_growth_benchmark import neoclassical_growth_benchmark

config.update("jax_enable_x64", True)

TRAIN_RESIDUAL_TOL = 1e-5
DOMAIN_EPS = 1e-6
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}
# Auto boundary as in the asset-pricing model: dense wins below the
# matrix_free threshold. Unlike asset pricing, auto only picks matrix_free
# for matern12 here: the growth dual operator's shadow-price block resists
# the blockwise K preconditioner, and the matern32 iterative window that
# worked there fails the growth gates at n >= 501 (measured). Explicit
# matern32 matrix_free remains allowed and is policed by the gates.
MATRIX_FREE_MIN_POINTS = 256
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
    benchmark_T: float = 80.0
    benchmark_points: int = 300


# The residual is nonlinear (production, mu*c products) but mildly so: with
# near-Gauss-Newton damping (tiny init, fast decrease) and geodesic
# acceleration (library default) the solve lands in 4-25 steps, where a
# cautious damping schedule needed 100+.
@dataclass(frozen=True)
class NLLSGramSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    # Nonlinear solves land in 4-25 steps; past ~20 the large-n dual sits at
    # its float64 conditioning floor and further steps buy nothing (the
    # acceptance gates certify the stall).
    max_steps: int = 50
    residual_atol: float = 1e-9  # on the 1/sqrt(3n)-scaled residual norm
    ic_metric_weight: float = 1e-6  # m_0 on (c_0, mu_0): PD-completion
    # "auto" | "dense" | "matrix_free". auto picks matrix_free only for
    # matern12 on strictly increasing grids at scale (see the module comment
    # on the growth dual operator); dense is the universal fallback.
    method: str = "auto"
    iterative_maxiter: int = 100  # CG inner budget per LM step (matrix_free)
    metric_jitter: float | None = None


def neoclassical_growth_kernel_nlls(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
    c_0_guess: float | None = None,  # overrides the flat steady-flow warm start
):
    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(0, data_set.train_T, data_set.train_points)
    else:
        train_data = jnp.array(data_set.train_points_list)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points)
    benchmark_grid = jnp.linspace(0, data_set.benchmark_T, data_set.benchmark_points)
    n_train = len(train_data)

    increasing_grid = bool(jnp.all(jnp.diff(train_data) > 0))
    matrix_free_ok = increasing_grid and kernel == "matern12"
    method = lm_set.method
    if method == "auto":
        if matrix_free_ok and n_train >= MATRIX_FREE_MIN_POINTS:
            method = "matrix_free"
        else:
            method = "dense"
    elif method not in ("dense", "matrix_free"):
        raise ValueError(f"unknown method: {lm_set.method!r}")
    if method == "matrix_free" and not (
        kernel in ("matern12", "matern32") and increasing_grid
    ):
        raise ValueError(
            'method="matrix_free" requires matern12 or matern32 on a strictly '
            'increasing train grid; use "dense" otherwise'
        )
    jitter = lm_set.metric_jitter
    if jitter is None:
        jitter = DEFAULT_METRIC_JITTER[kernel] * sigma**2
    if method == "matrix_free" and kernel == "matern12" and jitter != 0.0:
        raise ValueError(
            'matern12 with method="matrix_free" uses the exact tridiagonal '
            "precision, which has no nugget; metric_jitter must be 0"
        )

    def production(k):
        return jnp.maximum(k, DOMAIN_EPS) ** a

    # Deliberately the unclamped derivative evaluated at max(k, eps): below
    # the clamp it stays large and positive (~a eps^{a-1}), blowing up the
    # euler residual and repelling the unconstrained LM iterates from the
    # k <= 0 region that IPOPT excluded with variable bounds. Autodiffing the
    # clamped production would instead give mpk = 0 there and no repulsion.
    def marginal_product(k):
        return a * jnp.maximum(k, DOMAIN_EPS) ** (a - 1.0)

    # theta is the tuple (alpha_mu, alpha_c, alpha_k, (c_0, mu_0)), which
    # flattens to [alpha_mu (n), alpha_c (n), alpha_k (n), c_0, mu_0] -- the
    # block order of the metric below.
    def path_values(theta, K_eval, K_tilde_eval):
        alpha_mu, alpha_c, alpha_k, initial = theta
        c_0, mu_0 = initial
        mu = mu_0 + K_tilde_eval @ alpha_mu
        c = c_0 + K_tilde_eval @ alpha_c
        k = k_0 + K_tilde_eval @ alpha_k
        dmu_dt = K_eval @ alpha_mu
        dk_dt = K_eval @ alpha_k
        return k, c, mu, dk_dt, dmu_dt

    if method == "matrix_free":

        def train_path_values(theta):
            alpha_mu, alpha_c, alpha_k, initial = theta
            c_0, mu_0 = initial
            dmu_dt, k_tilde_mu = matern_kernel_matvec_pair(
                kernel, train_data, alpha_mu, sigma, ell
            )
            _, k_tilde_c = matern_kernel_matvec_pair(
                kernel, train_data, alpha_c, sigma, ell
            )
            dk_dt, k_tilde_k = matern_kernel_matvec_pair(
                kernel, train_data, alpha_k, sigma, ell
            )
            return (
                k_0 + k_tilde_k,
                c_0 + k_tilde_c,
                mu_0 + k_tilde_mu,
                dk_dt,
                dmu_dt,
            )

        if kernel == "matern12":
            alpha_metric = metric_from_tridiagonal_precision(
                *matern_0p5_tridiagonal_precision(train_data, sigma, ell)
            )
        else:
            # Exact O(n) state-space metric M = K + jitter I with all four
            # callbacks (the nugget is folded into the metric itself).
            alpha_metric = metric_from_state_space(
                train_data,
                *matern_state_space(
                    sigma, ell, {"matern32": 1.5, "matern52": 2.5}[kernel]
                ),
                nugget=jitter,
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

    # The residual is scaled by 1/sqrt(3n) (the NN files' convention) so that
    # residual_atol is a scale-free per-point bound rather than tightening
    # with grid refinement.
    residual_scale = 1.0 / jnp.sqrt(3.0 * n_train)

    def residual_fn(theta):
        k, c, mu, dk_dt, dmu_dt = train_path_values(theta)
        resource = dk_dt - (production(k) - delta * k - c)
        euler = dmu_dt + mu * (marginal_product(k) - delta - rho_hat)
        shadow_price = mu * c - 1.0
        return residual_scale * jnp.concatenate([resource, euler, shadow_price])

    # 3n nonlinear residuals in 3n + 2 unknowns: LM with the block metric
    # M = blockdiag(K, K, K, m_0 I_2) selects the interpolant minimizing the
    # summed RKHS norms (the original objective) plus the PD-completion on the
    # free initial values (c_0, mu_0).
    m_0 = lm_set.ic_metric_weight
    metric = blockdiag_metric(
        [
            (alpha_metric, n_train),
            (alpha_metric, n_train),
            (alpha_metric, n_train),
            (metric_from_diagonal(jnp.full(2, m_0)), 2),
        ]
    )

    # Blockwise K-solve preconditioner for the CG dual system: the resource
    # and euler diagonal blocks of J M^{-1} J' are K plus bounded terms, and
    # CG absorbs the few rank-1 spikes from the (c_0, mu_0) columns on its
    # own. The shadow-price block is smoother (K_tilde K^{-1} K_tilde'-like),
    # so the match is partial -- the budget below reflects that.
    dual_preconditioner = None
    if method != "dense":
        K_solve = alpha_metric.solve

        def dual_preconditioner(v, damping):
            return jnp.concatenate(
                [
                    K_solve(v[:n_train]),
                    K_solve(v[n_train : 2 * n_train]),
                    K_solve(v[2 * n_train :]),
                ]
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

    c_flow_guess = max(production(k_0) - delta * k_0, DOMAIN_EPS)
    if c_0_guess is not None:
        c_flow_guess = max(float(c_0_guess), DOMAIN_EPS)
    theta_0 = (
        jnp.zeros(n_train),
        jnp.zeros(n_train),
        jnp.zeros(n_train),
        jnp.array([c_flow_guess, 1.0 / c_flow_guess]),
    )

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
    alpha_mu, alpha_c, alpha_k, initial = theta
    c_0, mu_0 = initial
    steps = int(result.steps)
    status = int(result.status)

    def evaluate_grid(points_data):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points_data, train_data, kernel, sigma, ell
        )
        k, c, mu, dk_dt, dmu_dt = path_values(theta, K_eval, K_tilde_eval)
        return {
            "t": points_data,
            "k": k,
            "c": c,
            "mu": mu,
            "Y": production(k),
            "resource": dk_dt - (production(k) - delta * k - c),
            "euler": dmu_dt + mu * (marginal_product(k) - delta - rho_hat),
            "shadow_price": mu * c - 1.0,
        }

    train_eval = evaluate_grid(train_data)
    test_eval = evaluate_grid(test_data)

    residual_names = ("resource", "euler", "shadow_price")
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_kernel_matrices(
            test_points_data, train_data, kernel, sigma, ell
        )
        k, c, _, _, _ = path_values(theta, K_test, K_tilde_test)
        return k, c

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name]))
            for grid in (train_eval, test_eval)
            for name in ("k", "c", "mu")
        )
        and all(
            jnp.min(grid[name]) > 0.0
            for grid in (train_eval, test_eval)
            for name in ("k", "c", "mu")
        )
    )

    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    solution_accepted = not rejection_reasons

    benchmark_solution = neoclassical_growth_benchmark(
        a, delta, rho_hat, 1.0, k_0, benchmark_grid
    )
    k_benchmark, c_benchmark = benchmark_solution(test_data)
    k_rel_error = jnp.abs(k_benchmark - test_eval["k"]) / k_benchmark
    c_rel_error = jnp.abs(c_benchmark - test_eval["c"]) / c_benchmark

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "k_0": k_0,
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        # Jitter-free alpha' K alpha, matching the IPOPT report (the metric
        # norm includes the nugget).
        "rkhs_norm_k": float(
            alpha_metric.norm(alpha_k) ** 2 - jitter * (alpha_k @ alpha_k)
        ),
        "rkhs_norm_c": float(
            alpha_metric.norm(alpha_c) ** 2 - jitter * (alpha_c @ alpha_c)
        ),
        "rkhs_norm_mu": float(
            alpha_metric.norm(alpha_mu) ** 2 - jitter * (alpha_mu @ alpha_mu)
        ),
        "max_train_residual": max_train_residual,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_eval["euler"]))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]): its
        # near-zero error is an artifact of the initial condition, not accuracy.
        "k_rel_error_mean": float(k_rel_error.mean()),
        "c_rel_error_mean": float(c_rel_error.mean()),
        "k_rel_error_max": float(k_rel_error[1:].max()),
        "c_rel_error_max": float(c_rel_error[1:].max()),
        "k_rel_error_min": float(k_rel_error[1:].min()),
        "c_rel_error_min": float(c_rel_error[1:].min()),
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
        f"E(|rel_error(k)|) = {results['k_rel_error_mean']:.3e}, "
        f"E(|rel_error(c)|) = {results['c_rel_error_mean']:.3e}"
    )

    train_df = pd.DataFrame(
        {
            name: np.asarray(train_eval[name])
            for name in ("t", "k", "c", "mu", "resource", "euler", "shadow_price")
        }
    )
    test_df = pd.DataFrame(
        {
            name: np.asarray(test_eval[name])
            for name in ("t", "k", "c", "mu", "Y", "resource", "euler", "shadow_price")
        }
    )
    test_df["k_benchmark"] = np.asarray(k_benchmark)
    test_df["c_benchmark"] = np.asarray(c_benchmark)
    test_df["k_rel_error"] = np.asarray(k_rel_error)
    test_df["c_rel_error"] = np.asarray(c_rel_error)

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "parameters": {"a": a, "delta": delta, "rho_hat": rho_hat},
        "initial_conditions": {"k_0": k_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_nlls)
