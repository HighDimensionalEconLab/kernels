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
from neoclassical_growth_benchmark import neoclassical_growth_benchmark

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
MPK_BOUND_TOL = 1e-3
DOMAIN_EPS = 1e-6
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
    benchmark_T: float = 80.0
    benchmark_points: int = 300


# The residual is nonlinear (production, mu*c products) but mildly so: with
# near-Gauss-Newton damping (tiny init, fast decrease) and geodesic
# acceleration (library default) the solve lands in 4-25 steps, where a
# cautious damping schedule needed 100+.
@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 50
    residual_atol: float = 2e-8  # on the 1/sqrt(3n)-scaled residual norm
    # The single shift of the unified metric M = blockdiag(K, K, K, 0) +
    # eps I: every kernel block is factored as K + eps I and the free
    # (c_0, mu_0) carry weight eps, so ||M^{-1}|| = 1/eps uniformly in n and
    # the min-M-norm interpolant is O(eps) from the pure RKHS-seminorm
    # solution with free initial values. Selection-side only; every
    # solver/representation combination uses the same eps so their selections
    # agree. The default balances the ends of the measured working range:
    # larger shifts visibly bias the near-singular smooth Grams (rbf), while
    # below ~1e-8 the
    # (1/eps)-weighted dual spike outruns the inner CG budget. Tuning:
    # https://highdimensionaleconlab.github.io/nlls_gram/tuning_guide/#the-metric
    metric_eps: float = 1e-7


def neoclassical_growth_kernel(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    c_0_guess: float | None = None,  # overrides the flat steady-flow warm start
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kinked_parameter_count = sum(parameter is not None for parameter in (A, b_1, b_2))
    if kinked_parameter_count not in {0, 3}:
        raise ValueError("Set all of A, b_1, and b_2, or set none of them.")
    use_kinked_production = kinked_parameter_count == 3
    A_value = 1.0 if A is None else float(A)
    b_1_value = 1.0 if b_1 is None else float(b_1)
    b_2_value = 0.0 if b_2 is None else float(b_2)

    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    if data_set.train_points_list is None:
        train_data = jnp.linspace(
            0, data_set.train_T, data_set.train_points, dtype=dtype
        )
    else:
        train_data = jnp.sort(jnp.array(data_set.train_points_list, dtype=dtype))
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_points = max(data_set.test_points, data_set.benchmark_points)
    validation_data = jnp.linspace(0, data_set.train_T, validation_points, dtype=dtype)
    benchmark_grid = jnp.linspace(
        0, data_set.benchmark_T, data_set.benchmark_points, dtype=dtype
    )
    n_train = len(train_data)

    eps = lm_set.metric_eps
    if eps <= 0:
        raise ValueError("metric_eps must be positive (the (c_0, mu_0) block needs PD)")

    def production(k):
        base = k**a
        return A_value * jnp.maximum(base, b_1_value * base - b_2_value)

    def marginal_product(k):
        base = k**a
        branch = jnp.where(b_1_value * base - b_2_value > base, b_1_value, 1.0)
        return A_value * a * branch * k ** (a - 1.0)

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

    metric_data = train_data.astype(jnp.float64)
    K_metric, K_tilde_metric = integrated_kernel_matrices(
        metric_data, metric_data, kernel, sigma, ell
    )
    K_metric = (K_metric + K_metric.T) / 2
    K_jax = K_metric.astype(dtype)
    K_tilde_jax = K_tilde_metric.astype(dtype)

    def train_path_values(theta):
        return path_values(theta, K_jax, K_tilde_jax)

    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K_metric + eps * jnp.eye(n_train, dtype=jnp.float64))
    )

    alpha_metric = metric_with_compute_dtype(alpha_metric, jnp.float64)

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

    # 3n nonlinear residuals in 3n + 2 unknowns: LM with the unified shifted
    # metric M = blockdiag(K, K, K, 0) + eps I selects the interpolant
    # minimizing the summed RKHS norms (the original objective) with free
    # initial values (c_0, mu_0), up to an O(eps) selection shift.
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

    c_flow_guess = max(production(k_0) - delta * k_0, DOMAIN_EPS)
    if c_0_guess is not None:
        c_flow_guess = max(float(c_0_guess), DOMAIN_EPS)
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.array([c_flow_guess, 1.0 / c_flow_guess], dtype=dtype),
    )

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        linear_solve_dtype=jnp.float64,
        metric=metric,
    )

    # The timed solve below reuses this solver instance, so its jit cache is
    # warm; max_steps/atol are dynamic and do not recompile.
    compile_start = time.perf_counter()
    jax.block_until_ready(
        solver.solve(
            theta_0,
            max_steps=1,
            atol=lm_set.residual_atol,
            max_steps_is_success=False,
        )
    )
    compile_time = time.perf_counter() - compile_start

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
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    residual_names = ("resource", "euler", "shadow_price")
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )

    def mpk_violations(grid):
        m1 = A_value * a * grid["k"] ** (a - 1.0)
        m2 = b_1_value * m1
        implied = marginal_product(grid["k"]) - grid["euler"] / grid["mu"]
        lower = jnp.minimum(m1, m2)
        upper = jnp.maximum(m1, m2)
        return jnp.maximum(lower - implied, 0.0), jnp.maximum(implied - upper, 0.0)

    train_lower, train_upper = mpk_violations(train_eval)
    validation_lower, validation_upper = mpk_violations(validation_eval)
    p_lower_violation = max(
        float(jnp.max(train_lower)), float(jnp.max(validation_lower))
    )
    p_upper_violation = max(
        float(jnp.max(train_upper)), float(jnp.max(validation_upper))
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
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "c", "mu")
        )
        and all(
            jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "c", "mu")
        )
    )

    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if use_kinked_production and max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    if use_kinked_production and (
        p_lower_violation > MPK_BOUND_TOL or p_upper_violation > MPK_BOUND_TOL
    ):
        rejection_reasons.append("p_bound")
    solution_accepted = not rejection_reasons

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "k_0": k_0,
        "A": A_value,
        "b_1": b_1_value,
        "b_2": b_2_value,
        "use_kinked_production": use_kinked_production,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": "float64",
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
            alpha_metric.norm(alpha_k) ** 2 - eps * (alpha_k @ alpha_k)
        ),
        "rkhs_norm_c": float(
            alpha_metric.norm(alpha_c) ** 2 - eps * (alpha_c @ alpha_c)
        ),
        "rkhs_norm_mu": float(
            alpha_metric.norm(alpha_mu) ** 2 - eps * (alpha_mu @ alpha_mu)
        ),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_eval["euler"]))),
        "solve_time": elapsed,
        "wall_time": elapsed,
        "solver": "nlls-gram",
        "linear_solver": "auto",
        "metric_eps": eps,
        "train_loss": float(result.info.loss),
        "steps": steps,
        "stopping_reason": STATUS_TO_REASON.get(status, "unknown"),
        "compile_time": compile_time,
        "solution_accepted": solution_accepted,
        "rejection_reason": "accepted"
        if solution_accepted
        else ",".join(rejection_reasons),
    }
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
    if not use_kinked_production:
        benchmark_solution = neoclassical_growth_benchmark(
            a, delta, rho_hat, 1.0, k_0, benchmark_grid
        )
        k_benchmark, c_benchmark = benchmark_solution(test_data)
        k_rel_error = jnp.abs(k_benchmark - test_eval["k"]) / k_benchmark
        c_rel_error = jnp.abs(c_benchmark - test_eval["c"]) / c_benchmark
        test_df["k_benchmark"] = np.asarray(k_benchmark)
        test_df["c_benchmark"] = np.asarray(c_benchmark)
        test_df["k_rel_error"] = np.asarray(k_rel_error)
        test_df["c_rel_error"] = np.asarray(c_rel_error)
        results.update(
            k_rel_error_mean=float(k_rel_error.mean()),
            c_rel_error_mean=float(c_rel_error.mean()),
            k_rel_error_max=float(k_rel_error[1:].max()),
            c_rel_error_max=float(c_rel_error[1:].max()),
            k_rel_error_min=float(k_rel_error[1:].min()),
            c_rel_error_min=float(c_rel_error[1:].min()),
        )
        print(
            f"solve_time(s) = {elapsed}, "
            f"E(|rel_error(k)|) = {results['k_rel_error_mean']:.3e}, "
            f"E(|rel_error(c)|) = {results['c_rel_error_mean']:.3e}"
        )
    else:
        print(f"solve_time(s) = {elapsed}")

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "A": A_value,
            "b_1": b_1_value,
            "b_2": b_2_value,
        },
        "initial_conditions": {"k_0": k_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel)
