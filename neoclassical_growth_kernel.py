from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

from kernels import (
    integrated_kernel_matrices,
    normalize_kernel,
    rkhs_norm_squared,
)
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)

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


# Accuracy is truncation- then density-limited: halving dt below 1.0 keeps
# improving errors at unchanged LM cost but slows the uno variant.
@dataclass(frozen=True)
class DataSettings:
    train_T: float = CONSTANTS.train_T
    train_points: int = 101
    train_points_list: list[float] | None = None
    test_T: float = CONSTANTS.sequence_test_T
    test_points: int = 100
    benchmark_T: float = CONSTANTS.benchmark_T
    benchmark_points: int = CONSTANTS.benchmark_points


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
    # The fixed ridge weight of the objective ||r||^2 + ridge * q(theta) with
    # q = sum_j alpha_j' K alpha_j and the free (c_0, mu_0) unpenalized (no
    # epsilon shift anywhere): the minimizer is O(ridge) from the pure
    # RKHS-seminorm interpolant. The converged residual norm is
    # O(ridge * ||dual||) (~25x ridge measured here), so ridge must sit low
    # enough that residual_atol is reachable; 1e-10 leaves the selection
    # unchanged (the null space is 2-dimensional and steeply penalized) while
    # meeting the same atol as the metric formulation in 5 steps. gtol bounds
    # the ridge stationarity ||J'r + ridge K alpha||; convergence requires
    # gtol AND residual_atol (conjunctive). Calibration (the
    # ~1e-3 * ridge * info.penalty_grad_norm recipe): ridge*||L'L theta|| is
    # ~4.6e-11 at convergence, so 1e-9 resolves the selection well below the
    # quality gates while sitting far above the ~1e-14 gradient floor.
    # Tuning: https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/
    ridge: float = 1e-10
    gtol: float = 1e-9


# The shared NEOCLASSICAL_GROWTH_CONSTANTS (utilities.py) maps the Aruoba,
# Fernandez-Villaverde & Rubio-Ramirez (2006 JEDC) benchmark values.
def neoclassical_growth_kernel(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
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
    if sigma_crra <= 0.0:
        raise ValueError("sigma_crra must be positive")
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

    # The residual is scaled by 1/sqrt(3n) (the NN files' convention) so that
    # residual_atol is a scale-free per-point bound rather than tightening
    # with grid refinement.
    residual_scale = 1.0 / jnp.sqrt(3.0 * n_train)

    # CRRA first-order condition u'(c) = mu, i.e. mu c^sigma_crra = 1; the
    # costate and resource residuals are utility-free.
    def residual_fn(theta):
        k, c, mu, dk_dt, dmu_dt = train_path_values(theta)
        resource = dk_dt - (production(k) - delta * k - c)
        euler = dmu_dt + mu * (marginal_product(k) - delta - rho_hat)
        shadow_price = mu * c**sigma_crra - 1.0
        return residual_scale * jnp.concatenate([resource, euler, shadow_price])

    # 3n nonlinear residuals in 3n + 2 unknowns: ridge LM on
    # ||r||^2 + ridge * (a_mu'K a_mu + a_c'K a_c + a_k'K a_k) selects the
    # interpolant minimizing the summed RKHS norms (the original objective)
    # with the trailing (c_0, mu_0) in the unpenalized free block. The metric
    # takes the upper Cholesky factor F with K = F'F, computed in float64 and
    # cast to the residual dtype.
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(K_metric, upper=True).astype(dtype),
        repeats=3,
    )

    c_flow_guess = max(production(k_0) - delta * k_0, DOMAIN_EPS)
    if c_0_guess is not None:
        c_flow_guess = max(float(c_0_guess), DOMAIN_EPS)
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.array([c_flow_guess, c_flow_guess**-sigma_crra], dtype=dtype),
    )

    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=lm_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )

    start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
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
            "shadow_price": mu * c**sigma_crra - 1.0,
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
        "sigma_crra": sigma_crra,
        "k_0": k_0,
        "A": A_value,
        "b_1": b_1_value,
        "b_2": b_2_value,
        "use_kinked_production": use_kinked_production,
        "residual_dtype": str(jnp.dtype(dtype)),
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        # Unshifted alpha' K alpha, matching the IPOPT report (the metric
        # norm includes the epsilon shift).
        "rkhs_norm_k": float(rkhs_norm_squared(alpha_k, K_metric)),
        "rkhs_norm_c": float(rkhs_norm_squared(alpha_c, K_metric)),
        "rkhs_norm_mu": float(rkhs_norm_squared(alpha_mu, K_metric)),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_eval["euler"]))),
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
            a, delta, rho_hat, sigma_crra, k_0, benchmark_grid
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
            "sigma_crra": sigma_crra,
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
