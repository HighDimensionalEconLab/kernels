from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import RidgeLevenbergMarquardt

from models.kernels import (
    integrated_kernel_matrices,
    integrated_metric_factorization,
    normalize_kernel,
    rkhs_norm_squared,
)
from models.neoclassical_growth import (
    DOMAIN_EPS,
    StructuralParameters,
    marginal_product,
    production,
    resolve_kink_family,
    steady_flow_consumption_guess,
)
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.utilities import (
    STATUS_TO_REASON,
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)


# theta is the tuple (alpha_mu, alpha_c, alpha_k, (c_0, mu_0)), which
# flattens to [alpha_mu (n), alpha_c (n), alpha_k (n), c_0, mu_0] -- the
# block order of the solver metric. Module-level so callers can evaluate
# solved paths on their own kernel matrices.
def path_values(theta, K_eval, K_tilde_eval, p):
    alpha_mu, alpha_c, alpha_k, initial = theta
    c_0, mu_0 = initial
    mu = mu_0 + K_tilde_eval @ alpha_mu
    c = c_0 + K_tilde_eval @ alpha_c
    k = p.k_0 + K_tilde_eval @ alpha_k
    dmu_dt = K_eval @ alpha_mu
    dk_dt = K_eval @ alpha_k
    return k, c, mu, dk_dt, dmu_dt


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


# Acceptance gates, CLI-reachable so a float32 caller can pass its own bar
# (the float32 replication driver does). The mpk bound applies only under
# kinked production.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 5e-3
    mpk_bound: float = 1e-3


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
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    c_0_guess: float | None = None,  # overrides the flat steady-flow warm start
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    use_kinked_production, A_value, b_1_value, b_2_value = resolve_kink_family(
        A, b_1, b_2
    )

    kernel = normalize_kernel(kernel_set.kernel)
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, dtype)
    n_train = len(train_data)

    # The structural parameters -- the kink family included -- and the
    # initial condition all ride in the solver's `p` argument so the solved
    # paths are implicitly differentiable with respect to every one of them
    # (nlls-gram's implicit JVP/VJP at the ridge stationary point).
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        k_0=k_0,
        A=A_value,
        b_1=b_1_value,
        b_2=b_2_value,
    )

    # 3n nonlinear residuals in 3n + 2 unknowns: ridge LM on
    # ||r||^2 + ridge * (a_mu'K a_mu + a_c'K a_c + a_k'K a_k) selects the
    # interpolant minimizing the summed RKHS norms (the original objective)
    # with the trailing (c_0, mu_0) in the unpenalized free block.
    K_jax, K_tilde_jax, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=3
    )

    # The residual is scaled by 1/sqrt(3n) (the NN files' convention) so that
    # residual_atol is a scale-free per-point bound rather than tightening
    # with grid refinement.
    residual_scale = 1.0 / jnp.sqrt(3.0 * n_train)

    # CRRA first-order condition u'(c) = mu, i.e. mu c^sigma_crra = 1; the
    # costate and resource residuals are utility-free. p carries the
    # structural parameters and the initial condition for implicit
    # differentiation; the model equations never read args.
    def residual_fn(theta, _, p):
        k, c, mu, dk_dt, dmu_dt = path_values(theta, K_jax, K_tilde_jax, p)
        resource = dk_dt - (production(k, p) - p.delta * k - c)
        euler = dmu_dt + mu * (marginal_product(k, p) - p.delta - p.rho_hat)
        shadow_price = mu * c**p.sigma_crra - 1.0
        return residual_scale * jnp.concatenate([resource, euler, shadow_price])

    c_flow_guess = steady_flow_consumption_guess(
        k_0, a, delta, A_value, b_1_value, b_2_value
    )
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

    solve_options = dict(
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )

    timer = Stopwatch()
    result = solver.solve(theta_0, None, p=p, **solve_options)
    elapsed = timer.elapsed(result)

    theta = result.x
    alpha_mu, alpha_c, alpha_k, initial = theta
    c_0, mu_0 = initial
    steps = int(result.steps)
    status = int(result.status)

    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_points = max(data_set.test_points, data_set.benchmark_points)
    validation_data = jnp.linspace(0, data_set.train_T, validation_points, dtype=dtype)

    def evaluate_grid(points_data):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points_data, train_data, kernel, sigma, ell
        )
        k, c, mu, dk_dt, dmu_dt = path_values(theta, K_eval, K_tilde_eval, p)
        return {
            "t": points_data,
            "k": k,
            "c": c,
            "mu": mu,
            "Y": production(k, p),
            "resource": dk_dt - (production(k, p) - p.delta * k - c),
            "euler": dmu_dt + mu * (marginal_product(k, p) - p.delta - p.rho_hat),
            "shadow_price": mu * c**p.sigma_crra - 1.0,
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
        m1 = p.A * p.a * grid["k"] ** (p.a - 1.0)
        m2 = p.b_1 * m1
        implied = marginal_product(grid["k"], p) - grid["euler"] / grid["mu"]
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
        k, c, _, _, _ = path_values(theta, K_test, K_tilde_test, p)
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
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    if (
        use_kinked_production
        and max_validation_residual > tolerance_set.validation_residual
    ):
        rejection_reasons.append("validation_residual")
    if use_kinked_production and (
        p_lower_violation > tolerance_set.mpk_bound
        or p_upper_violation > tolerance_set.mpk_bound
    ):
        rejection_reasons.append("p_bound")

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
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["euler"].dtype),
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "rkhs_norm_k": float(rkhs_norm_squared(alpha_k, K_jax)),
        "rkhs_norm_c": float(rkhs_norm_squared(alpha_c, K_jax)),
        "rkhs_norm_mu": float(rkhs_norm_squared(alpha_mu, K_jax)),
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": tolerance_set.train_residual,
        "max_validation_residual": max_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_eval["euler"]))),
        "solve_time": elapsed,
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
        benchmark_grid = jnp.linspace(
            0, data_set.benchmark_T, data_set.benchmark_points, dtype=dtype
        )
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
        # min/max error statistics exclude the pinned t = 0 point ([1:]): its
        # near-zero error is an artifact of the initial condition, not accuracy.
        results.update(
            test_k_rel_error_mean=float(k_rel_error.mean()),
            test_c_rel_error_mean=float(c_rel_error.mean()),
            test_k_rel_error_max=float(k_rel_error[1:].max()),
            test_c_rel_error_max=float(c_rel_error[1:].max()),
            test_k_rel_error_min=float(k_rel_error[1:].min()),
            test_c_rel_error_min=float(c_rel_error[1:].min()),
        )
        print(
            f"solve_time(s) = {elapsed}, "
            f"E(|rel_error(k)|) = {results['test_k_rel_error_mean']:.3e}, "
            f"E(|rel_error(c)|) = {results['test_c_rel_error_mean']:.3e}"
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
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": p,
            "residual_args": None,
            "multi_start": None,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel)
