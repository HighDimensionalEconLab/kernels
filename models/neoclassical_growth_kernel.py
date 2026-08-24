from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from nlls_gram import RidgeLevenbergMarquardt

from models.kernels import (
    KernelSettings,
    integrated_metric_factorization,
)
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    DataSettings,
    StructuralParameters,
    Theta,
    evaluate_paths,
    residual_fn,
    snapshot,
    steady_flow_consumption_guess,
)
from models.utilities import (
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)


# CLI-reachable bars; only train_residual gates acceptance.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    log_tvc_threshold: float = 0.0


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 50
    residual_atol: float = 2e-8  # on the 1/sqrt(3n)-scaled residual norm
    # Ridge weight of ||r||^2 + ridge * sum_j alpha_j' K alpha_j with the free
    # (c_0, mu_0) unpenalized; the converged residual norm is
    # O(ridge * ||dual||), so ridge must sit low enough that residual_atol
    # stays reachable. Convergence requires gtol AND residual_atol.
    # Tuning: https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/
    ridge: float = 1e-10
    gtol: float = 1e-9


def neoclassical_growth_kernel(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    k_0: float = 1.0,
    # Hicks-neutral technology z(t) = z_0 e^{(1-a) g t}: the unit trend enters
    # the residuals as data, z_0 rides in p; learn_g_hat frees the trend rate
    # (init 0) instead of fixing it at g.
    z_0: float = 1.0,
    g: float = 0.0,
    learn_g_hat: bool = False,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    c_0_guess: float | None = None,  # overrides the flat steady-flow warm start
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    # Kinked production is enabled exactly when all of A, b_1, b_2 are given.
    use_kinked_production = not (A is None or b_1 is None or b_2 is None)
    if use_kinked_production:
        A, b_1, b_2 = float(A), float(b_1), float(b_2)
        k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a)
    else:
        A, b_1, b_2, k_thr = 1.0, 1.0, 0.0, float("inf")

    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, dtype)
    n_train = len(train_data)

    # Everything in p, the kink family included, is implicitly
    # differentiable through the solve (nlls-gram's implicit JVP/VJP at the
    # ridge stationary point).
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=z_0,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        k_thr=k_thr,
    )

    # 3n nonlinear residuals in 3n + 2 unknowns: ridge LM on
    # ||r||^2 + ridge * (a_mu'K a_mu + a_c'K a_c + a_k'K a_k) selects the
    # interpolant minimizing the summed RKHS norms with the trailing
    # (c_0, mu_0) in the unpenalized free block.
    K_jax, K_tilde_jax, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=3
    )

    def z_path(t_grid):
        return z_0 * jnp.exp((1.0 - a) * g * t_grid) if g != 0.0 else z_0

    g_hat_fixed = jnp.asarray(g, dtype=dtype) if g != 0.0 and not learn_g_hat else None
    residual_args = {
        "K": K_jax,
        "K_tilde": K_tilde_jax,
        "t": train_data,
        "z_trend": jnp.exp((1.0 - a) * g * train_data) if g != 0.0 else 1.0,
        "g_hat": g_hat_fixed,
    }

    c_flow_guess = steady_flow_consumption_guess(z_0, k_0, a, delta, A, b_1, b_2)
    if c_0_guess is not None:
        c_flow_guess = float(c_0_guess)
    theta_0 = Theta(
        alpha_mu=jnp.zeros(n_train, dtype=dtype),
        alpha_c=jnp.zeros(n_train, dtype=dtype),
        alpha_k=jnp.zeros(n_train, dtype=dtype),
        c_0=jnp.asarray(c_flow_guess, dtype=dtype),
        mu_0=jnp.asarray(c_flow_guess**-sigma_crra, dtype=dtype),
        g_hat=jnp.zeros((), dtype=dtype) if learn_g_hat else None,
    )

    # Ridge Levenberg-Marquardt solve.
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
    result = solver.solve(theta_0, residual_args, p=p, **solve_options)
    elapsed = timer.elapsed(result)

    theta = result.x
    c_0, mu_0 = theta.c_0, theta.mu_0
    steps = int(result.steps)

    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    test_benchmark = (
        None
        if use_kinked_production
        else neoclassical_growth_benchmark(
            a, delta, rho_hat, sigma_crra, g=g, z_0=z_0, k_0=k_0
        )
    )
    train_df = snapshot(
        theta,
        train_data,
        p,
        train_data,
        kernel_set,
        g_hat_fixed,
        z=z_path(train_data),
    )
    test_df = snapshot(
        theta,
        test_data,
        p,
        train_data,
        kernel_set,
        g_hat_fixed,
        benchmark=test_benchmark,
        z=z_path(test_data),
    )

    tvc_t = jnp.asarray([data_set.test_tvc_T], dtype=dtype)
    paths_T = evaluate_paths(theta, tvc_t, train_data, kernel_set, p, g_hat_fixed)
    k_T, c_T = paths_T["k"].value, paths_T["c"].value
    log_tvc = float(
        -rho_hat * data_set.test_tvc_T - sigma_crra * jnp.log(c_T[0]) + jnp.log(k_T[0])
    )
    transversality_ok = bool(log_tvc < tolerance_set.log_tvc_threshold)

    finite_positive = bool(
        all(
            np.isfinite(df[name]).all() and (df[name] > 0.0).all()
            for df in (train_df, test_df)
            for name in ("k", "c", "mu")
        )
    )
    max_train_residual = float(
        max(
            train_df["resource"].abs().max(),
            train_df["euler"].abs().max(),
            train_df["shadow_price"].abs().max(),
        )
    )
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")

    g_hat_value = float(theta.g_hat) if learn_g_hat else float(g)

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "z_0": z_0,
        "k_0": k_0,
        "g": g,
        "g_hat": g_hat_value,
        "g_hat_error": g_hat_value - g,
        "A": A,
        "b_1": b_1,
        "b_2": b_2,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(result.x.alpha_mu.dtype),
        "kernel": kernel,
        "ell": ell,
        "sigma": sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": tolerance_set.train_residual,
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "test_tvc_T": data_set.test_tvc_T,
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "solve_time": elapsed,
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": steps,
        **acceptance_fields(rejection_reasons),
    }
    if not use_kinked_production:
        k_rel_error = test_df["k_rel_error"].to_numpy()
        c_rel_error = test_df["c_rel_error"].to_numpy()
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
        if verbose:
            print(
                f"solve_time(s) = {elapsed}, "
                f"E(|rel_error(k)|) = {results['test_k_rel_error_mean']:.3e}, "
                f"E(|rel_error(c)|) = {results['test_c_rel_error_mean']:.3e}"
            )
    elif verbose:
        print(f"solve_time(s) = {elapsed}")

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "g": g,
            "learn_g_hat": learn_g_hat,
            "A": A,
            "b_1": b_1,
            "b_2": b_2,
        },
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
    }

    @jax.jit
    def kernel_solution(t_grid):
        paths = evaluate_paths(theta, t_grid, train_data, kernel_set, p, g_hat_fixed)
        return (
            paths["k"].value,
            paths["c"].value,
            paths["mu"].value,
            paths["k"].d_dt,
            paths["mu"].d_dt,
        )

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
            "residual_args": residual_args,
            "multi_start": None,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel)
