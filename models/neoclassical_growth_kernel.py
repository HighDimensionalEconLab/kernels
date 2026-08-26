from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import RidgeLevenbergMarquardt

from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    integrated_metric_factorization,
)
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    DataSettings,
    StructuralParameters,
    mpk,
    production,
    steady_flow_consumption_guess,
)
from models.utilities import (
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
)


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# g_hat is the trend rate of the approximation class, None unless learned.
class Theta(NamedTuple):
    alpha_mu: jax.Array
    alpha_c: jax.Array
    alpha_k: jax.Array
    c_0: jax.Array
    mu_0: jax.Array
    g_hat: jax.Array | None = None


# CRRA first-order condition u'(c) = mu; the costate and resource residuals
# are utility-free. batch["g_hat"] is the fixed trend rate (0 without
# growth); a learned theta.g_hat overrides it.
def residual_fn(theta, args, p):
    g_hat = theta.g_hat if theta.g_hat is not None else args.batch["g_hat"]
    trend_mu = -p.sigma_crra * g_hat
    paths = integrated_kernel_paths(
        {
            "mu": IntegratedPath(theta.alpha_mu, theta.mu_0, trend_mu),
            "c": IntegratedPath(theta.alpha_c, theta.c_0, g_hat),
            "k": IntegratedPath(theta.alpha_k, p.k_0, g_hat),
        },
        args.K,
        args.K_tilde,
        args.batch["t"],
    )
    k, dk_dt = paths["k"]
    mu, dmu_dt = paths["mu"]
    c = paths["c"].value
    z = p.z_0 * args.batch["z_trend"]
    resource = dk_dt - (production(z, k, p) - p.delta * k - c)
    euler = dmu_dt + mu * (mpk(z, k, p) - p.delta - p.rho_hat)
    shadow_price = mu * c**p.sigma_crra - 1.0
    residual = jnp.concatenate([resource, euler, shadow_price])
    return residual / jnp.sqrt(residual.size)


# CLI-reachable bars; only train_residual gates acceptance.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    log_tvc_threshold: float = 0.0


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
    # atol is on the 1/sqrt(3n)-scaled residual norm; convergence requires
    # gtol AND atol.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.1,
        damping_increase=4.0,
        max_steps=50,
        atol=2e-8,
        gtol=1e-9,
    ),
    # Ridge weight of ||r||^2 + ridge * sum_j alpha_j' K alpha_j with the free
    # (c_0, mu_0) unpenalized; the converged residual norm is
    # O(ridge * ||dual||), so ridge must sit low enough that atol stays
    # reachable. Tuning:
    # https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/
    ridge_set: RidgeSettings = RidgeSettings(ridge=1e-10),
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
        dtype=dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=z_0,
        k_0=k_0,
        g=g,
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
        return z_0 * jnp.exp((1.0 - a) * g * t_grid)

    g_hat_fixed = jnp.asarray(g, dtype=dtype) if g != 0.0 and not learn_g_hat else None
    residual_args = KernelResidualArgs(
        batch={
            "t": train_data,
            "z_trend": jnp.exp((1.0 - a) * g * train_data),
            "g_hat": jnp.asarray(g, dtype=dtype),
        },
        K=K_jax,
        K_tilde=K_tilde_jax,
    )

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
        ridge=ridge_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )

    solve_options = dict(
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.atol,
        max_steps_is_success=False,
    )

    timer = Stopwatch()
    result = solver.solve(theta_0, residual_args, p=p, **solve_options)
    elapsed = timer.elapsed(result)

    theta = result.x
    c_0, mu_0 = theta.c_0, theta.mu_0
    steps = int(result.steps)

    # Paths and residual rows on an evaluation grid through fresh
    # cross-kernel matrices against the training grid.
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        g_hat = theta.g_hat if theta.g_hat is not None else g_hat_fixed
        trend_mu = None if g_hat is None else -p.sigma_crra * g_hat
        paths = integrated_kernel_paths(
            {
                "mu": IntegratedPath(theta.alpha_mu, theta.mu_0, trend_mu),
                "c": IntegratedPath(theta.alpha_c, theta.c_0, g_hat),
                "k": IntegratedPath(theta.alpha_k, p.k_0, g_hat),
            },
            K_eval,
            K_tilde_eval,
            points,
        )
        k, dk_dt = paths["k"]
        mu, dmu_dt = paths["mu"]
        c = paths["c"].value
        z = z_path(points)
        return {
            "t": points,
            "k": k,
            "c": c,
            "mu": mu,
            "dk_dt": dk_dt,
            "dmu_dt": dmu_dt,
            "resource": dk_dt - (production(z, k, p) - p.delta * k - c),
            "euler": dmu_dt + mu * (mpk(z, k, p) - p.delta - p.rho_hat),
            "shadow_price": mu * c**p.sigma_crra - 1.0,
        }

    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    test_benchmark = None if use_kinked_production else neoclassical_growth_benchmark(p)
    train_eval = evaluate_grid(train_data)
    test_eval = evaluate_grid(test_data)
    frame_columns = ("t", "k", "c", "mu", "resource", "euler", "shadow_price")
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_columns}
    )
    test_df = pd.DataFrame(
        {name: np.asarray(test_eval[name]) for name in frame_columns}
    )
    if test_benchmark is not None:
        k_b, c_b = test_benchmark(test_data)
        test_df["k_benchmark"] = np.asarray(k_b)
        test_df["c_benchmark"] = np.asarray(c_b)
        test_df["k_rel_error"] = np.abs(test_df["k"] - test_df["k_benchmark"]) / np.abs(
            test_df["k_benchmark"]
        )
        test_df["c_rel_error"] = np.abs(test_df["c"] - test_df["c_benchmark"]) / np.abs(
            test_df["c_benchmark"]
        )

    tvc_t = jnp.asarray([data_set.test_tvc_T], dtype=dtype)
    tvc_eval = evaluate_grid(tvc_t)
    k_T, c_T = tvc_eval["k"], tvc_eval["c"]
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
        "parameters": {**p.to_dict(), "learn_g_hat": learn_g_hat},
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
        "tolerance_set": asdict(tolerance_set),
        "g_hat": g_hat_value,
        "g_hat_error": g_hat_value - g,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "max_train_residual": max_train_residual,
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "solve_time": elapsed,
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

    @jax.jit
    def kernel_solution(t_grid):
        values = evaluate_grid(t_grid)
        return (
            values["k"],
            values["c"],
            values["mu"],
            values["dk_dt"],
            values["dmu_dt"],
        )

    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
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
