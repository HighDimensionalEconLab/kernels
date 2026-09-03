from __future__ import annotations

from dataclasses import asdict
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
    ToleranceSettings,
    mpk,
    production,
)
from models.utilities import (
    STATUS_TO_REASON,
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
    gate_rejections,
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
    values = {
        "k": k,
        "c": c,
        "mu": mu,
        "dk_dt": dk_dt,
        "dmu_dt": dmu_dt,
        "resource": dk_dt - (production(z, k, p) - p.delta * k - c),
        "euler": dmu_dt + mu * (mpk(z, k, p) - p.delta - p.rho_hat),
        "shadow_price": mu * c**p.sigma_crra - 1.0,
    }
    residual = jnp.concatenate(
        [values["resource"], values["euler"], values["shadow_price"]]
    )
    return residual / jnp.sqrt(residual.size), values


def neoclassical_growth_kernel(
    a: float = CONSTANTS["a"],
    delta: float = CONSTANTS["delta"],
    rho_hat: float = CONSTANTS["rho_hat"],
    sigma_crra: float = CONSTANTS["sigma_crra"],
    k_0: float = 1.0,
    z_0: float = 1.0,
    g: float = 0.0,
    learn_g_hat: bool = False,
    g_hat_init: float = 0.0,  # theta_0 trend entry when learned
    A: float = 1.0,
    b_1: float = 1.0,
    b_2: float = 0.0,
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
    ridge_set: RidgeSettings = RidgeSettings(ridge=1e-10),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a) if b_1 > 1.0 else float("inf")

    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, dtype)
    n_train = len(train_data)

    # Everything in p, the butterfly family included, is implicitly
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

    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=3
    )

    residual_args = KernelResidualArgs(
        batch={
            "t": train_data,
            "z_trend": jnp.exp((1.0 - a) * g * train_data),
            "g_hat": jnp.asarray(g, dtype=dtype),
        },
        K=K,
        K_tilde=K_tilde,
    )

    c_0_init = production(p.z_0, p.k_0, p) - p.delta * p.k_0
    theta_0 = Theta(
        alpha_mu=jnp.zeros(n_train, dtype=dtype),
        alpha_c=jnp.zeros(n_train, dtype=dtype),
        alpha_k=jnp.zeros(n_train, dtype=dtype),
        c_0=c_0_init,
        mu_0=c_0_init**-sigma_crra,
        g_hat=jnp.asarray(g_hat_init, dtype=dtype) if learn_g_hat else None,
    )

    # Ridge Levenberg-Marquardt solve.
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        has_aux=True,
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
        values = residual_fn(
            theta,
            KernelResidualArgs(
                batch={
                    "t": points,
                    "z_trend": jnp.exp((1.0 - a) * g * points),
                    "g_hat": jnp.asarray(g, dtype=dtype),
                },
                K=K_eval,
                K_tilde=K_tilde_eval,
            ),
            p,
        )[1]
        return {"t": points, **values}

    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    test_benchmark = neoclassical_growth_benchmark(p)
    train_eval = evaluate_grid(train_data)
    test_eval = evaluate_grid(test_data)
    validation_eval = evaluate_grid(validation_data)
    frame_columns = ("t", "k", "c", "mu", "resource", "euler", "shadow_price")
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_columns}
    )
    test_df = pd.DataFrame(
        {name: np.asarray(test_eval[name]) for name in frame_columns}
    )
    validation_df = pd.DataFrame(
        {name: np.asarray(validation_eval[name]) for name in frame_columns}
    )
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
            for df in (train_df, test_df, validation_df)
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
    max_validation_residual = float(
        max(
            validation_df["resource"].abs().max(),
            validation_df["euler"].abs().max(),
            validation_df["shadow_price"].abs().max(),
        )
    )
    mean_validation_residual = float(
        validation_df[list(("resource", "euler", "shadow_price"))]
        .abs()
        .to_numpy()
        .mean()
    )
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    rejection_reasons += gate_rejections(
        tolerance_set, max_train_residual, mean_validation_residual
    )

    g_hat_value = float(theta.g_hat) if learn_g_hat else float(g)

    results = {
        "parameters": {**p.to_dict(), "learn_g_hat": learn_g_hat},
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": {
            "init_damping": lm_set.init_damping,
            "damping_decrease": lm_set.damping_decrease,
            "damping_increase": lm_set.damping_increase,
            "max_steps": lm_set.max_steps,
            "atol": lm_set.atol,
            "gtol": lm_set.gtol,
        },
        "ridge_set": {"ridge": ridge_set.ridge},
        "tolerance_set": asdict(tolerance_set),
        "g_hat": g_hat_value,
        "g_hat_error": g_hat_value - g,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "mean_validation_residual": mean_validation_residual,
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "test_k_rel_error_abs": float(test_df["k_rel_error"].mean()),
        "test_c_rel_error_abs": float(test_df["c_rel_error"].mean()),
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "solve_time": elapsed,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": steps,
        "solver_status": STATUS_TO_REASON.get(int(result.status), "unknown"),
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        print(
            f"solve_time(s) = {elapsed}, "
            f"E(|rel_error(k)|) = {results['test_k_rel_error_abs']:.3e}, "
            f"E(|rel_error(c)|) = {results['test_c_rel_error_abs']:.3e}"
        )

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
