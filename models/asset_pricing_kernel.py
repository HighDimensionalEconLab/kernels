from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    RidgeLevenbergMarquardt,
)

from models.asset_pricing_shared import (
    DataSettings,
    StructuralParameters,
    ToleranceSettings,
    dividends,
    kernel_horizon_bias_p0,
    mu_f_array,
)
from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    integrated_metric_factorization,
    rkhs_norm_squared,
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


# Alpha first (the metric penalizes the leading flat coordinates).
class Theta(NamedTuple):
    alpha: jax.Array
    p_0: jax.Array


# No-arbitrage residual dp/dt = r p - x along the dividend path.
def residual_fn(theta, args, p):
    paths = integrated_kernel_paths(
        {"price": IntegratedPath(theta.alpha, theta.p_0)}, args.K, args.K_tilde
    )
    price, dprice_dt = paths["price"]
    residual = dprice_dt - (p.r * price - dividends(args.batch["t"], p))
    return residual, {"p": price, "dp_dt": dprice_dt, "residual": residual}


def asset_pricing_kernel(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    # atol bounds the unscaled residual norm; gtol must resolve the weakly
    # penalized free p_0. Convergence requires gtol AND atol.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.5,
        damping_increase=4.0,
        max_steps=100,
        atol=1e-9,
        gtol=1e-13,
    ),
    ridge_set: RidgeSettings = RidgeSettings(ridge=1e-10),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = kernel_set.kernel
    ell, sigma = kernel_set.ell, kernel_set.sigma
    train_data = make_train_t_grid(data_set, dtype)
    test_data = jnp.linspace(0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    p = StructuralParameters.create(dtype=dtype, r=r, c=c, g=g, x_0=x_0)

    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=1
    )
    residual_args = KernelResidualArgs(batch={"t": train_data}, K=K, K_tilde=K_tilde)

    solver = RidgeLevenbergMarquardt(
        residual_fn,
        has_aux=True,
        metric=metric,
        ridge=ridge_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )
    theta_0 = Theta(
        alpha=jnp.zeros(len(train_data), dtype=dtype),
        p_0=jnp.zeros((), dtype=dtype),
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
    alpha, p_0 = theta.alpha, theta.p_0
    steps = int(result.steps)
    train_values = residual_fn(theta, residual_args, p)[1]
    p_train = train_values["p"]
    train_residual = train_values["residual"]
    max_train_residual = float(jnp.max(jnp.abs(train_residual)))

    # Paths and the no-arbitrage residual on an evaluation grid through fresh
    # cross-kernel matrices against the training grid.
    @jax.jit
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points, train_data, kernel, sigma, ell
        )
        return residual_fn(
            theta,
            KernelResidualArgs(batch={"t": points}, K=K_eval, K_tilde=K_tilde_eval),
            p,
        )[1]

    test_values = evaluate_grid(test_data)
    p_test = test_values["p"]
    test_residual = test_values["residual"]
    p_benchmark = mu_f_array(test_data, p)
    p_rel_error = jnp.abs(p_benchmark - p_test) / p_benchmark

    validation_values = evaluate_grid(validation_data)
    p_validation = validation_values["p"]
    max_validation_residual = float(jnp.max(jnp.abs(validation_values["residual"])))
    mean_validation_residual = float(jnp.mean(jnp.abs(validation_values["residual"])))

    # Analytic finite-horizon bias in the matern12 case.
    horizon_bias_p0 = kernel_horizon_bias_p0(p, kernel_set, T=float(train_data[-1]))

    finite = bool(
        jnp.all(jnp.isfinite(p_test))
        and jnp.all(jnp.isfinite(p_benchmark))
        and jnp.all(jnp.isfinite(p_validation))
    )
    rejection_reasons = []
    if not finite:
        rejection_reasons.append("nonfinite")
    if float(p_0) < 0.0:
        rejection_reasons.append("negative_p_0")
    rejection_reasons += gate_rejections(
        tolerance_set, max_train_residual, mean_validation_residual
    )

    results = {
        "parameters": p.to_dict(),
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
        "p_0": float(p_0),
        "horizon_bias_p0": horizon_bias_p0,
        "rkhs_norm_p": float(rkhs_norm_squared(alpha, K)),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "mean_validation_residual": mean_validation_residual,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
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
    return results, {
        "model": evaluate_grid,
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
    jsonargparse.CLI(asset_pricing_kernel)
