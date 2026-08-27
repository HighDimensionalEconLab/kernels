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
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
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
    p0_init: float = 0.0,  # seeds theta_0's free p_0 coefficient
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    # atol is on the unscaled residual norm, so it bounds max|residual|; the
    # converged residual floors at O(ridge * ||dual||), so atol must sit above
    # it. Convergence requires gtol AND atol, and gtol must be tight enough to
    # resolve the weakly penalized p_0 direction. Tuning:
    # https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.5,
        damping_increase=4.0,
        max_steps=100,
        atol=1e-9,
        gtol=1e-13,
    ),
    # Ridge weight of ||r||^2 + ridge * a'Ka with the free p_0 unpenalized.
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
    n_train = len(train_data)
    p = StructuralParameters.create(dtype=dtype, r=r, c=c, g=g, x_0=x_0)

    # The residual is affine in theta, so each ridge-LM step is an exact
    # kernel ridge solve converging to the minimizer of ||r||^2 + ridge * a'Ka
    # with p_0 in the unpenalized free block -- the min-RKHS-norm selection
    # (the original a'Ka objective) up to an O(ridge) bias.
    K_metric, K_tilde_jax, metric = integrated_metric_factorization(
        train_data, kernel, sigma, ell, repeats=1
    )
    residual_args = KernelResidualArgs(
        batch={"t": train_data}, K=K_metric, K_tilde=K_tilde_jax
    )

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
        alpha=jnp.zeros(n_train, dtype=dtype),
        p_0=jnp.asarray(p0_init, dtype=dtype),
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
    # stalls at its residual floor below the train gate is feasible (the
    # IPOPT baseline's FEASIBLE_SMALL_STEP semantics) -- the smooth kernels
    # routinely end there because atol is unreachable for their
    # ill-conditioned Grams. stopping_reason still reports the code.
    train_residual_tolerance = max(
        tolerance_set.train_residual,
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

    results = {
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
        "tolerance_set": asdict(tolerance_set),
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "p_0": float(p_0),
        "horizon_bias_p0": horizon_bias_p0,
        "rkhs_norm_p": float(rkhs_norm_squared(alpha, K_metric)),
        "max_train_residual": max_train_residual,
        "train_residual_tolerance": train_residual_tolerance,
        "max_test_residual": float(jnp.max(jnp.abs(test_residual))),
        # min/max error statistics exclude the pinned t = 0 point ([1:]).
        "p_rel_error_mean": float(p_rel_error.mean()),
        "p_rel_error_max": float(p_rel_error[1:].max()),
        "p_rel_error_min": float(p_rel_error[1:].min()),
        "solve_time": elapsed,
        "train_loss": float(result.info.resid_loss),
        "final_damping": float(result.lm_state.damping),
        "steps": steps,
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
