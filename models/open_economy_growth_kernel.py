from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    QR,
    AnnealRidge,
    Cholesky,
    LMStatus,
    RidgeLevenbergMarquardt,
)

from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    integrated_metric_factorization,
    rkhs_norm_squared,
)
from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.open_economy_growth_shared import (
    OPEN_ECONOMY_GROWTH_CONSTANTS as CONSTANTS,
)
from models.open_economy_growth_shared import (
    DataSettings,
    StructuralParameters,
    investment_expenditure,
    path_residuals,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.utilities import (
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
    make_sampled_train_grid,
)


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# the trailing scalars are the free log-anchors of the jump variables.
class Theta(NamedTuple):
    alpha_mu: jax.Array
    alpha_c: jax.Array
    alpha_k: jax.Array
    alpha_a: jax.Array
    alpha_q: jax.Array
    log_c_0: jax.Array
    log_mu_0: jax.Array
    log_q_0: jax.Array


def residual_fn(theta, args, p):
    # The asset block is the one path with no free scalar anchor, so its
    # scale enters the minimum-norm selection metric; the scale must be
    # differentiation-inert or every parameter sensitivity picks up a
    # selection-drift term from the metric moving with p. Time-0 output
    # sets it from the initial conditions alone.
    output_scale_p = jax.lax.stop_gradient(production(p.k_0, 1.0, p))
    paths = integrated_kernel_paths(
        {
            "log_mu": IntegratedPath(theta.alpha_mu, theta.log_mu_0),
            "log_c": IntegratedPath(theta.alpha_c, theta.log_c_0),
            "log_k": IntegratedPath(theta.alpha_k, jnp.log(p.k_0)),
            "a_raw": IntegratedPath(theta.alpha_a, jnp.zeros((), theta.alpha_a.dtype)),
            "log_q": IntegratedPath(theta.alpha_q, theta.log_q_0),
        },
        args.K,
        args.K_tilde,
    )
    k = jnp.exp(paths["log_k"].value)
    a = p.a_0 + output_scale_p * paths["a_raw"].value
    q = jnp.exp(paths["log_q"].value)
    c = jnp.exp(paths["log_c"].value)
    mu = jnp.exp(paths["log_mu"].value)
    dlog_k_dt = paths["log_k"].d_dt
    da_dt = output_scale_p * paths["a_raw"].d_dt
    dlog_q_dt = paths["log_q"].d_dt
    dlog_mu_dt = paths["log_mu"].d_dt
    residuals = path_residuals(
        k,
        a,
        q,
        c,
        mu,
        jnp.ones_like(k),
        dlog_k_dt,
        da_dt,
        dlog_q_dt,
        dlog_mu_dt,
        p,
    )
    residual = jnp.concatenate(
        [
            residuals["capital"],
            residuals["assets"],
            residuals["tobin_q"],
            residuals["marginal_wealth"],
            residuals["consumption_foc"],
        ]
    )
    return residual / jnp.sqrt(residual.size)


# Numerical acceptance and ex-post quality thresholds, CLI-reachable so a
# lower-precision caller can pass its own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 1e-2
    # The benchmark threshold is an HPO/quality diagnostic; it does not
    # redefine convergence of a finite, feasible numerical solution.
    benchmark_match: float = 1e-3


def open_economy_growth_kernel(
    alpha: float = CONSTANTS.alpha,
    gamma: float = CONSTANTS.gamma,
    delta: float = CONSTANTS.delta,
    r: float = CONSTANTS.r,
    eta: float = CONSTANTS.eta,
    r_star: float = CONSTANTS.r_star,
    k_0: float = 1.0,
    a_0: float = 0.0,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(ell=20.0),
    data_set: DataSettings = DataSettings(),
    # Conjunctive stopping in the whitened variable y = F_bar theta the
    # solver works in: gtol bounds the whitened stationarity
    # ||J~'r + ridge [y_m; 0]||, atol additionally demands the scaled
    # residual norm solved (the converged residual floors at
    # O(ridge * ||dual||), so atol sits above it).
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.1,
        damping_increase=4.0,
        max_steps=100,
        atol=5e-7,
        gtol=5e-8,
    ),
    # The ridge keeps the minimizer O(ridge) from the pure RKHS-seminorm
    # interpolant with the free offsets unpenalized (no epsilon shift
    # anywhere). ridge_init enables continuation annealed down to the ridge
    # floor by factors of ridge_decrease on inner stationarity; the metric's
    # factorization never involves the ridge, so the anneal composes
    # unchanged. linear_solver "auto" is the whitened cholesky: the whitened
    # normal matrix has its spectral floor at the ridge, so the DAE
    # Jacobian's scale spread stays within float64 ("qr"/"cholesky" remain
    # valid; all paths run whitened).
    ridge_set: RidgeSettings = RidgeSettings(
        ridge=3e-12,
        ridge_init=1e-7,
        ridge_decrease=0.01,
        ridge_grad_rtol=1e-2,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    kernel = kernel_set.kernel
    train_data = make_sampled_train_grid(data_set, dtype)

    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_data)
    # Everything in p is implicitly differentiable through the solve
    # (nlls-gram's implicit JVP/VJP at the ridge stationary point).
    p = StructuralParameters.create(
        dtype,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        r=r,
        eta=eta,
        r_star=r_star,
        k_0=k_0,
        a_0=a_0,
    )
    steady = steady_state(p)
    output_scale = steady["output"]
    consumption_scale = steady["g"]

    # There are 5n DAE equations in 5n + 3 unknowns. No terminal equation is
    # appended: the ridge penalty supplies the three-dimensional selection.
    # Every linear solver works on the whitened subproblem; the metric takes
    # the upper Cholesky factor F with K = F'F and the trailing offsets stay
    # in the unpenalized free block, untransformed.
    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, kernel_set.sigma, kernel_set.ell, repeats=5
    )

    residual_args = KernelResidualArgs(K=K, K_tilde=K_tilde)

    # Flat alpha blocks carry no saddle-path or present-value information;
    # the free anchors are seeded from the initial conditions alone:
    # consumption at the flow sustainable at k_0 (mu consistent via the
    # FOC), q at the replacement-investment price 1/eta.
    log_c_seed = jnp.log(production(p.k_0, 1.0, p) - p.delta * p.k_0)
    theta_0 = Theta(
        alpha_mu=jnp.zeros(n_train, dtype=dtype),
        alpha_c=jnp.zeros(n_train, dtype=dtype),
        alpha_k=jnp.zeros(n_train, dtype=dtype),
        alpha_a=jnp.zeros(n_train, dtype=dtype),
        alpha_q=jnp.zeros(n_train, dtype=dtype),
        log_c_0=log_c_seed,
        log_mu_0=-p.gamma * log_c_seed,
        log_q_0=-jnp.log(p.eta),
    )

    if ridge_set.ridge_init is None:
        initial_ridge = ridge_set.ridge
        continuation_options = {}
    else:
        initial_ridge = ridge_set.ridge_init
        callback = AnnealRidge(
            ridge_floor=ridge_set.ridge,
            decrease=ridge_set.ridge_decrease,
            grad_rtol=ridge_set.ridge_grad_rtol,
        )
        continuation_options = {
            "callback": callback,
            "user_state": callback.init_state(dtype),
        }
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=initial_ridge,
        linear_solver={"auto": Cholesky(), "cholesky": Cholesky(), "qr": QR()}[
            ridge_set.linear_solver
        ],
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
    result = solver.solve(
        theta_0, residual_args, p=p, **solve_options, **continuation_options
    )
    solve_time = timer.elapsed(result)
    theta = result.x

    # Paths, model quantities, and DAE residuals on an evaluation grid
    # through fresh cross-kernel matrices against the training grid.
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel_set.kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        output_scale_p = jax.lax.stop_gradient(production(p.k_0, 1.0, p))
        paths = integrated_kernel_paths(
            {
                "log_mu": IntegratedPath(theta.alpha_mu, theta.log_mu_0),
                "log_c": IntegratedPath(theta.alpha_c, theta.log_c_0),
                "log_k": IntegratedPath(theta.alpha_k, jnp.log(p.k_0)),
                "a_raw": IntegratedPath(
                    theta.alpha_a, jnp.zeros((), theta.alpha_a.dtype)
                ),
                "log_q": IntegratedPath(theta.alpha_q, theta.log_q_0),
            },
            K_eval,
            K_tilde_eval,
        )
        k = jnp.exp(paths["log_k"].value)
        a = p.a_0 + output_scale_p * paths["a_raw"].value
        q = jnp.exp(paths["log_q"].value)
        c = jnp.exp(paths["log_c"].value)
        mu = jnp.exp(paths["log_mu"].value)
        dlog_k_dt = paths["log_k"].d_dt
        da_dt = output_scale_p * paths["a_raw"].d_dt
        dlog_q_dt = paths["log_q"].d_dt
        dlog_mu_dt = paths["log_mu"].d_dt
        z = jnp.ones_like(k)
        residuals = path_residuals(
            k,
            a,
            q,
            c,
            mu,
            z,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
            p,
        )
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": q,
            "c": c,
            "mu": mu,
            "x": investment_expenditure(k, q, p),
            "output": production(k, z, p),
            "current_account": da_dt,
            **residuals,
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    tail = tail_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        p,
    )
    residual_names = (
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth",
        "consumption_foc",
    )
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in residual_names
    )

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(p)
    benchmark_values = benchmark(np.asarray(test_data))
    k_relative_error = jnp.abs(
        test_eval["k"] - jnp.asarray(benchmark_values["k"])
    ) / jnp.asarray(benchmark_values["k"])
    q_relative_error = jnp.abs(
        test_eval["q"] - jnp.asarray(benchmark_values["q"])
    ) / jnp.asarray(benchmark_values["q"])
    c_relative_error = jnp.abs(
        test_eval["c"] - jnp.asarray(benchmark_values["c"])
    ) / jnp.asarray(benchmark_values["c"])
    a_scaled_error = (
        jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
    )
    max_benchmark_error = max(
        float(jnp.max(k_relative_error)),
        float(jnp.max(q_relative_error)),
        float(jnp.max(c_relative_error)),
        float(jnp.max(a_scaled_error)),
    )

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(grid[name])) and jnp.min(grid[name]) > 0.0
            for grid in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "mu", "x")
        )
        and all(
            jnp.all(jnp.isfinite(grid["a"]))
            for grid in (train_eval, validation_eval, test_eval)
        )
    )
    c = train_eval["c"][0]
    pv_gap = present_value_gap(
        validation_eval["t"],
        validation_eval["k"],
        validation_eval["q"],
        validation_eval["c"],
        p,
    )
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        train_eval["t"][-1],
        p,
    )

    rejection_reasons = []
    if int(result.status) != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    validation_pass = max_validation_residual <= tolerance_set.validation_residual
    if not validation_pass:
        rejection_reasons.append("validation_residual")

    alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, log_c_0, log_mu_0, log_q_0 = theta
    rkhs_norm_mu = rkhs_norm_squared(alpha_mu, K)
    rkhs_norm_c = rkhs_norm_squared(alpha_c, K)
    rkhs_norm_k = rkhs_norm_squared(alpha_k, K)
    rkhs_norm_a = rkhs_norm_squared(alpha_a, K)
    rkhs_norm_q = rkhs_norm_squared(alpha_q, K)
    rkhs_norm_total = (
        rkhs_norm_mu + rkhs_norm_c + rkhs_norm_k + rkhs_norm_a + rkhs_norm_q
    )

    @jax.jit
    def kernel_solution(points):
        values = evaluate_grid(points)
        return values["k"], values["a"], values["q"], values["c"]

    results = {
        "alpha": alpha,
        "gamma": gamma,
        "delta": delta,
        "r": r,
        "eta": eta,
        "r_star": r_star,
        "r_star_resolved": r_star,
        "k_0": k_0,
        "a_0": a_0,
        "a_0_over_output_ss": a_0 / float(output_scale),
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["capital"].dtype),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "k_ss": float(steady["k"]),
        "q_ss": float(steady["q"]),
        "output_ss": float(output_scale),
        "g_ss": float(consumption_scale),
        "c": float(c),
        "q_0": float(jnp.exp(log_q_0)),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "validation_pass": validation_pass,
        "max_benchmark_error": max_benchmark_error,
        "differential_residual_count": 4 * n_train,
        "algebraic_residual_count": n_train,
        "dae_residual_count": 5 * n_train,
        "unknown_count": 5 * n_train + 3,
        "dynamic_rkhs_function_count": 4,
        "algebraic_rkhs_function_count": 1,
        "free_initial_value_count": 3,
        "metric_selection_dimension": 3,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "ex_post_present_value_gap": float(jnp.abs(pv_gap)),
        "ex_post_current_account": float(jnp.abs(train_eval["current_account"][-1])),
        "ex_post_stable_gap": abs(float(tail["stable"])),
        "ex_post_present_value_tail_gap": abs(float(tail["present_value"])),
        "ex_post_discounted_assets": float(tvc["discounted_assets"]),
        "ex_post_asset_tvc": float(tvc["asset_tvc"]),
        "ex_post_capital_tvc": float(tvc["capital_tvc"]),
        "benchmark_present_value_residual": abs(
            float(benchmark_diagnostics["present_value_residual"])
        ),
        "max_marginal_wealth_residual": float(
            jnp.max(jnp.abs(train_eval["marginal_wealth"]))
        ),
        "max_consumption_foc_residual": float(
            jnp.max(jnp.abs(train_eval["consumption_foc"]))
        ),
        "max_consumption_variation": float(
            jnp.max(jnp.abs(train_eval["c"] - train_eval["c"][0]))
        ),
        "rkhs_norm_mu": float(rkhs_norm_mu),
        "rkhs_norm_c": float(rkhs_norm_c),
        "rkhs_norm_k": float(rkhs_norm_k),
        "rkhs_norm_a": float(rkhs_norm_a),
        "rkhs_norm_q": float(rkhs_norm_q),
        "rkhs_norm_total": float(rkhs_norm_total),
        "solve_time": solve_time,
        "ridge": ridge_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        **acceptance_fields(rejection_reasons),
        "log_c_0": float(log_c_0),
        "log_mu_0": float(log_mu_0),
    }
    if verbose:
        print(
            f"solve_time(s) = {solve_time}, "
            f"max_benchmark_error = {max_benchmark_error:.3e}"
        )

    frame_names = (
        "t",
        "k",
        "a",
        "q",
        "c",
        "mu",
        "x",
        "output",
        "current_account",
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth",
        "consumption_foc",
    )
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_names}
    )
    test_df = pd.DataFrame({name: np.asarray(test_eval[name]) for name in frame_names})
    for name in ("k", "a", "q", "c", "x", "output", "current_account"):
        test_df[f"{name}_benchmark"] = np.asarray(benchmark_values[name])
    test_df["k_relative_error"] = np.asarray(k_relative_error)
    test_df["q_relative_error"] = np.asarray(q_relative_error)
    test_df["c_relative_error"] = np.asarray(c_relative_error)
    test_df["a_scaled_error"] = np.asarray(a_scaled_error)

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
        "tolerance_set": asdict(tolerance_set),
        "parameters": {
            "alpha": alpha,
            "gamma": gamma,
            "delta": delta,
            "r": r,
            "eta": eta,
            "r_star": r_star,
        },
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
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
            "residual_args": residual_args,
            "multi_start": None,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_kernel)
