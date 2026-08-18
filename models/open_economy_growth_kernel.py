from __future__ import annotations

from dataclasses import asdict, dataclass

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
    integrated_kernel_matrices,
    integrated_metric_factorization,
    normalize_kernel,
    rkhs_norm_squared,
)
from models.open_economy_growth import (
    ModelParameters,
    investment_expenditure,
    path_residuals,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.utilities import (
    STATUS_TO_REASON,
    SolveSpec,
    Stopwatch,
    acceptance_fields,
    make_sampled_train_grid,
)


# Numerical acceptance and ex-post quality thresholds, CLI-reachable so a
# float32 caller can pass its own bars (the float32 replication driver does).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    # Validation and benchmark thresholds are HPO/quality diagnostics; they do
    # not redefine convergence of a finite, feasible numerical solution.
    validation_residual: float = 1e-2
    benchmark_match: float = 1e-3


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern12"
    ell: float = 20.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 160.0
    train_points: int = 192
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 15
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401
    benchmark_T: float = 200.0
    benchmark_points: int = 241


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 100
    # Conjunctive stopping in the whitened variable y = F_bar theta the
    # solver works in: gtol bounds the whitened stationarity
    # ||J~'r + ridge [y_m; 0]||, residual_atol additionally demands the
    # scaled residual norm solved (the converged residual floors at
    # O(ridge * ||dual||), so residual_atol sits above it). The ridge keeps
    # the minimizer O(ridge) from the pure RKHS-seminorm interpolant with
    # the free offsets unpenalized (no epsilon shift anywhere).
    # The whitened calibration recipe reads
    # gtol ~ 1e-3 * ridge * sqrt(rkhs_norm_total) ~ 1e-14 at ridge 3e-12
    # (sqrt(q) ~ 3.7); the retained 5e-8 is a much looser MEASURED value
    # carried over from the unwhitened noise-floor calibration -- the
    # whitened runs converge with it in 8-38 steps across the parameter
    # variants with every quality gate intact, and residual_atol (sitting
    # BETWEEN the ridge-floor residual ~1.3-2.9e-7 and the last
    # intermediate continuation level's ~9e-7) is what pins the solve to
    # the floor.
    residual_atol: float = 5e-7
    ridge: float = 3e-12
    gtol: float = 5e-8
    # ridge_init=None solves at the fixed ridge above; a value enables ridge
    # continuation annealed from ridge_init down to `ridge` (the floor) by
    # factors of `ridge_decrease` on inner stationarity. The metric's
    # factorization never involves the ridge, so the anneal composes
    # unchanged.
    ridge_init: float | None = 1e-7
    ridge_decrease: float = 0.01
    # Relative stationarity threshold that triggers the next ridge level:
    # looser anneals faster (fewer steps per level), tighter solves each
    # level out more completely before descending.
    ridge_grad_rtol: float = 1e-2
    # "auto" (whitened cholesky) is the default: the whitened normal matrix
    # J~'J~ + ridge*I_head has its spectral floor at the ridge, so the DAE
    # Jacobian's scale spread no longer pushes the cholesky path past
    # float64 at this ridge -- measured 1.8-3x faster wall-clock than the
    # previously required "qr" at identical solution quality. "qr" and
    # "cholesky" remain valid (all paths run whitened).
    linear_solver: str = "auto"


def open_economy_growth_kernel(
    k_0: float = 1.0,
    a_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    if k_0 <= 0.0:
        raise ValueError("k_0 must be positive")
    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_sampled_train_grid(data_set, dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.benchmark_T < max(data_set.test_T, float(train_data[-1])):
        raise ValueError("benchmark_T must cover the train and test horizons")

    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    benchmark_grid = np.linspace(0.0, data_set.benchmark_T, data_set.benchmark_points)
    n_train = len(train_data)
    steady = steady_state(parameters)
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

    # theta = (alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q,
    #          (c_offset, mu_offset, q_offset)), matching the metric blocks.
    def path_values(theta, K_eval, K_tilde_eval):
        alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, initial = theta
        c_offset, mu_offset, q_offset = initial
        log_mu = (
            -parameters.gamma * jnp.log(consumption_scale)
            + mu_offset
            + K_tilde_eval @ alpha_mu
        )
        log_c = jnp.log(consumption_scale) + c_offset + K_tilde_eval @ alpha_c
        log_k = jnp.log(k_0) + K_tilde_eval @ alpha_k
        a = a_0 + output_scale * (K_tilde_eval @ alpha_a)
        log_q = jnp.log(steady["q"]) + q_offset + K_tilde_eval @ alpha_q
        mu = jnp.exp(log_mu)
        c = jnp.exp(log_c)
        k = jnp.exp(log_k)
        q = jnp.exp(log_q)
        dlog_mu_dt = K_eval @ alpha_mu
        dlog_k_dt = K_eval @ alpha_k
        da_dt = output_scale * (K_eval @ alpha_a)
        dlog_q_dt = K_eval @ alpha_q
        return (
            k,
            a,
            q,
            c,
            mu,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
        )

    residual_scale = 1.0 / jnp.sqrt(5.0 * n_train)

    def residual_fn(theta):
        values = path_values(theta, K, K_tilde)
        (
            k,
            a,
            q,
            c,
            mu,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
        ) = values
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
            parameters,
        )
        return residual_scale * jnp.concatenate(
            [
                residuals["capital"],
                residuals["assets"],
                residuals["tobin_q"],
                residuals["marginal_wealth"],
                residuals["consumption_foc"],
            ]
        )

    # The zero coordinate contains no saddle-path or present-value information:
    # it is flat in the derivative RKHS at the model's scale normalizations.
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(3, dtype=dtype),
    )

    if lm_set.ridge_init is None:
        initial_ridge = lm_set.ridge
        continuation_options = {}
    else:
        initial_ridge = lm_set.ridge_init
        callback = AnnealRidge(
            ridge_floor=lm_set.ridge,
            decrease=lm_set.ridge_decrease,
            grad_rtol=lm_set.ridge_grad_rtol,
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
            lm_set.linear_solver
        ],
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
    result = solver.solve(theta_0, **solve_options, **continuation_options)
    solve_time = timer.elapsed(result)
    theta = result.x

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        (
            k,
            a,
            q,
            c,
            mu,
            dlog_k_dt,
            da_dt,
            dlog_q_dt,
            dlog_mu_dt,
        ) = path_values(theta, K_eval, K_tilde_eval)
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
            parameters,
        )
        x = investment_expenditure(k, q, parameters)
        output = production(k, z, parameters)
        current_account = da_dt
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": q,
            "c": c,
            "mu": mu,
            "x": x,
            "output": output,
            "current_account": current_account,
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
        parameters,
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

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(
        k_0,
        a_0,
        benchmark_grid,
        parameters=parameters,
    )
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
        a_0,
        parameters,
    )
    tvc = tvc_diagnostics(
        train_eval["k"][-1],
        train_eval["a"][-1],
        train_eval["q"][-1],
        train_eval["c"][-1],
        train_eval["t"][-1],
        parameters,
    )

    rejection_reasons = []
    if int(result.status) != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    validation_pass = max_validation_residual <= tolerance_set.validation_residual
    benchmark_match = max_benchmark_error <= tolerance_set.benchmark_match

    alpha_mu, alpha_c, alpha_k, alpha_a, alpha_q, initial = theta
    c_offset, mu_offset, q_offset = initial
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
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        k, a, q, c, *_ = path_values(theta, K_eval, K_tilde_eval)
        return k, a, q, c

    results = {
        **asdict(parameters),
        "r_star_resolved": parameters.interest_rate,
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
        "q_0": float(steady["q"] * jnp.exp(q_offset)),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "validation_pass": validation_pass,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_match": benchmark_match,
        "terminal_residual_count": 0,
        "differential_residual_count": 4 * n_train,
        "algebraic_residual_count": n_train,
        "dae_residual_count": 5 * n_train,
        "unknown_count": 5 * n_train + 3,
        "dynamic_rkhs_function_count": 4,
        "algebraic_rkhs_function_count": 1,
        "free_initial_value_count": 3,
        "metric_selection_dimension": 3,
        "selection": "minimum_rkhs_norm_ridge",
        "initialization": "zero_coordinate",
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
        "benchmark_a_0_gap": abs(float(benchmark_diagnostics["a_0_gap"])),
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
        "wall_time": solve_time,
        "solver": "nlls-gram",
        "linear_solver": lm_set.linear_solver,
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        **acceptance_fields(rejection_reasons),
        "c_offset": float(c_offset),
        "mu_offset": float(mu_offset),
    }

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
        "parameters": asdict(parameters),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
    }
    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
        "solve": SolveSpec(
            solver=solver,
            theta_0=theta_0,
            p=None,
            residual_args=None,
            multi_start=None,
            options=solve_options,
        ),
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_kernel)
