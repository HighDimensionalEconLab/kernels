from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    CG,
    QR,
    BlockEigenPreconditioner,
    Cholesky,
    LMSolveAction,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
    ridge_continuation,
)

from kernels import (
    integrated_kernel_matrices,
    normalize_kernel,
    rkhs_norm_squared,
)
from multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    multicountry_long_run_diagnostics,
    multicountry_output_scale,
    multicountry_parameter_values,
    multicountry_path_values,
    multicountry_reference_marginal_wealth,
    multicountry_steady_state,
    multicountry_tail_diagnostics,
    multicountry_tvc_diagnostics,
)
from multicountry_growth_benchmark import multicountry_growth_benchmark
from multicountry_growth_preconditioner import (
    block_eigen_permutation,
    multicountry_block_eigen_state,
)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
ACCOUNTING_RESIDUAL_TOL = 5e-5
INITIAL_CLEARING_TOL = 1e-10
BENCHMARK_MATCH_TOL = 2e-3
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class ToleranceSettings:
    # Solution acceptance gates. The train gate bounds the max scaled DAE
    # residual on the collocation grid; at a fixed ridge the converged
    # residual floors at the O(ridge) bias, so a run at a deliberately
    # larger ridge (e.g. an iterative linear solver whose Krylov iteration
    # count scales like 1/sqrt(ridge)) must pass a ridge-aware value here
    # rather than the float64 production default.
    train_residual: float = TRAIN_RESIDUAL_TOL
    validation_residual: float = VALIDATION_RESIDUAL_TOL
    accounting: float = ACCOUNTING_RESIDUAL_TOL
    benchmark_match: float = BENCHMARK_MATCH_TOL


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern52"
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 160.0
    train_points: int = 96
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 15
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401
    benchmark_T: float = 200.0
    benchmark_points: int = 241
    # Long-run validity checks are evaluated at t ~ long_run_check_T (flat
    # consumption/capital/debt, capital at the theory steady state, r ~ rho),
    # with flatness measured against t = long_run_check_T - long_run_window.
    long_run_check_T: float = 100.0
    long_run_window: float = 10.0


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 80
    # Stopping leans on gtol alone by default: gtol = 1e-11 sits ABOVE the
    # ridge-floor stationarity ~ ridge * sqrt(q) ~ 2e-12 and BELOW the last
    # intermediate continuation level's ~ 6e-10, so it can only fire at the
    # floor -- no residual_atol needed to rule intermediate levels out. The
    # converged residual floors at O(ridge * ||dual||), which moves with N
    # and with the drawn initial conditions, so a fixed positive atol is
    # draw-fragile (it silently blocks the stopping certificate on solves
    # that pass every quality gate); pass one explicitly only for a fixed
    # production configuration whose floor has been measured.
    residual_atol: float = 0.0
    # "auto" (whitened cholesky) is the default: the whitened normal matrix
    # J~'J~ + ridge*I_head has its spectral floor at the ridge, so the
    # cholesky path keeps float64 accuracy at this ridge -- measured ~2x
    # faster wall-clock than the previously required "qr" at identical
    # solution quality. "qr" remains valid (all paths run whitened). "cg"
    # is the matrix-free scaling path: preconditioned CG on the whitened
    # normal operator with the block-eigen preconditioner (per-country
    # blocks + rate block + free block, assembled analytically and rebuilt
    # by the continuation callback whenever the anneal advances a level).
    linear_solver: str = "auto"
    # Measured (N=2, n_train=96): the block-eigen-preconditioned CG count
    # SATURATES as the ridge anneals to the floor (~941 iterations at ridge
    # 1e-7, ~1516 at 3e-12 -- the preconditioner's diagonal blocks carry the
    # same ridge floor as the operator, so the deep-ridge modes stay
    # clustered); the budget must cover that saturated count or the
    # truncated steps stall the endgame.
    cg_tol: float = 1e-10
    cg_maxiter: int = 2500
    geodesic_acceleration: bool = True
    ridge: float = 3e-12
    # The whitened calibration recipe reads
    # gtol ~ 1e-3 * ridge * sqrt(rkhs_norm_total) ~ 1e-15 at ridge 3e-12
    # (sqrt(q) ~ 0.35); the retained 1e-11 is a looser MEASURED value
    # carried over from the unwhitened noise-floor calibration -- the
    # whitened solve converges with it in ~9 steps with every quality gate
    # intact, and the extra resolution the recipe would demand was verified
    # (unwhitened) to move every quality metric only in the sixth
    # significant digit.
    gtol: float = 1e-11
    # ridge_init=None solves at the fixed ridge above; a value enables ridge
    # continuation annealed from ridge_init down to `ridge` (the floor) by
    # factors of `ridge_decrease` on inner stationarity.
    ridge_init: float | None = 1e-7
    ridge_decrease: float = 0.01
    # Relative stationarity threshold that triggers the next ridge level:
    # looser anneals faster (fewer steps per level), tighter solves each
    # level out more completely before descending.
    ridge_grad_rtol: float = 1e-3
    # Frozen-anneal escape hatch (ridge_continuation's stall_rtol): at
    # larger N the compounding per-level references can freeze the anneal
    # below the gradient noise floor while steps keep being accepted;
    # ~0.99 advances the level on accepted steps with negligible gradient
    # progress. Off by default.
    ridge_stall_rtol: float = 0.0


def make_train_grid(data_set):
    if data_set.train_points_list is not None:
        train_data = jnp.sort(jnp.asarray(data_set.train_points_list))
    elif data_set.sampling == "uniform":
        train_data = jnp.linspace(0.0, data_set.train_T, data_set.train_points)
    elif data_set.sampling == "front_loaded":
        dense_points = data_set.train_points - data_set.tail_points
        if (
            dense_points < 2
            or data_set.tail_points < 1
            or not 0.0 < data_set.dense_T < data_set.train_T
        ):
            raise ValueError(
                "front_loaded sampling requires at least 2 dense points, "
                "1 tail point, and 0 < dense_T < train_T"
            )
        dense_data = jnp.linspace(0.0, data_set.dense_T, dense_points)
        tail_data = jnp.linspace(
            data_set.dense_T,
            data_set.train_T,
            data_set.tail_points + 1,
        )[1:]
        train_data = jnp.concatenate([dense_data, tail_data])
    else:
        raise ValueError("sampling must be 'front_loaded' or 'uniform'")
    if len(train_data) < 3 or not bool(jnp.all(jnp.diff(train_data) > 0.0)):
        raise ValueError("training points must be strictly increasing")
    if float(train_data[0]) != 0.0:
        raise ValueError("training points must start at zero")
    return train_data


def multicountry_growth_kernel(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    sensitivity_time: float | None = None,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    num_countries = parameters.num_countries
    # Identical initial conditions are degenerate (no cross-country
    # variation), so unspecified k_0/a_0 are DRAWN from the seeded
    # initial_set distributions rather than defaulted to symmetric values.
    if k_0 is None or a_0 is None:
        drawn_k_0, drawn_a_0 = draw_initial_conditions(initial_set, parameters)
        if k_0 is None:
            k_0 = drawn_k_0
        if a_0 is None:
            a_0 = drawn_a_0
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive capital stock per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one NFA position per country")
    supported_linear_solvers = ("auto", "qr", "cg")
    if lm_set.linear_solver not in supported_linear_solvers:
        raise ValueError(f"linear_solver must be one of {supported_linear_solvers}")

    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_train_grid(data_set).astype(dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.benchmark_T < max(data_set.test_T, float(train_data[-1])):
        raise ValueError("benchmark_T must cover the train and test horizons")
    if sensitivity_time is not None and not (
        0.0 <= sensitivity_time <= float(train_data[-1])
    ):
        raise ValueError("sensitivity_time must lie within the training horizon")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    benchmark_grid = np.linspace(0.0, data_set.benchmark_T, data_set.benchmark_points)
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=k_0_array.dtype)
    output_scale = multicountry_output_scale(parameters, k_0_array.dtype)
    initial_asset_gap = jnp.sum(populations * a_0_array) / output_scale
    if abs(float(initial_asset_gap)) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")
    q_reference = jnp.full(
        num_countries,
        1.0 / parameters.installation_elasticity,
        dtype=k_0_array.dtype,
    )
    marginal_wealth_reference = multicountry_reference_marginal_wealth(
        parameters, k_0_array.dtype
    )

    metric_data = train_data.astype(jnp.float64)
    K_metric, K_tilde_metric = integrated_kernel_matrices(
        metric_data,
        metric_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K_metric = (K_metric + K_metric.T) / 2.0
    K = K_metric.astype(dtype)
    K_tilde = K_tilde_metric.astype(dtype)

    # The structural parameters and initial conditions ride in the solver's
    # `p` argument so the solved paths are implicitly differentiable with
    # respect to every one of them (nlls-gram's implicit JVP/VJP at the
    # ridge stationary point). All parameter-dependent quantities the paths
    # consume are recomputed from p inside.
    p = {
        "parameters": multicountry_parameter_values(parameters, dtype),
        "k_0": k_0_array,
        "a_0": a_0_array,
    }

    def path_coordinates(theta, K_eval, K_tilde_eval, p):
        parameters_p = p["parameters"]
        populations_p = parameters_p.populations
        k_0_p = p["k_0"]
        # Population demeaning is a no-op at the primal point (the inputs
        # already clear) and projects AD perturbations of a_0 onto the
        # feasible clearing set, so every direction is meaningful.
        a_0_p = p["a_0"] - jnp.sum(populations_p * p["a_0"]) / jnp.sum(populations_p)
        output_scale_p = multicountry_output_scale(parameters_p, k_0_p.dtype)
        marginal_wealth_reference_p = multicountry_reference_marginal_wealth(
            parameters_p, k_0_p.dtype
        )
        alpha_k, alpha_a, alpha_q, alpha_m, alpha_r, initial = theta
        q_offsets = initial[:num_countries]
        marginal_wealth_offsets = initial[num_countries : 2 * num_countries]
        interest_rate_offset = initial[-1]
        log_k = jnp.log(k_0_p) + K_tilde_eval @ alpha_k.T
        a = a_0_p + output_scale_p * (K_tilde_eval @ alpha_a.T)
        log_q = (
            -jnp.log(parameters_p.installation_elasticity)
            + q_offsets
            + K_tilde_eval @ alpha_q.T
        )
        log_marginal_wealth = (
            jnp.log(marginal_wealth_reference_p)
            + marginal_wealth_offsets
            + K_tilde_eval @ alpha_m.T
        )
        dlog_k_dt = K_eval @ alpha_k.T
        da_dt = output_scale_p * (K_eval @ alpha_a.T)
        dlog_q_dt = K_eval @ alpha_q.T
        dlog_marginal_wealth_dt = K_eval @ alpha_m.T
        interest_rate = jnp.exp(
            jnp.log(parameters_p.rho) + interest_rate_offset + K_tilde_eval @ alpha_r
        )
        return {
            "k": jnp.exp(log_k),
            "a": a,
            "tobin_q": jnp.exp(log_q),
            "marginal_wealth": jnp.exp(log_marginal_wealth),
            "interest_rate": interest_rate,
            "dlog_k_dt": dlog_k_dt,
            "da_dt": da_dt,
            "dlog_q_dt": dlog_q_dt,
            "dlog_marginal_wealth_dt": dlog_marginal_wealth_dt,
        }

    def evaluate_coordinates(
        theta,
        K_eval,
        K_tilde_eval,
        p,
    ):
        coordinates = path_coordinates(theta, K_eval, K_tilde_eval, p)
        return multicountry_path_values(**coordinates, parameters=p["parameters"])

    residual_count = (4 * num_countries + 1) * n_train + 1
    residual_scale = 1.0 / jnp.sqrt(float(residual_count))

    # The aggregate rows are population-weighted SUMS, so their natural size
    # grows with N; per-capita normalization keeps every scaled residual
    # entry N-free (without it the 1/sqrt(count) scaling leaves the
    # aggregate rows ~sqrt(N) large and residual_atol becomes unreachable at
    # large N).
    total_population = jnp.sum(populations)

    # args carries only the cg preconditioner state (None on the dense
    # paths); the model equations never read it. p carries the structural
    # parameters and initial conditions for implicit differentiation.
    def residual_fn(theta, _, p):
        values = evaluate_coordinates(theta, K, K_tilde, p)
        total_population_p = jnp.sum(p["parameters"].populations)
        # The level-clearing equation at t=0 is fixed by a_0 and has a zero
        # Jacobian. G(0) = 0 replaces that row, while G_dot(0) = 0 supplies the
        # index-consistency equation that identifies the initial algebraic rate.
        global_constraints = (
            jnp.concatenate(
                [
                    values["resource"][:1],
                    values["resource_drift"][:1],
                    values["asset_clearing"][1:],
                ]
            )
            / total_population_p
        )
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
                global_constraints,
            ]
        )

    num_dynamic_functions = 4 * num_countries
    num_rkhs_functions = num_dynamic_functions + 1
    num_initial_values = 2 * num_countries + 1
    # Every linear solver works on the whitened subproblem; the metric takes
    # the upper Cholesky factor F with K = F'F and the trailing initial
    # values stay in the unpenalized free block, untransformed.
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(K_metric, upper=True).astype(dtype),
        repeats=num_rkhs_functions,
    )
    theta_0 = (
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(num_initial_values, dtype=dtype),
    )
    # CLI string -> typed solver config ("auto" = the whitened cholesky
    # default; "cg" = matrix-free preconditioned CG on the whitened normal
    # operator, the block-eigen preconditioner state riding in args).
    if lm_set.linear_solver == "qr":
        linear_solver = QR()
        ad_solver = None
    elif lm_set.linear_solver == "cg":
        linear_solver = CG(
            BlockEigenPreconditioner(),
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
        # The implicit tangent/cotangent CG must be preconditioned
        # explicitly (ad_solver=None would run it unpreconditioned) and run
        # to tolerance; it reads the block-eigen state from result.args at
        # zero damping.
        ad_solver = CG(
            BlockEigenPreconditioner(),
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
    else:
        linear_solver = Cholesky()
        ad_solver = None
    if lm_set.linear_solver == "cg":
        permutation = jnp.asarray(block_eigen_permutation(num_countries, n_train))

        def build_preconditioner_state(theta_value):
            return multicountry_block_eigen_state(
                theta_value,
                K,
                K_tilde,
                metric.F,
                permutation,
                parameters,
                k_0_array,
                a_0_array,
                q_reference,
                marginal_wealth_reference,
                output_scale,
                residual_scale,
            )

        args = {"preconditioner": build_preconditioner_state(theta_0)}
    else:
        args = None
    if lm_set.ridge_init is None:
        initial_ridge = lm_set.ridge
        continuation_options = {}
    else:
        initial_ridge = lm_set.ridge_init
        continuation_callback, user_state0 = ridge_continuation(
            ridge_floor=lm_set.ridge,
            decrease=lm_set.ridge_decrease,
            grad_rtol=lm_set.ridge_grad_rtol,
            stall_rtol=lm_set.ridge_stall_rtol,
            dtype=dtype,
        )
        if lm_set.linear_solver == "cg":
            # Rebuild the preconditioner from the live iterate exactly when
            # the anneal advances a level: that step's convergence test is
            # already suppressed by the ridge change, and on every other
            # step the returned state is value-identical (no suppression).
            def callback(ctx):
                action = continuation_callback(ctx)
                advanced = action.lm_state.ridge < ctx.lm_state.ridge
                fresh = build_preconditioner_state(ctx.x)
                state = jax.tree_util.tree_map(
                    lambda old, new: jnp.where(advanced, new, old),
                    ctx.args["preconditioner"],
                    fresh,
                )
                return LMSolveAction(
                    lm_state=action.lm_state,
                    user_state=action.user_state,
                    args={"preconditioner": state},
                )
        else:
            callback = continuation_callback
        continuation_options = {"callback": callback, "user_state": user_state0}

    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=initial_ridge,
        linear_solver=linear_solver,
        ad_solver=ad_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
    )
    solve_options = dict(
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )

    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        args,
        p=p,
        **solve_options,
        **continuation_options,
    )
    jax.block_until_ready(result)
    solve_time = time.perf_counter() - solve_start
    theta = result.x

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        return evaluate_coordinates(theta, K_eval, K_tilde_eval, p)

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    dynamic_residual_names = (
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth_euler",
    )
    # Aggregate-row diagnostics are per capita, matching the objective's
    # normalization: the raw population-weighted sums grow with N, so the
    # per-capita values are the scale-free quantities the N-independent
    # gates bound.
    train_global_constraints = (
        jnp.concatenate(
            [
                train_eval["resource"][:1],
                train_eval["resource_drift"][:1],
                train_eval["asset_clearing"][1:],
            ]
        )
        / total_population
    )
    max_train_residual = max(
        max(
            float(jnp.max(jnp.abs(train_eval[name]))) for name in dynamic_residual_names
        ),
        float(jnp.max(jnp.abs(train_global_constraints))),
    )
    max_validation_residual = max(
        max(
            float(jnp.max(jnp.abs(validation_eval[name])))
            for name in dynamic_residual_names
        ),
        float(jnp.max(jnp.abs(validation_eval["resource"] / total_population))),
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"] / total_population))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"]))
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "x", "marginal_wealth")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account", "interest_rate")
        )
    )
    rejection_reasons = []
    if int(result.status) != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    if max_validation_residual > tolerance_set.validation_residual:
        rejection_reasons.append("validation_residual")
    if max_asset_clearing_residual > tolerance_set.accounting:
        rejection_reasons.append("asset_clearing")
    if max_rate_consistency_residual > tolerance_set.accounting:
        rejection_reasons.append("rate_consistency")
    solution_accepted = not rejection_reasons

    benchmark_available = num_countries == 2
    benchmark_values = None
    benchmark_diagnostics = None
    if benchmark_available:
        benchmark, benchmark_diagnostics = multicountry_growth_benchmark(
            k_0,
            (a_0[0],),
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
        interest_rate_scaled_error = (
            jnp.abs(
                test_eval["interest_rate"]
                - jnp.asarray(benchmark_values["interest_rate"])
            )
            / parameters.rho
        )
        max_benchmark_error = max(
            float(jnp.max(k_relative_error)),
            float(jnp.max(q_relative_error)),
            float(jnp.max(c_relative_error)),
            float(jnp.max(a_scaled_error)),
            float(jnp.max(interest_rate_scaled_error)),
        )
        benchmark_match = max_benchmark_error <= tolerance_set.benchmark_match
    else:
        k_relative_error = jnp.asarray(jnp.nan)
        q_relative_error = jnp.asarray(jnp.nan)
        c_relative_error = jnp.asarray(jnp.nan)
        a_scaled_error = jnp.asarray(jnp.nan)
        interest_rate_scaled_error = jnp.asarray(jnp.nan)
        max_benchmark_error = float("nan")
        benchmark_match = False

    resolved_steady = multicountry_steady_state(
        validation_eval["marginal_wealth"][-1], parameters
    )
    tail = multicountry_tail_diagnostics(validation_eval, parameters)
    tvc = multicountry_tvc_diagnostics(validation_data, validation_eval, parameters)
    long_run = multicountry_long_run_diagnostics(
        validation_data,
        validation_eval,
        a_0_array,
        parameters,
        check_T=data_set.long_run_check_T,
        window=data_set.long_run_window,
    )
    relative_weights = test_eval["relative_welfare_weights"][0]
    log_relative_weights = jnp.log(relative_weights)
    coefficient_groups = {
        "k": theta[0],
        "a": theta[1],
        "q": theta[2],
        "m": theta[3],
    }
    rkhs_norms = {
        f"{name}_{country + 1}": rkhs_norm_squared(coefficients[country], K_metric)
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_rate = rkhs_norm_squared(theta[4], K_metric)
    rkhs_norm_total = sum(rkhs_norms.values()) + rkhs_norm_rate

    sensitivity_df = None
    if sensitivity_time is not None:
        K_sens, K_tilde_sens = integrated_kernel_matrices(
            jnp.asarray([sensitivity_time], dtype=dtype),
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )

        # One reverse-mode pass through the solve yields the comparative
        # static of the global rate with respect to every structural
        # parameter and initial condition together: the implicit dependence
        # of the solved coefficients on p plus the direct dependence of the
        # rate evaluation. The a_0 gradient is the projected one (demeaning
        # inside path_coordinates), so it measures feasible redistributions.
        def rate_at_sensitivity_time(p_value):
            sensitivity_result = solver.solve(
                theta_0,
                args,
                p=p_value,
                **solve_options,
                **continuation_options,
            )
            coordinates = path_coordinates(
                sensitivity_result.x, K_sens, K_tilde_sens, p_value
            )
            return coordinates["interest_rate"][0]

        rate_value, rate_grads = jax.value_and_grad(rate_at_sensitivity_time)(p)
        jax.block_until_ready(rate_grads)
        gradient_parameters = rate_grads["parameters"]
        scalar_sensitivities = {
            "alpha": float(gradient_parameters.alpha),
            "gamma": float(gradient_parameters.gamma),
            "delta": float(gradient_parameters.delta),
            "rho": float(gradient_parameters.rho),
            "installation_elasticity": float(
                gradient_parameters.installation_elasticity
            ),
        }
        country_sensitivities = {
            "productivity": np.asarray(gradient_parameters.productivities),
            "population": np.asarray(gradient_parameters.populations),
            "k_0": np.asarray(rate_grads["k_0"]),
            "a_0": np.asarray(rate_grads["a_0"]),
        }
        sensitivity_df = pd.DataFrame(
            [
                {"name": name, "country": 0, "value": value}
                for name, value in scalar_sensitivities.items()
            ]
            + [
                {"name": name, "country": country + 1, "value": float(values[country])}
                for name, values in country_sensitivities.items()
                for country in range(num_countries)
            ]
        )

    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "rho": parameters.rho,
        "installation_elasticity": parameters.installation_elasticity,
        "num_countries": num_countries,
        "residual_dtype": str(jnp.dtype(dtype)),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "max_train_residual": max_train_residual,
        "initial_resource_residual": float(
            train_eval["resource"][0] / total_population
        ),
        "initial_resource_drift_residual": float(
            train_eval["resource_drift"][0] / total_population
        ),
        "max_validation_residual": max_validation_residual,
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_rate_consistency_residual": max_rate_consistency_residual,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_available": benchmark_available,
        "benchmark_match": benchmark_match,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "interest_rate_scaled_error_max": float(jnp.max(interest_rate_scaled_error)),
        "terminal_residual_count": 0,
        "dae_residual_count": residual_count,
        "unknown_count": (4 * num_countries + 1) * n_train + 2 * num_countries + 1,
        "dynamic_rkhs_function_count": num_dynamic_functions,
        "algebraic_rkhs_function_count": 1,
        "free_initial_value_count": num_initial_values,
        "metric_selection_dimension": 2 * num_countries,
        "selection": "minimum_rkhs_norm_ridge",
        "initialization": "zero_coordinate",
        "global_closure": "explicit_assets_and_algebraic_clearing",
        "long_run_check_T": data_set.long_run_check_T,
        "long_run_window": data_set.long_run_window,
        "long_run_c_flatness": float(long_run["c_flatness"]),
        "long_run_k_flatness": float(long_run["k_flatness"]),
        "long_run_a_flatness": float(long_run["a_flatness"]),
        "long_run_k_theory_gap": float(long_run["k_theory_gap"]),
        "long_run_mpk_equalization_gap": float(long_run["mpk_equalization_gap"]),
        "long_run_interest_rate_gap": float(long_run["interest_rate_gap"]),
        "long_run_consumption_ratio_drift": float(long_run["consumption_ratio_drift"]),
        "long_run_wealth_proportionality_gap": float(
            long_run["wealth_proportionality_gap"]
        ),
        "long_run_wealth_rank_violation": float(long_run["wealth_rank_violation"]),
        "long_run_c_rank_matches_wealth": long_run["c_rank_matches_wealth"],
        "long_run_c_rank_matches_a_0": long_run["c_rank_matches_a_0"],
        "long_run_terminal_a_rank_matches_a_0": long_run["terminal_a_rank_matches_a_0"],
        "ex_post_k_log_gap": float(jnp.max(tail["k_log_gap"])),
        "ex_post_q_log_gap": float(jnp.max(tail["q_log_gap"])),
        "ex_post_asset_scaled_gap": float(jnp.max(tail["asset_scaled_gap"])),
        "ex_post_interest_rate_gap": float(tail["interest_rate_gap"]),
        "ex_post_discounted_assets": float(jnp.max(tvc["discounted_assets"])),
        "ex_post_asset_tvc": float(jnp.max(tvc["asset_tvc"])),
        "ex_post_capital_tvc": float(jnp.max(tvc["capital_tvc"])),
        "rkhs_norm_total": float(rkhs_norm_total),
        "rkhs_norm_interest_rate": float(rkhs_norm_rate),
        "solve_time": solve_time,
        "wall_time": solve_time,
        "solver": "nlls-gram",
        "linear_solver": lm_set.linear_solver,
        "preconditioner": ("block_eigen" if lm_set.linear_solver == "cg" else "none"),
        "cg_tol": lm_set.cg_tol,
        "cg_maxiter": lm_set.cg_maxiter,
        "residual_atol": lm_set.residual_atol,
        "train_residual_tol": tolerance_set.train_residual,
        "geodesic_acceleration": lm_set.geodesic_acceleration,
        "initial_seed": initial_set.seed,
        "k_0_log_sigma": initial_set.k_0_log_sigma,
        "a_0_sigma": initial_set.a_0_sigma,
        "ridge": lm_set.ridge,
        "gtol": lm_set.gtol,
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted" if solution_accepted else ",".join(rejection_reasons)
        ),
    }
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"population_{suffix}": parameters.populations[country],
                f"productivity_{suffix}": parameters.productivities[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
                f"a_0_{suffix}_over_output_ss": a_0[country] / float(output_scale),
                f"q_0_{suffix}": float(test_eval["q"][0, country]),
                f"m_0_{suffix}": float(test_eval["marginal_wealth"][0, country]),
                f"relative_weight_{suffix}": float(relative_weights[country]),
                f"log_relative_weight_{suffix}": float(log_relative_weights[country]),
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"long_run_a_{suffix}": float(resolved_steady["a"][country]),
                f"long_run_c_{suffix}": float(resolved_steady["c"][country]),
                f"initial_wealth_{suffix}": float(long_run["initial_wealth"][country]),
                **{
                    f"rkhs_norm_{name}_{suffix}": float(rkhs_norms[f"{name}_{suffix}"])
                    for name in coefficient_groups
                },
            }
        )
    if sensitivity_time is not None:
        results["sensitivity_time"] = sensitivity_time
        results["interest_rate_at_sensitivity_time"] = float(rate_value)
        results.update(
            {f"d_rate_d_{name}": value for name, value in scalar_sensitivities.items()}
        )
        for country in range(num_countries):
            suffix = country + 1
            results.update(
                {
                    f"d_rate_d_{name}_{suffix}": float(values[country])
                    for name, values in country_sensitivities.items()
                }
            )
    if benchmark_available:
        results.update(
            {
                "log_relative_weight": float(log_relative_weights[1]),
                "relative_weight": float(relative_weights[1]),
                "benchmark_log_relative_weight": benchmark_diagnostics[
                    "log_relative_weight"
                ],
                "benchmark_boundary_residual": benchmark_diagnostics[
                    "boundary_residual_max"
                ],
                "stable_roots": benchmark_diagnostics["stable_roots"],
                "unstable_roots": benchmark_diagnostics["unstable_roots"],
                "center_roots": benchmark_diagnostics["center_roots"],
                "minimum_rate_denominator": benchmark_diagnostics[
                    "minimum_rate_denominator"
                ],
            }
        )

    def frame(values, points):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "resource_residual": np.asarray(values["resource"] / total_population),
            "resource_drift_residual": np.asarray(
                values["resource_drift"] / total_population
            ),
            "asset_clearing_residual": np.asarray(
                values["asset_clearing"] / total_population
            ),
            "interest_rate_country_gap": np.asarray(
                jnp.max(
                    jnp.abs(values["interest_rate_country_gap"]),
                    axis=1,
                )
            ),
        }
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "x",
                "output",
                "resources",
                "current_account",
                "capital",
                "assets",
                "tobin_q",
                "marginal_wealth",
                "marginal_wealth_euler",
                "relative_welfare_weights",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        return pd.DataFrame(data)

    train_df = frame(train_eval, train_data)
    validation_df = frame(validation_eval, validation_data)
    test_df = frame(test_eval, test_data)
    if benchmark_available:
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "x",
                "output",
                "current_account",
            ):
                test_df[f"{name}_{suffix}_benchmark"] = benchmark_values[name][
                    :, country
                ]
        test_df["interest_rate_benchmark"] = benchmark_values["interest_rate"]

    hyperparameters = {
        "parameters": asdict(parameters),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
    }

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        return evaluate_coordinates(theta, K_eval, K_tilde_eval, p)

    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "sensitivity_df": sensitivity_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_growth_kernel)
