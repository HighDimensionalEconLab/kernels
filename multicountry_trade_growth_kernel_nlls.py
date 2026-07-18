from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import scipy.interpolate
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMStatus,
    PreconditionerFactory,
    RecycleConfig,
    WhitenedPreconditioner,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_with_compute_dtype,
    repeated_blockdiag_metric,
)

from kernels import integrated_kernel_matrices, normalize_kernel
from multicountry_growth_kernel_nlls import make_train_grid
from multicountry_trade_growth import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    consumption_basket_log_price,
    multicountry_trade_dae_residual,
    multicountry_trade_max_dae_residual,
    multicountry_trade_path_values,
    multicountry_trade_pointwise_residual_norms,
    multicountry_trade_tvc_diagnostics,
    static_trade_equilibrium,
)
from trade_growth_preconditioners import (
    build_trade_growth_time_block_preconditioner,
    residual_time_permutation,
    trade_growth_preconditioner_apply,
)
from trade_growth_whitened import build_schur_preconditioner

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
ACCOUNTING_RESIDUAL_TOL = 5e-5
INITIAL_CLEARING_TOL = 1e-10
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern52"
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 80.0
    train_points: int = 24
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 30.0
    tail_points: int = 6
    test_T: float = 40.0
    test_points: int = 81
    validation_points: int = 101
    quality_T: float = 100.0
    static_evaluation: str = "pchip"
    rank_audit: bool = False


@dataclass(frozen=True)
class NLLSGramSettings:
    init_damping: float = 1e-7
    damping_decrease: float = 0.2
    damping_increase: float = 4.0
    max_steps: int = 80
    residual_atol: float = 2e-8
    linear_solver: str = "cholesky"
    metric_eps: float = 1e-7
    # Matrix-free CG path (linear_solver="cg"): time-block preconditioner +
    # Krylov recycling + a forcing-style inner-budget schedule (loose early,
    # tightening near convergence). Ignored on the cholesky/qr paths.
    recycle_rank: int = 24
    recycle_window: int | None = None
    cg_maxiter_min: int = 60
    cg_maxiter_max: int = 250
    cg_tol_loose: float = 3e-2
    cg_tol_tight: float = 1e-7
    cg_atol: float = 0.0
    cg_geodesic: bool = False
    cg_min_damping: float = 0.0
    implicit_maxiter: int = 400
    implicit_tol: float = 1e-8
    # Two-phase endgame (linear_solver="cg" only): the phase-1 CG solve floors at
    # the dual cond^2 limit (~1e-6 residual) with the underdetermined selection
    # (asset_clearing) stuck ~1e-3. A whitened-LSMR phase with a parameter-space
    # Schur preconditioner -- built once at the plateau iterate with pinned
    # damping -- restores the selection. Off by default; cholesky/qr and the
    # single-phase cg path are unchanged.
    two_phase_endgame: bool = False
    phase2_damping: float = 1e-6
    phase2_max_steps: int = 3
    phase2_mode: str = "dense"
    phase2_maxiter: int = 2000
    phase2_tol: float = 1e-13


def multicountry_trade_growth_kernel_nlls(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: TradeGrowthParameters = TradeGrowthParameters(),
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    num_countries = parameters.num_countries
    if k_0 is None:
        k_0 = (1.0,) * num_countries
    if a_0 is None:
        a_0 = (0.0,) * num_countries
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive value per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one value per country")
    if lm_set.metric_eps <= 0.0:
        raise ValueError("metric_eps must be positive")
    if lm_set.linear_solver not in ("cholesky", "qr", "cg"):
        raise ValueError("linear_solver must be 'cholesky', 'qr', or 'cg'")
    if lm_set.phase2_mode not in ("dense", "blockfunc"):
        raise ValueError("phase2_mode must be 'dense' or 'blockfunc'")

    train_data = make_train_grid(data_set).astype(dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.quality_T <= 0.0:
        raise ValueError("quality_T must be positive")
    if data_set.static_evaluation not in ("pchip", "resolve"):
        raise ValueError("static_evaluation must be 'pchip' or 'resolve'")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_data)
    kernel = normalize_kernel(kernel_set.kernel)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=k_0_array.dtype)
    populations = jnp.asarray(parameters.populations, dtype=k_0_array.dtype)
    initial_asset_gap = jnp.sum(populations * a_0_array)
    if abs(float(initial_asset_gap)) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")

    base_q = jnp.full(
        num_countries,
        1.0 / parameters.installation_elasticity,
        dtype=k_0_array.dtype,
    )
    base_m = jnp.ones(num_countries, dtype=k_0_array.dtype)
    base_static, _ = static_trade_equilibrium(
        k_0_array,
        base_q,
        base_m,
        parameters,
    )
    basket_price = jnp.exp(
        consumption_basket_log_price(base_static["log_static"], parameters)
    )
    price_scale = 1.0 / basket_price
    q_reference = price_scale * base_q
    marginal_wealth_reference = base_m / price_scale
    static_reference, static_reference_diagnostics = static_trade_equilibrium(
        k_0_array,
        q_reference,
        marginal_wealth_reference,
        parameters,
        log_initial=base_static["log_static"] + jnp.log(price_scale),
    )
    output_scale = static_reference["world_gdp"] / jnp.sum(populations)
    eps = lm_set.metric_eps

    metric_data = train_data.astype(jnp.float64)
    K_metric, K_tilde_metric = integrated_kernel_matrices(
        metric_data,
        metric_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K_metric = 0.5 * (K_metric + K_metric.T)
    K = K_metric.astype(dtype)
    K_tilde = K_tilde_metric.astype(dtype)
    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K_metric + eps * jnp.eye(n_train, dtype=jnp.float64))
    )

    static_size = n_train * len(STATIC_BLOCK_NAMES) * num_countries
    algebraic_size = n_train + 2 * num_countries + static_size

    def split_algebraic(algebraic):
        rate_coordinates = algebraic[:n_train]
        q_offsets = algebraic[n_train : n_train + num_countries]
        m_start = n_train + num_countries
        marginal_wealth_offsets = algebraic[m_start : m_start + num_countries]
        log_static = algebraic[m_start + num_countries :].reshape(
            n_train, len(STATIC_BLOCK_NAMES), num_countries
        )
        return rate_coordinates, q_offsets, marginal_wealth_offsets, log_static

    def path_coordinates(theta, K_eval, K_tilde_eval, rate_coordinates=None):
        alpha_k, alpha_a, alpha_q, alpha_m, algebraic = theta
        _, q_offsets, marginal_wealth_offsets, _ = split_algebraic(algebraic)
        log_k = jnp.log(k_0_array) + K_tilde_eval @ alpha_k.T
        a = a_0_array + output_scale * (K_tilde_eval @ alpha_a.T)
        log_q = jnp.log(q_reference) + q_offsets + K_tilde_eval @ alpha_q.T
        log_m = (
            jnp.log(marginal_wealth_reference)
            + marginal_wealth_offsets
            + K_tilde_eval @ alpha_m.T
        )
        dlog_k_dt = K_eval @ alpha_k.T
        da_dt = output_scale * (K_eval @ alpha_a.T)
        dlog_q_dt = K_eval @ alpha_q.T
        dlog_m_dt = K_eval @ alpha_m.T
        if rate_coordinates is None:
            interest_rate = parameters.rho - jnp.mean(dlog_m_dt, axis=1)
        else:
            interest_rate = parameters.rho * (1.0 + rate_coordinates)
        return {
            "k": jnp.exp(log_k),
            "a": a,
            "tobin_q": jnp.exp(log_q),
            "marginal_wealth": jnp.exp(log_m),
            "interest_rate": interest_rate,
            "dlog_k_dt": dlog_k_dt,
            "da_dt": da_dt,
            "dlog_q_dt": dlog_q_dt,
            "dlog_marginal_wealth_dt": dlog_m_dt,
        }

    def evaluate_coordinates(theta, K_eval, K_tilde_eval, rate_coordinates, log_static):
        return multicountry_trade_path_values(
            **path_coordinates(theta, K_eval, K_tilde_eval, rate_coordinates),
            log_static=log_static,
            parameters=parameters,
        )

    residual_count = (12 * num_countries + 1) * n_train

    def residual_fn(theta):
        rate_coordinates, _, _, log_static = split_algebraic(theta[-1])
        values = evaluate_coordinates(
            theta,
            K,
            K_tilde,
            rate_coordinates,
            log_static,
        )
        return multicountry_trade_dae_residual(values, log_static, parameters)

    num_dynamic_functions = 4 * num_countries
    metric = metric_with_compute_dtype(
        repeated_blockdiag_metric(
            alpha_metric,
            n_train,
            num_dynamic_functions,
            additional=(
                metric_from_diagonal(jnp.full(algebraic_size, eps, dtype=jnp.float64)),
                algebraic_size,
            ),
        ),
        jnp.float64,
    )
    initial_algebraic = jnp.concatenate(
        [
            jnp.zeros(n_train + 2 * num_countries, dtype=dtype),
            jnp.broadcast_to(
                static_reference["log_static"],
                (n_train, len(STATIC_BLOCK_NAMES), num_countries),
            ).reshape(-1),
        ]
    )
    theta_0 = (
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        initial_algebraic,
    )
    if lm_set.linear_solver == "cg":
        reference_state = {
            "k_0_array": k_0_array,
            "a_0_array": a_0_array,
            "q_reference": q_reference,
            "marginal_wealth_reference": marginal_wealth_reference,
            "output_scale": output_scale,
            "log_static_reference": static_reference["log_static"],
        }
        time_permutation, native_permutation = residual_time_permutation(
            num_countries, n_train
        )

        # theta-adaptive preconditioner: rebuilt from the live iterate each step
        # (a preconditioner frozen at theta_0 breaks the inner CG down once the
        # operator drifts). Model-structured build from the pytree iterate.
        def prepare_preconditioner(x, args, p):
            return build_trade_growth_time_block_preconditioner(
                x,
                K,
                K_tilde,
                reference_state,
                parameters,
                eps,
                time_permutation,
                native_permutation,
            )

        factory = PreconditionerFactory(
            prepare_preconditioner, trade_growth_preconditioner_apply
        )
        recycle = (
            RecycleConfig(rank=lm_set.recycle_rank, window=lm_set.recycle_window)
            if lm_set.recycle_rank > 0
            else None
        )
        recycle_on = recycle is not None
        target_loss = lm_set.residual_atol**2

        def cg_callback(context):
            state = context.user_state
            loss = context.info.loss
            loss_0 = jnp.where(context.step == 0, loss, state["loss_0"])
            log_target = jnp.log(jnp.asarray(target_loss + 1e-300, dtype))
            span = jnp.maximum(jnp.log(loss_0) - log_target, 1.0)
            frac = jnp.clip((jnp.log(loss) - log_target) / span, 0.0, 1.0)
            tol = (
                lm_set.cg_tol_tight
                * (lm_set.cg_tol_loose / lm_set.cg_tol_tight) ** frac
            )
            budget = lm_set.cg_maxiter_min + (1.0 - frac) * (
                lm_set.cg_maxiter_max - lm_set.cg_maxiter_min
            )
            step_iterations = (
                context.lm_state.recycle.iterations
                if recycle_on
                else jnp.asarray(0, dtype=jnp.int32)
            )
            new_state = {
                "loss_0": loss_0,
                "total_inner_iterations": state["total_inner_iterations"]
                + step_iterations,
                "iterations_trajectory": state["iterations_trajectory"]
                .at[context.step]
                .set(step_iterations),
            }
            new_hyper = dataclasses.replace(
                context.lm_state.hyper,
                iterative_tol=jnp.asarray(tol, dtype),
                iterative_maxiter=jnp.round(budget).astype(jnp.int32),
            )
            # optional damping floor: keep the dual operator out of the
            # small-lambda regime the C3 preconditioner cannot condition.
            floored_damping = jnp.maximum(
                context.lm_state.damping,
                jnp.asarray(lm_set.cg_min_damping, dtype),
            )
            return LMSolveAction(
                lm_state=dataclasses.replace(
                    context.lm_state, hyper=new_hyper, damping=floored_damping
                ),
                user_state=new_state,
            )

        def initial_user_state(steps):
            return {
                "loss_0": jnp.asarray(1.0, dtype),
                "total_inner_iterations": jnp.asarray(0, dtype=jnp.int32),
                "iterations_trajectory": jnp.zeros(steps + 1, dtype=jnp.int32),
            }

        solver = LevenbergMarquardt(
            residual_fn,
            linear_solver="cg",
            init_damping=lm_set.init_damping,
            damping_decrease=lm_set.damping_decrease,
            damping_increase=lm_set.damping_increase,
            iterative_tol=lm_set.cg_tol_loose,
            iterative_atol=lm_set.cg_atol,
            iterative_maxiter=lm_set.cg_maxiter_max,
            preconditioner_factory=factory,
            implicit_tol=lm_set.implicit_tol,
            implicit_maxiter=lm_set.implicit_maxiter,
            metric=metric,
            recycle=recycle,
            geodesic_acceleration=lm_set.cg_geodesic,
        )

        def run_solve(max_steps):
            return solver.solve(
                theta_0,
                max_steps=max_steps,
                atol=lm_set.residual_atol,
                callback=cg_callback,
                user_state=initial_user_state(max_steps),
            )

        def run_phase_two(x_plateau):
            # Build the parameter-space Schur R once at the plateau iterate with a
            # pinned damping, then a right-preconditioned whitened-LSMR solve.
            build_start = time.perf_counter()
            schur = build_schur_preconditioner(
                x_plateau,
                K,
                K_tilde,
                reference_state,
                parameters,
                eps,
                lm_set.phase2_damping,
                mode=lm_set.phase2_mode,
            )
            jax.block_until_ready(schur)
            build_time = time.perf_counter() - build_start
            whitened = WhitenedPreconditioner(
                lambda v, damping: schur.solve(v),
                lambda w, damping: schur.solve_transpose(w),
            )
            phase2_solver = LevenbergMarquardt(
                residual_fn,
                linear_solver="lsmr",
                init_damping=lm_set.phase2_damping,
                damping_decrease=1.0,
                damping_increase=1.0,
                iterative_tol=lm_set.phase2_tol,
                iterative_atol=lm_set.phase2_tol,
                iterative_maxiter=lm_set.phase2_maxiter,
                whitened_preconditioner=whitened,
                metric=metric,
                geodesic_acceleration=False,
            )
            warmup = phase2_solver.solve(
                x_plateau, max_steps=1, atol=lm_set.residual_atol
            )
            jax.block_until_ready(warmup)
            solve_start = time.perf_counter()
            phase2 = phase2_solver.solve(
                x_plateau,
                max_steps=lm_set.phase2_max_steps,
                atol=lm_set.residual_atol,
            )
            jax.block_until_ready(phase2)
            phase2_time = time.perf_counter() - solve_start
            return phase2, build_time, phase2_time
    else:
        solver = LevenbergMarquardt(
            residual_fn,
            linear_solver=lm_set.linear_solver,
            init_damping=lm_set.init_damping,
            damping_decrease=lm_set.damping_decrease,
            damping_increase=lm_set.damping_increase,
            dual_solve_dtype=(
                jnp.float64 if lm_set.linear_solver == "cholesky" else None
            ),
            metric=metric,
        )

        def run_solve(max_steps):
            return solver.solve(theta_0, max_steps=max_steps, atol=lm_set.residual_atol)

    compile_start = time.perf_counter()
    compile_result = run_solve(1)
    jax.block_until_ready(compile_result)
    compile_time = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    phase1_result = run_solve(lm_set.max_steps)
    jax.block_until_ready(phase1_result)
    solve_time = time.perf_counter() - solve_start
    if lm_set.linear_solver == "cg":
        total_inner_iterations = int(phase1_result.user_state["total_inner_iterations"])
        iterations_trajectory = np.asarray(
            phase1_result.user_state["iterations_trajectory"][
                : int(phase1_result.steps)
            ]
        ).tolist()
    else:
        total_inner_iterations = None
        iterations_trajectory = None

    two_phase = lm_set.linear_solver == "cg" and lm_set.two_phase_endgame
    if two_phase:
        result, phase2_build_time, phase2_solve_time = run_phase_two(phase1_result.x)
        phase2_steps = int(result.steps)
    else:
        result = phase1_result
        phase2_build_time = None
        phase2_solve_time = None
        phase2_steps = None
    theta = result.x
    final_loss = float(result.info.loss)
    final_steps = int(phase1_result.steps)
    final_status = int(result.status)
    total_wall = solve_time + (phase2_build_time or 0.0) + (phase2_solve_time or 0.0)
    if data_set.rank_audit:
        theta_flat, unravel_theta = ravel_pytree(theta)
        dae_jacobian = np.asarray(
            jax.jacrev(lambda flat: residual_fn(unravel_theta(flat)))(theta_flat)
        )
        dae_singular_values = np.linalg.svd(dae_jacobian, compute_uv=False)
        rank_tolerance = (
            max(dae_jacobian.shape)
            * np.finfo(dae_jacobian.dtype).eps
            * dae_singular_values[0]
        )
        dae_jacobian_rank = int(np.sum(dae_singular_values > rank_tolerance))
        dae_min_singular_value = float(dae_singular_values[-1])
    else:
        dae_jacobian_rank = None
        dae_min_singular_value = None
    rate_coordinates, _, _, train_log_static = split_algebraic(theta[-1])
    train_eval = evaluate_coordinates(
        theta,
        K,
        K_tilde,
        rate_coordinates,
        train_log_static,
    )

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        coordinates = path_coordinates(theta, K_eval, K_tilde_eval)
        if data_set.static_evaluation == "pchip":
            log_static = jnp.asarray(
                scipy.interpolate.PchipInterpolator(
                    np.asarray(train_data),
                    np.asarray(train_log_static),
                    axis=0,
                )(np.asarray(points)),
                dtype=k_0_array.dtype,
            )
        else:
            log_static_values = []
            log_initial = train_log_static[0]
            for index in range(len(points)):
                static, _ = static_trade_equilibrium(
                    coordinates["k"][index],
                    coordinates["tobin_q"][index],
                    coordinates["marginal_wealth"][index],
                    parameters,
                    log_initial=log_initial,
                )
                log_initial = static["log_static"]
                log_static_values.append(log_initial)
            log_static = jnp.stack(log_static_values)
        return multicountry_trade_path_values(
            **coordinates,
            log_static=log_static,
            parameters=parameters,
        ), log_static

    validation_eval, validation_log_static = evaluate_grid(validation_data)
    test_eval, test_log_static = evaluate_grid(test_data)
    quality_T = min(data_set.quality_T, float(validation_data[-1]))
    quality_count = int(
        np.searchsorted(np.asarray(validation_data), quality_T, side="right")
    )
    quality_slice = slice(quality_count)
    tail_start = int(
        np.searchsorted(
            np.asarray(validation_data),
            0.9 * float(validation_data[-1]),
        )
    )
    plateau_index = int(
        np.argmin(
            np.abs(
                np.asarray(validation_data) - (2.0 / 3.0) * float(validation_data[-1])
            )
        )
    )
    max_train_residual = float(
        multicountry_trade_max_dae_residual(
            train_eval,
            train_log_static,
            parameters,
        )
    )
    max_validation_residual = float(
        multicountry_trade_max_dae_residual(
            validation_eval,
            validation_log_static,
            parameters,
            quality_slice,
        )
    )
    max_full_horizon_validation_residual = float(
        multicountry_trade_max_dae_residual(
            validation_eval,
            validation_log_static,
            parameters,
        )
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"][quality_slice]))
    )
    max_full_horizon_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"]))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"][quality_slice]))
    )
    max_full_horizon_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"]))
    )
    validation_residual_norms = multicountry_trade_pointwise_residual_norms(
        validation_eval,
        validation_log_static,
        parameters,
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name][selection]))
            and jnp.min(values[name][selection]) > 0.0
            for values, selection in (
                (train_eval, slice(None)),
                (validation_eval, quality_slice),
                (test_eval, slice(None)),
            )
            for name in (
                "k",
                "q",
                "marginal_wealth",
                "consumption",
                "investment",
            )
        )
        and all(
            jnp.all(jnp.isfinite(values[name][selection]))
            for values, selection in (
                (train_eval, slice(None)),
                (validation_eval, quality_slice),
                (test_eval, slice(None)),
            )
            for name in ("a", "interest_rate", "current_account")
        )
    )
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    if max_asset_clearing_residual > ACCOUNTING_RESIDUAL_TOL:
        rejection_reasons.append("asset_clearing")
    if max_rate_consistency_residual > ACCOUNTING_RESIDUAL_TOL:
        rejection_reasons.append("rate_consistency")

    tvc = multicountry_trade_tvc_diagnostics(
        validation_data, validation_eval, parameters
    )

    def endpoint_max(name: str, index: int) -> float:
        return float(jnp.max(jnp.abs(tvc[name][index])))

    def endpoint_decay_ratio(name: str) -> float:
        initial = endpoint_max(name, 0)
        terminal = endpoint_max(name, -1)
        scale = max(1e-10, 100.0 * np.finfo(np.asarray(tvc[name]).dtype).eps)
        if initial <= scale:
            return 0.0
        return terminal / initial

    def endpoint_decay_ratio_applicable(name: str) -> bool:
        initial = endpoint_max(name, 0)
        scale = max(1e-10, 100.0 * np.finfo(np.asarray(tvc[name]).dtype).eps)
        return initial > scale

    coefficient_groups = {
        "k": theta[0],
        "a": theta[1],
        "q": theta[2],
        "m": theta[3],
    }
    rkhs_norms = {
        f"{name}_{country + 1}": alpha_metric.norm(coefficients[country]) ** 2
        - eps * (coefficients[country] @ coefficients[country])
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_total = sum(rkhs_norms.values())
    unknown_count = num_dynamic_functions * n_train + algebraic_size

    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "rho": parameters.rho,
        "installation_elasticity": parameters.installation_elasticity,
        "trade_elasticity": parameters.trade_elasticity,
        "num_countries": num_countries,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": "float64",
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "quality_T": quality_T,
        "quality_validation_point_count": quality_count,
        "num_train_points": n_train,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_full_horizon_validation_residual": (max_full_horizon_validation_residual),
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_full_horizon_asset_clearing_residual": (
            max_full_horizon_asset_clearing_residual
        ),
        "max_rate_consistency_residual": max_rate_consistency_residual,
        "max_full_horizon_rate_consistency_residual": (
            max_full_horizon_rate_consistency_residual
        ),
        **{
            f"max_validation_{name}": float(jnp.max(values[quality_slice]))
            for name, values in validation_residual_norms.items()
        },
        **{
            f"max_full_horizon_{name}": float(jnp.max(values))
            for name, values in validation_residual_norms.items()
        },
        "terminal_residual_count": 0,
        "dae_residual_count": residual_count,
        "unknown_count": unknown_count,
        "dynamic_rkhs_function_count": num_dynamic_functions,
        "static_algebraic_value_count": static_size,
        "algebraic_rate_value_count": n_train,
        "free_initial_value_count": 2 * num_countries,
        "metric_selection_dimension": unknown_count - residual_count,
        "dae_jacobian_rank": dae_jacobian_rank,
        "dae_jacobian_min_singular_value": dae_min_singular_value,
        "numeraire": "population_weighted_geometric_consumption_price",
        "selection": "minimum_shifted_rkhs_metric",
        "initialization": "normalized_static_reference",
        "global_closure": "all_country_budgets_with_ex_post_asset_clearing",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "dtype": str(k_0_array.dtype),
        "minimum_interest_rate": float(
            jnp.min(validation_eval["interest_rate"][quality_slice])
        ),
        "full_horizon_minimum_interest_rate": float(
            jnp.min(validation_eval["interest_rate"])
        ),
        "ex_post_discounted_assets_initial": endpoint_max("discounted_assets", 0),
        "ex_post_discounted_assets": endpoint_max("discounted_assets", -1),
        "ex_post_discounted_assets_decay_ratio": endpoint_decay_ratio(
            "discounted_assets"
        ),
        "ex_post_discounted_assets_decay_ratio_applicable": (
            endpoint_decay_ratio_applicable("discounted_assets")
        ),
        "ex_post_asset_tvc_initial": endpoint_max("asset_tvc", 0),
        "ex_post_asset_tvc": endpoint_max("asset_tvc", -1),
        "ex_post_asset_tvc_decay_ratio": endpoint_decay_ratio("asset_tvc"),
        "ex_post_asset_tvc_decay_ratio_applicable": endpoint_decay_ratio_applicable(
            "asset_tvc"
        ),
        "ex_post_capital_tvc_initial": endpoint_max("capital_tvc", 0),
        "ex_post_capital_tvc": endpoint_max("capital_tvc", -1),
        "ex_post_capital_tvc_decay_ratio": endpoint_decay_ratio("capital_tvc"),
        "ex_post_capital_tvc_decay_ratio_applicable": (
            endpoint_decay_ratio_applicable("capital_tvc")
        ),
        "terminal_interest_rate": float(validation_eval["interest_rate"][-1]),
        "terminal_interest_rate_gap": float(
            abs(validation_eval["interest_rate"][-1] - parameters.rho)
        ),
        "plateau_time": float(validation_data[plateau_index]),
        "plateau_interest_rate": float(validation_eval["interest_rate"][plateau_index]),
        "plateau_interest_rate_gap": float(
            abs(validation_eval["interest_rate"][plateau_index] - parameters.rho)
        ),
        "rkhs_norm_total": float(rkhs_norm_total),
        "solve_time": solve_time,
        "wall_time": total_wall,
        "compile_time": compile_time,
        "solver": "nlls-gram",
        "linear_solver_resolved": lm_set.linear_solver,
        "two_phase_endgame": two_phase,
        "phase1_solve_time": solve_time,
        "phase2_build_time": phase2_build_time,
        "phase2_solve_time": phase2_solve_time,
        "phase2_steps": phase2_steps,
        "total_inner_iterations": total_inner_iterations,
        "inner_iterations_trajectory": iterations_trajectory,
        "recycle_rank": lm_set.recycle_rank if lm_set.linear_solver == "cg" else None,
        "metric_eps": eps,
        "train_loss": final_loss,
        "steps": final_steps,
        "stopping_reason": STATUS_TO_REASON.get(final_status, "unknown"),
        "solution_accepted": not rejection_reasons,
        "rejection_reason": (
            "accepted" if not rejection_reasons else ",".join(rejection_reasons)
        ),
        "static_reference_max_residual": static_reference_diagnostics["max_residual"],
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
    }
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"population_{suffix}": parameters.populations[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
                f"q_0_{suffix}": float(test_eval["q"][0, country]),
                f"m_0_{suffix}": float(test_eval["marginal_wealth"][0, country]),
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"terminal_k_{suffix}": float(validation_eval["k"][-1, country]),
                f"terminal_consumption_{suffix}": float(
                    validation_eval["consumption"][-1, country]
                ),
                f"terminal_investment_rate_{suffix}": float(
                    validation_eval["investment"][-1, country]
                    / validation_eval["k"][-1, country]
                ),
                f"terminal_current_account_{suffix}": float(
                    validation_eval["current_account"][-1, country]
                ),
                f"terminal_trade_balance_{suffix}": float(
                    validation_eval["trade_balance"][-1, country]
                ),
                f"terminal_steady_trade_gap_{suffix}": float(
                    validation_eval["trade_balance"][-1, country]
                    + validation_eval["interest_rate"][-1]
                    * validation_eval["a"][-1, country]
                ),
                f"plateau_a_{suffix}": float(
                    validation_eval["a"][plateau_index, country]
                ),
                f"plateau_k_{suffix}": float(
                    validation_eval["k"][plateau_index, country]
                ),
                f"plateau_consumption_{suffix}": float(
                    validation_eval["consumption"][plateau_index, country]
                ),
                f"plateau_investment_rate_{suffix}": float(
                    validation_eval["investment"][plateau_index, country]
                    / validation_eval["k"][plateau_index, country]
                ),
                f"plateau_investment_rate_gap_{suffix}": float(
                    validation_eval["investment"][plateau_index, country]
                    / validation_eval["k"][plateau_index, country]
                    - parameters.delta
                ),
                f"plateau_current_account_{suffix}": float(
                    validation_eval["current_account"][plateau_index, country]
                ),
                f"plateau_trade_balance_{suffix}": float(
                    validation_eval["trade_balance"][plateau_index, country]
                ),
                f"plateau_steady_trade_gap_{suffix}": float(
                    validation_eval["trade_balance"][plateau_index, country]
                    + validation_eval["interest_rate"][plateau_index]
                    * validation_eval["a"][plateau_index, country]
                ),
                f"tail_asset_change_{suffix}": float(
                    validation_eval["a"][-1, country]
                    - validation_eval["a"][tail_start, country]
                ),
                f"tail_log_capital_change_{suffix}": float(
                    jnp.log(
                        validation_eval["k"][-1, country]
                        / validation_eval["k"][tail_start, country]
                    )
                ),
                f"tail_log_consumption_change_{suffix}": float(
                    jnp.log(
                        validation_eval["consumption"][-1, country]
                        / validation_eval["consumption"][tail_start, country]
                    )
                ),
                f"tail_max_current_account_{suffix}": float(
                    jnp.max(
                        jnp.abs(
                            validation_eval["current_account"][tail_start:, country]
                        )
                    )
                ),
                **{
                    f"rkhs_norm_{name}_{suffix}": float(rkhs_norms[f"{name}_{suffix}"])
                    for name in coefficient_groups
                },
            }
        )

    if num_countries >= 3:
        symmetry_paths = {
            "capital": "k",
            "assets": "a",
            "tobin_q": "q",
            "marginal_wealth": "marginal_wealth",
            "consumption": "consumption",
            "investment_rate": "investment_rate",
            "trade_balance": "trade_balance",
            "current_account": "current_account",
            **{name: name for name in STATIC_BLOCK_NAMES},
        }
        for label, name in symmetry_paths.items():
            results[f"max_country_2_3_{label}_gap"] = float(
                jnp.max(
                    jnp.abs(
                        validation_eval[name][quality_slice, 1]
                        - validation_eval[name][quality_slice, 2]
                    )
                )
            )
        permutation = jnp.asarray((0, 2, 1, *range(3, num_countries)))
        permuted_trade_shares = jnp.take(
            jnp.take(validation_eval["trade_shares"], permutation, axis=-2),
            permutation,
            axis=-1,
        )
        results["max_country_2_3_trade_share_permutation_gap"] = float(
            jnp.max(
                jnp.abs(
                    validation_eval["trade_shares"][quality_slice]
                    - permuted_trade_shares[quality_slice]
                )
            )
        )

    def frame(points, values, log_static):
        data = {"t": np.asarray(points)}
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "marginal_wealth",
                "consumption",
                "investment",
                "investment_rate",
                "trade_balance",
                "current_account",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        data["interest_rate"] = np.asarray(values["interest_rate"])
        data["consumption_basket_log_price"] = np.asarray(
            consumption_basket_log_price(log_static, parameters)
        )
        for name, residual_norm in multicountry_trade_pointwise_residual_norms(
            values,
            log_static,
            parameters,
        ).items():
            data[name] = np.asarray(residual_norm)
        return pd.DataFrame(data)

    extra = {
        "train_df": frame(train_data, train_eval, train_log_static),
        "validation_df": frame(
            validation_data,
            validation_eval,
            validation_log_static,
        ),
        "test_df": frame(test_data, test_eval, test_log_static),
        "hyperparameters": {
            "parameters": asdict(parameters),
            "kernel_set": asdict(kernel_set),
            "data_set": asdict(data_set),
            "lm_set": asdict(lm_set),
            "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
        },
    }
    return results, extra


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_trade_growth_kernel_nlls)
