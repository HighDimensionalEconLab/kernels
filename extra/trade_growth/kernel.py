from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import scipy.linalg
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    CG,
    QR,
    AnnealRidge,
    BlockEigenPreconditioner,
    Cholesky,
    GramCG,
    LevenbergMarquardt,
    LMAction,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

from extra.trade_growth.model import (
    DYNAMIC_RESIDUAL_NAMES,
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    consumption_basket_log_price,
    multicountry_trade_max_dae_residual,
    multicountry_trade_path_values,
    multicountry_trade_pointwise_residual_norms,
    multicountry_trade_tvc_diagnostics,
    static_trade_equilibrium,
    trade_growth_parameter_values,
)
from extra.trade_growth.preconditioners import (
    TradeGrowthBalancedPreconditioner,
    TradeGrowthEquationBlockPreconditioner,
    TradeGrowthHybridPreconditioner,
    TradeGrowthModePreconditioner,
    TradeGrowthMultiplicativePreconditioner,
    TradeGrowthTimeBlockPreconditioner,
    assemble_trade_growth_equation_blocks,
    assemble_trade_growth_time_blocks,
    trade_growth_block_eigen_families,
    trade_growth_block_permutation,
    trade_growth_local_linearization,
    trade_growth_mode_blocks,
)
from kernels import integrated_kernel_matrices, normalize_kernel, rkhs_norm_squared
from multicountry_growth import InitialConditionSettings, draw_initial_conditions
from multicountry_growth_kernel import make_train_grid

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
class ToleranceSettings:
    train_residual: float = TRAIN_RESIDUAL_TOL
    validation_residual: float = VALIDATION_RESIDUAL_TOL
    accounting: float = ACCOUNTING_RESIDUAL_TOL


@dataclass(frozen=True)
class KernelSettings:
    kernel: str = "matern52"
    ell: float = 10.0
    sigma: float = 1.0


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 100.0
    train_points: int = 48
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 8
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401
    quality_T: float = 100.0
    rank_audit: bool = False
    spectrum_audit: bool = False


@dataclass(frozen=True)
class SolverSettings:
    optimizer: str = "ridge_lm"
    metric_eps: float = 1e-7
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 100
    residual_atol: float = 0.0
    gtol: float = 5e-12
    jacobian_mode: str = "auto"
    cache_jacobian: bool = True
    linear_solver: str = "auto"
    geodesic_acceleration: bool = True
    ridge: float = 3e-12
    ridge_init: float | None = 1e-7
    ridge_decrease: float = 0.01
    ridge_grad_rtol: float = 1e-3
    ridge_stall_rtol: float = 0.0
    cg_tol: float = 1e-10
    cg_maxiter: int = 2500
    cg_tol_initial: float | None = None
    cg_maxiter_initial: int = 60
    cg_maxiter_middle: int = 200
    preconditioner: str = "balanced_modes"
    preconditioner_refresh_steps: int = 0
    asset_homotopy_steps: int = 4
    homotopy_max_steps: int = 200
    homotopy_stall_rtol: float = 0.99


def multicountry_trade_growth_kernel(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: TradeGrowthParameters = TradeGrowthParameters(),
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
    if k_0 is None or a_0 is None:
        drawn_k_0, drawn_a_0 = draw_initial_conditions(
            initial_set, parameters, dtype=dtype
        )
        if k_0 is None:
            k_0 = drawn_k_0
        if a_0 is None:
            a_0 = drawn_a_0
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive capital stock per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one NFA position per country")
    supported_optimizers = ("ridge_lm", "metric_lm")
    if lm_set.optimizer not in supported_optimizers:
        raise ValueError(f"optimizer must be one of {supported_optimizers}")
    if lm_set.metric_eps <= 0.0:
        raise ValueError("metric_eps must be positive")
    supported_jacobian_modes = ("auto", "fwd", "rev")
    if lm_set.jacobian_mode not in supported_jacobian_modes:
        raise ValueError(f"jacobian_mode must be one of {supported_jacobian_modes}")
    supported_linear_solvers = ("auto", "cholesky", "qr", "cg", "gram_cg")
    if lm_set.linear_solver not in supported_linear_solvers:
        raise ValueError(f"linear_solver must be one of {supported_linear_solvers}")
    if lm_set.preconditioner_refresh_steps < 0:
        raise ValueError("preconditioner_refresh_steps must be nonnegative")
    if lm_set.cg_tol_initial is not None and lm_set.cg_tol_initial < lm_set.cg_tol:
        raise ValueError("cg_tol_initial must be no smaller than cg_tol")
    if min(lm_set.cg_maxiter_initial, lm_set.cg_maxiter_middle) < 1:
        raise ValueError("scheduled CG iteration budgets must be positive")
    if lm_set.asset_homotopy_steps < 0:
        raise ValueError("asset_homotopy_steps must be nonnegative")
    if lm_set.homotopy_max_steps < 1:
        raise ValueError("homotopy_max_steps must be positive")
    if not 0.0 <= lm_set.homotopy_stall_rtol <= 1.0:
        raise ValueError("homotopy_stall_rtol must lie in [0, 1]")
    if lm_set.preconditioner not in (
        "country_blocks",
        "country_modes",
        "balanced_modes",
        "time_blocks",
        "equation_blocks",
        "hybrid_blocks",
        "multiplicative_blocks",
    ):
        raise ValueError(
            "preconditioner must be 'country_blocks', 'country_modes', "
            "'balanced_modes', 'time_blocks', 'equation_blocks', or "
            "'hybrid_blocks', or 'multiplicative_blocks'"
        )
    if lm_set.linear_solver == "gram_cg" and lm_set.optimizer != "metric_lm":
        raise ValueError("gram_cg is available only for metric_lm")
    residual_preconditioners = (
        "time_blocks",
        "equation_blocks",
        "hybrid_blocks",
        "multiplicative_blocks",
    )
    if (
        lm_set.linear_solver == "gram_cg"
        and lm_set.preconditioner not in residual_preconditioners
    ):
        raise ValueError("gram_cg requires a residual-space preconditioner")
    if (
        lm_set.linear_solver == "cg"
        and lm_set.preconditioner in residual_preconditioners
    ):
        raise ValueError("parameter-space cg cannot use the residual time blocks")

    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_train_grid(data_set).astype(dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if not 0.0 < data_set.quality_T <= float(train_data[-1]):
        raise ValueError("quality_T must lie within the training horizon")
    if sensitivity_time is not None and not (
        0.0 <= sensitivity_time <= float(train_data[-1])
    ):
        raise ValueError("sensitivity_time must lie within the training horizon")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    initial_asset_gap = jnp.sum(populations * a_0_array) / jnp.sum(populations)
    if abs(float(initial_asset_gap)) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")

    base_q = jnp.full(
        num_countries,
        1.0 / parameters.eta,
        dtype=dtype,
    )
    base_mu = jnp.ones(num_countries, dtype=dtype)
    base_static, _ = static_trade_equilibrium(
        k_0_array,
        base_q,
        base_mu,
        parameters,
    )
    basket_price = jnp.exp(
        consumption_basket_log_price(base_static["log_static"], parameters)
    )
    price_scale = 1.0 / basket_price
    q_initial_guess = price_scale * base_q
    mu_initial_guess = base_mu / price_scale
    static_initial_guess, static_initial_diagnostics = static_trade_equilibrium(
        k_0_array,
        q_initial_guess,
        mu_initial_guess,
        parameters,
        log_initial=base_static["log_static"] + jnp.log(price_scale),
    )
    log_static_initial_guess = static_initial_guess["log_static"]

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

    p = {
        "parameters": trade_growth_parameter_values(parameters, dtype),
        "k_0": k_0_array,
        "a_0": a_0_array,
    }
    num_static = len(STATIC_BLOCK_NAMES)
    num_country_functions = 4 + num_static
    num_functions = num_country_functions * num_countries + 1
    num_country_initials = 2 + num_static
    num_initials = num_country_initials * num_countries + 1

    def path_coordinates(theta, K_eval, K_tilde_eval, p_value):
        parameters_p = p_value["parameters"]
        populations_p = parameters_p.populations
        k_0_p = p_value["k_0"]
        a_0_p = p_value["a_0"] - jnp.sum(populations_p * p_value["a_0"]) / jnp.sum(
            populations_p
        )
        alpha_k, alpha_a, alpha_q, alpha_m, alpha_static, alpha_r, initial = theta
        log_q_0 = initial[:num_countries]
        log_mu_0 = initial[num_countries : 2 * num_countries]
        log_static_0 = initial[2 * num_countries : -1].reshape(
            num_static, num_countries
        )
        log_interest_rate_0 = initial[-1]
        log_k = jnp.log(k_0_p) + K_tilde_eval @ alpha_k.T
        a = a_0_p + K_tilde_eval @ alpha_a.T
        log_q = log_q_0 + K_tilde_eval @ alpha_q.T
        log_mu = log_mu_0 + K_tilde_eval @ alpha_m.T
        static_levels = jnp.einsum("tm,snm->tsn", K_tilde_eval, alpha_static)
        log_static = log_static_0 + static_levels
        log_interest_rate = log_interest_rate_0 + K_tilde_eval @ alpha_r
        return {
            "k": jnp.exp(log_k),
            "a": a,
            "q": jnp.exp(log_q),
            "mu": jnp.exp(log_mu),
            "interest_rate": jnp.exp(log_interest_rate),
            "log_static": log_static,
            "dlog_k_dt": K_eval @ alpha_k.T,
            "da_dt": K_eval @ alpha_a.T,
            "dlog_q_dt": K_eval @ alpha_q.T,
            "dlog_mu_dt": K_eval @ alpha_m.T,
        }

    def evaluate_coordinates(theta, K_eval, K_tilde_eval, p_value):
        coordinates = path_coordinates(theta, K_eval, K_tilde_eval, p_value)
        return multicountry_trade_path_values(
            **coordinates,
            parameters=p_value["parameters"],
        )

    residual_count = num_functions * n_train
    residual_scale = jnp.asarray(1.0 / np.sqrt(residual_count), dtype=dtype)

    def residual_fn(theta, _, p_value):
        coordinates = path_coordinates(theta, K, K_tilde, p_value)
        values = multicountry_trade_path_values(
            **coordinates,
            parameters=p_value["parameters"],
        )
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
                values["static_residuals"].reshape(-1),
                consumption_basket_log_price(
                    coordinates["log_static"], p_value["parameters"]
                ).reshape(-1),
            ]
        )

    metric_shift = lm_set.metric_eps if lm_set.optimizer == "metric_lm" else 0.0
    metric_free_scale = lm_set.metric_eps if lm_set.optimizer == "metric_lm" else 1.0
    shifted_metric_matrix = K_metric + metric_shift * jnp.eye(
        n_train, dtype=K_metric.dtype
    )
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(shifted_metric_matrix, upper=True).astype(dtype),
        repeats=num_functions,
        free_scale=metric_free_scale,
    )
    country_coefficient_0 = jnp.zeros((num_countries, n_train), dtype=dtype)
    theta_0 = (
        country_coefficient_0,
        country_coefficient_0,
        country_coefficient_0,
        country_coefficient_0,
        jnp.zeros((num_static, num_countries, n_train), dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
        jnp.concatenate(
            [
                jnp.log(q_initial_guess),
                jnp.log(mu_initial_guess),
                log_static_initial_guess.reshape(-1),
                jnp.log(jnp.asarray(parameters.r, dtype=dtype))[None],
            ]
        ),
    )

    if lm_set.optimizer == "metric_lm":
        initial_ridge = 0.0
        anneal = None
    elif lm_set.ridge_init is None:
        initial_ridge = lm_set.ridge
        anneal = None
    else:
        initial_ridge = lm_set.ridge_init
        anneal = AnnealRidge(
            ridge_floor=lm_set.ridge,
            decrease=lm_set.ridge_decrease,
            grad_rtol=lm_set.ridge_grad_rtol,
            stall_rtol=lm_set.ridge_stall_rtol,
        )

    def build_preconditioner(theta_value, p_value, ridge_value):
        if lm_set.preconditioner in residual_preconditioners:
            C, D, U, K_check, K_derivative = trade_growth_local_linearization(
                theta_value,
                K,
                K_tilde,
                metric.F,
                p_value,
                residual_scale,
                metric.free_scale,
            )
            coefficient_blocks = assemble_trade_growth_time_blocks(
                C, D, U, K_check, K_derivative
            )
            equation_blocks = assemble_trade_growth_equation_blocks(
                C, D, U, K_check, K_derivative
            )
            time_preconditioner = TradeGrowthTimeBlockPreconditioner(
                coefficient_blocks=coefficient_blocks,
                initial_jacobian=U,
                num_countries=num_countries,
            )
            if lm_set.preconditioner == "time_blocks":
                return time_preconditioner
            equation_preconditioner = TradeGrowthEquationBlockPreconditioner(
                blocks=equation_blocks,
                num_countries=num_countries,
            )
            if lm_set.preconditioner == "equation_blocks":
                return equation_preconditioner
            if lm_set.preconditioner == "hybrid_blocks":
                return TradeGrowthHybridPreconditioner(
                    time_blocks=time_preconditioner,
                    equation_blocks=equation_preconditioner,
                )
            return TradeGrowthMultiplicativePreconditioner(
                time_blocks=time_preconditioner,
                equation_blocks=equation_preconditioner,
                C=C,
                D=D,
                U=U,
                K_check=K_check,
                K_derivative=K_derivative,
            )
        if lm_set.preconditioner in (
            "country_modes",
            "balanced_modes",
        ):
            (
                mode_matrix,
                aggregate_block,
                idiosyncratic_blocks,
                aggregate_cross,
            ) = trade_growth_mode_blocks(
                theta_value,
                K,
                K_tilde,
                metric.F,
                p_value,
                residual_scale,
                ridge_value,
                metric.free_scale,
            )
            if lm_set.preconditioner == "balanced_modes":
                return TradeGrowthBalancedPreconditioner(
                    aggregate_block=aggregate_block,
                    idiosyncratic_blocks=idiosyncratic_blocks,
                    aggregate_cross=aggregate_cross,
                    mode_matrix=mode_matrix,
                )
            return TradeGrowthModePreconditioner(
                aggregate_block=aggregate_block,
                idiosyncratic_blocks=idiosyncratic_blocks,
                mode_matrix=mode_matrix,
            )
        return BlockEigenPreconditioner(
            trade_growth_block_eigen_families(
                theta_value,
                K,
                K_tilde,
                metric.F,
                p_value,
                residual_scale,
                ridge_value,
                metric.free_scale,
            ),
            jnp.asarray(trade_growth_block_permutation(num_countries, n_train)),
        )

    is_iterative = lm_set.linear_solver in ("cg", "gram_cg")
    scheduled_cg = is_iterative and lm_set.cg_tol_initial is not None
    initial_cg_tol = lm_set.cg_tol_initial if scheduled_cg else lm_set.cg_tol
    initial_cg_maxiter = (
        lm_set.cg_maxiter_initial if scheduled_cg else lm_set.cg_maxiter
    )
    initial_preconditioner = None
    if is_iterative:
        initial_preconditioner = build_preconditioner(
            theta_0,
            p,
            jnp.asarray(initial_ridge, dtype=dtype),
        )
    if lm_set.linear_solver == "qr":
        linear_solver = QR()
        ad_solver = None
    elif lm_set.linear_solver == "cg":
        linear_solver = CG(
            initial_preconditioner,
            tol=initial_cg_tol,
            maxiter=initial_cg_maxiter,
        )
        ad_solver = CG(None, tol=lm_set.cg_tol, maxiter=lm_set.cg_maxiter)
    elif lm_set.linear_solver == "gram_cg":
        linear_solver = GramCG(
            initial_preconditioner,
            tol=initial_cg_tol,
            maxiter=initial_cg_maxiter,
        )
        ad_solver = GramCG(None, tol=lm_set.cg_tol, maxiter=lm_set.cg_maxiter)
    else:
        linear_solver = Cholesky()
        ad_solver = None

    callback = None
    user_state = None

    def update_iterative_forcing(lm_state, loss):
        if not scheduled_cg:
            return lm_state
        residual_norm = jnp.sqrt(jnp.maximum(loss, 0.0))
        tolerance = jnp.clip(
            0.1 * residual_norm,
            jnp.asarray(lm_set.cg_tol, dtype=dtype),
            jnp.asarray(lm_set.cg_tol_initial, dtype=dtype),
        )
        budget = jnp.where(
            tolerance > 1e-3,
            lm_set.cg_maxiter_initial,
            jnp.where(
                tolerance > 1e-5,
                lm_set.cg_maxiter_middle,
                lm_set.cg_maxiter,
            ),
        ).astype(jnp.int32)
        return dataclasses.replace(
            lm_state,
            hyper=dataclasses.replace(
                lm_state.hyper,
                iterative_tol=tolerance,
                iterative_maxiter=budget,
            ),
        )

    if anneal is not None:
        user_state = anneal.init_state(dtype)
        if is_iterative:

            def callback(ctx):
                action = anneal(ctx)
                advanced = action.lm_state.ridge < ctx.lm_state.ridge
                if lm_set.preconditioner_refresh_steps > 0:
                    periodic = (
                        (ctx.step > 0)
                        & (ctx.step % lm_set.preconditioner_refresh_steps == 0)
                        & ctx.info.accepted
                    )
                    refresh = advanced | periodic
                else:
                    refresh = advanced
                preconditioner = jax.lax.cond(
                    refresh,
                    lambda: build_preconditioner(
                        ctx.x,
                        ctx.p,
                        action.lm_state.ridge,
                    ),
                    lambda: ctx.lm_state.preconditioner,
                )
                return dataclasses.replace(
                    action,
                    lm_state=update_iterative_forcing(
                        dataclasses.replace(
                            action.lm_state, preconditioner=preconditioner
                        ),
                        ctx.info.loss,
                    ),
                )

        else:
            callback = anneal
    elif is_iterative and (lm_set.preconditioner_refresh_steps > 0 or scheduled_cg):

        def callback(ctx):
            if lm_set.preconditioner_refresh_steps > 0:
                refresh = (
                    (ctx.step > 0)
                    & (ctx.step % lm_set.preconditioner_refresh_steps == 0)
                    & ctx.info.accepted
                )
            else:
                refresh = jnp.asarray(False)
            preconditioner = jax.lax.cond(
                refresh,
                lambda: build_preconditioner(
                    ctx.x,
                    ctx.p,
                    jnp.asarray(0.0, dtype=dtype)
                    if ctx.lm_state.ridge is None
                    else ctx.lm_state.ridge,
                ),
                lambda: ctx.lm_state.preconditioner,
            )
            return LMAction(
                lm_state=update_iterative_forcing(
                    dataclasses.replace(ctx.lm_state, preconditioner=preconditioner),
                    ctx.info.loss,
                ),
                user_state=ctx.user_state,
            )

    solver_kwargs = {
        "metric": metric,
        "linear_solver": linear_solver,
        "ad_solver": ad_solver,
        "init_damping": lm_set.init_damping,
        "damping_decrease": lm_set.damping_decrease,
        "damping_increase": lm_set.damping_increase,
        "jacobian_mode": lm_set.jacobian_mode,
        "cache_jacobian": lm_set.cache_jacobian,
        "geodesic_acceleration": lm_set.geodesic_acceleration,
    }
    if lm_set.optimizer == "metric_lm":
        solver = LevenbergMarquardt(residual_fn, **solver_kwargs)
    else:
        solver = RidgeLevenbergMarquardt(
            residual_fn,
            ridge=initial_ridge,
            **solver_kwargs,
        )
    solve_options = {
        "max_steps": lm_set.max_steps,
        "gtol": lm_set.gtol,
        "atol": lm_set.residual_atol,
        "max_steps_is_success": False,
    }
    continuation_options = (
        {} if callback is None else {"callback": callback, "user_state": user_state}
    )

    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        None,
        p=p,
        **solve_options,
        **continuation_options,
    )
    jax.block_until_ready(result)
    homotopy_used = False
    homotopy_steps = []
    if int(result.status) != LMStatus.CONVERGED and lm_set.asset_homotopy_steps > 0:
        homotopy_used = True
        if lm_set.optimizer == "metric_lm":
            homotopy_continuation = continuation_options
        elif lm_set.ridge_init is None:
            homotopy_continuation = {}
        else:
            homotopy_anneal = AnnealRidge(
                ridge_floor=lm_set.ridge,
                decrease=lm_set.ridge_decrease,
                grad_rtol=lm_set.ridge_grad_rtol,
                stall_rtol=max(
                    lm_set.ridge_stall_rtol,
                    lm_set.homotopy_stall_rtol,
                ),
            )
            if is_iterative:

                def homotopy_callback(ctx):
                    action = homotopy_anneal(ctx)
                    advanced = action.lm_state.ridge < ctx.lm_state.ridge
                    preconditioner = jax.lax.cond(
                        advanced,
                        lambda: build_preconditioner(
                            ctx.x,
                            ctx.p,
                            action.lm_state.ridge,
                        ),
                        lambda: ctx.lm_state.preconditioner,
                    )
                    return dataclasses.replace(
                        action,
                        lm_state=dataclasses.replace(
                            action.lm_state,
                            preconditioner=preconditioner,
                        ),
                    )

            else:
                homotopy_callback = homotopy_anneal
            homotopy_continuation = {
                "callback": homotopy_callback,
                "user_state": homotopy_anneal.init_state(dtype),
            }
        homotopy_options = {
            **solve_options,
            "max_steps": max(
                lm_set.max_steps,
                lm_set.homotopy_max_steps,
            ),
        }
        zero_asset_p = {**p, "a_0": jnp.zeros_like(p["a_0"])}
        warm_result = solver.solve(
            theta_0,
            None,
            p=zero_asset_p,
            **homotopy_options,
            **homotopy_continuation,
        )
        jax.block_until_ready(warm_result)
        homotopy_steps.append(int(warm_result.steps))
        warm_theta = warm_result.x
        for homotopy_step in range(1, lm_set.asset_homotopy_steps + 1):
            fraction = homotopy_step / lm_set.asset_homotopy_steps
            step_p = {**p, "a_0": fraction * p["a_0"]}
            warm_result = solver.solve(
                warm_theta,
                None,
                p=step_p,
                **homotopy_options,
                **homotopy_continuation,
            )
            jax.block_until_ready(warm_result)
            homotopy_steps.append(int(warm_result.steps))
            warm_theta = warm_result.x
        result = warm_result
    solve_time = time.perf_counter() - solve_start
    theta = result.x

    if lm_set.linear_solver == "qr":
        parametric_linear_solver = QR()
        parametric_ad_solver = None
    elif lm_set.linear_solver == "cg":
        parametric_linear_solver = CG(
            result.lm_state.preconditioner,
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
        parametric_ad_solver = CG(
            None,
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
    elif lm_set.linear_solver == "gram_cg":
        parametric_linear_solver = GramCG(
            result.lm_state.preconditioner,
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
        parametric_ad_solver = GramCG(
            None,
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
    else:
        parametric_linear_solver = Cholesky()
        parametric_ad_solver = None
    parametric_solver_kwargs = {
        **solver_kwargs,
        "linear_solver": parametric_linear_solver,
        "ad_solver": parametric_ad_solver,
    }
    if lm_set.optimizer == "metric_lm":
        parametric_solver = LevenbergMarquardt(residual_fn, **parametric_solver_kwargs)
    else:
        parametric_solver = RidgeLevenbergMarquardt(
            residual_fn,
            ridge=lm_set.ridge,
            **parametric_solver_kwargs,
        )

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        coordinates = path_coordinates(theta, K_eval, K_tilde_eval, p)
        return multicountry_trade_path_values(
            **coordinates,
            parameters=p["parameters"],
        ), coordinates["log_static"]

    train_eval, train_log_static = evaluate_grid(train_data)
    validation_eval, validation_log_static = evaluate_grid(validation_data)
    test_eval, test_log_static = evaluate_grid(test_data)
    quality_count = int(
        np.searchsorted(np.asarray(validation_data), data_set.quality_T, side="right")
    )
    quality_slice = slice(quality_count)
    max_train_residual = float(
        multicountry_trade_max_dae_residual(
            train_eval, train_log_static, p["parameters"]
        )
    )
    max_validation_residual = float(
        multicountry_trade_max_dae_residual(
            validation_eval,
            validation_log_static,
            p["parameters"],
            quality_slice,
        )
    )
    max_full_horizon_validation_residual = float(
        multicountry_trade_max_dae_residual(
            validation_eval, validation_log_static, p["parameters"]
        )
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"][quality_slice]))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"][quality_slice]))
    )
    max_budget_identity_gap = float(
        jnp.max(jnp.abs(validation_eval["budget_trade_gap"][quality_slice]))
    )
    max_world_trade_balance = float(
        jnp.max(jnp.abs(validation_eval["world_trade_balance"][quality_slice]))
    )
    max_trade_share_sum_gap = float(
        jnp.max(
            jnp.abs(
                jnp.sum(validation_eval["trade_shares"][quality_slice], axis=-1) - 1.0
            )
        )
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
                "interest_rate",
            )
        )
        and all(
            jnp.all(jnp.isfinite(values[name][selection]))
            for values, selection in (
                (train_eval, slice(None)),
                (validation_eval, quality_slice),
                (test_eval, slice(None)),
            )
            for name in ("a", "current_account")
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
    if max_world_trade_balance > tolerance_set.accounting:
        rejection_reasons.append("world_trade_balance")
    equation_diagnostics_pass = (
        max(max_budget_identity_gap, max_trade_share_sum_gap) < 1e-9
        and max_world_trade_balance <= tolerance_set.accounting
    )

    tail_start = int(0.9 * (len(validation_data) - 1))
    terminal_income_scale = jnp.mean(validation_eval["factor_income"][-1])
    tail_max_log_capital_change = float(
        jnp.max(
            jnp.abs(
                jnp.log(validation_eval["k"][-1] / validation_eval["k"][tail_start])
            )
        )
    )
    tail_max_log_consumption_change = float(
        jnp.max(
            jnp.abs(
                jnp.log(
                    validation_eval["consumption"][-1]
                    / validation_eval["consumption"][tail_start]
                )
            )
        )
    )
    tail_max_asset_change = float(
        jnp.max(jnp.abs(validation_eval["a"][-1] - validation_eval["a"][tail_start]))
        / terminal_income_scale
    )
    tail_interest_rate_gap = float(
        jnp.abs(validation_eval["interest_rate"][-1] - parameters.r) / parameters.r
    )
    tail_max_current_account = float(
        jnp.max(jnp.abs(validation_eval["current_account"][tail_start:]))
        / terminal_income_scale
    )

    theta_flat, unravel_theta = ravel_pytree(theta)
    dae_jacobian_rank = None
    dae_min_singular_value = None
    normal_condition_number = None
    preconditioned_condition_number = None
    normal_min_eigenvalue = None
    normal_max_eigenvalue = None
    preconditioned_min_eigenvalue = None
    preconditioned_max_eigenvalue = None
    spectrum_operator = None
    if data_set.rank_audit or data_set.spectrum_audit:
        dae_jacobian = np.asarray(
            jax.jacrev(lambda flat: residual_fn(unravel_theta(flat), None, p))(
                theta_flat
            )
        )
        singular_values = np.linalg.svd(dae_jacobian, compute_uv=False)
        rank_tolerance = (
            max(dae_jacobian.shape)
            * np.finfo(dae_jacobian.dtype).eps
            * singular_values[0]
        )
        dae_jacobian_rank = int(np.sum(singular_values > rank_tolerance))
        dae_min_singular_value = float(singular_values[-1])
        if data_set.spectrum_audit:
            metric_size = metric.size
            Jt = jnp.asarray(dae_jacobian.T)
            whitened_Jt = np.asarray(
                jnp.concatenate(
                    [
                        metric.factor_solve_transpose(Jt[:metric_size], None),
                        Jt[metric_size:]
                        / jnp.sqrt(jnp.asarray(metric.free_scale, dtype=dtype)),
                    ],
                    axis=0,
                )
            )
            ridge_final = (
                0.0 if result.lm_state.ridge is None else float(result.lm_state.ridge)
            )
            if lm_set.linear_solver == "gram_cg":
                normal = whitened_Jt.T @ whitened_Jt
                spectrum_operator = "dual_residual_gram"
            else:
                normal = whitened_Jt @ whitened_Jt.T
                spectrum_operator = "whitened_parameter_normal"
                if lm_set.optimizer == "ridge_lm":
                    metric_diagonal = np.arange(metric_size)
                    normal[metric_diagonal, metric_diagonal] += ridge_final
            normal = 0.5 * (normal + normal.T)
            normal_eigenvalues = np.linalg.eigvalsh(normal)
            normal_min_eigenvalue = float(normal_eigenvalues[0])
            normal_max_eigenvalue = float(normal_eigenvalues[-1])
            normal_numerical_floor = (
                np.finfo(normal.dtype).eps * normal.shape[0] * normal_eigenvalues[-1]
            )
            normal_positive = normal_eigenvalues[
                normal_eigenvalues > normal_numerical_floor
            ]
            normal_condition_number = float(normal_eigenvalues[-1] / normal_positive[0])
            audit_preconditioner = build_preconditioner(
                theta,
                p,
                jnp.asarray(ridge_final, dtype=dtype),
            )
            audit_context = SimpleNamespace(
                lm_state=SimpleNamespace(ridge=jnp.asarray(ridge_final, dtype=dtype))
            )
            inverse_preconditioner = np.asarray(
                jax.vmap(
                    lambda column: audit_preconditioner.apply(
                        column,
                        jnp.zeros((), dtype=dtype),
                        audit_context,
                    ),
                    in_axes=1,
                    out_axes=1,
                )(jnp.eye(normal.shape[0], dtype=dtype))
            )
            inverse_preconditioner = 0.5 * (
                inverse_preconditioner + inverse_preconditioner.T
            )
            inverse_factor = scipy.linalg.cholesky(
                inverse_preconditioner, lower=True, check_finite=False
            )
            symmetric_preconditioned = inverse_factor.T @ normal @ inverse_factor
            preconditioned_eigenvalues = scipy.linalg.eigvalsh(
                0.5 * (symmetric_preconditioned + symmetric_preconditioned.T),
                check_finite=False,
            )
            preconditioned_min_eigenvalue = float(preconditioned_eigenvalues[0])
            preconditioned_max_eigenvalue = float(preconditioned_eigenvalues[-1])
            preconditioned_numerical_floor = (
                np.finfo(preconditioned_eigenvalues.dtype).eps
                * preconditioned_eigenvalues.size
                * preconditioned_eigenvalues[-1]
            )
            preconditioned_positive = preconditioned_eigenvalues[
                preconditioned_eigenvalues > preconditioned_numerical_floor
            ]
            preconditioned_condition_number = float(
                preconditioned_eigenvalues[-1] / preconditioned_positive[0]
            )

    validation_residual_norms = multicountry_trade_pointwise_residual_norms(
        validation_eval, validation_log_static, p["parameters"]
    )
    tvc = multicountry_trade_tvc_diagnostics(
        validation_data, validation_eval, p["parameters"]
    )
    coefficient_groups = {
        "k": theta[0],
        "a": theta[1],
        "q": theta[2],
        "m": theta[3],
        **{name: theta[4][index] for index, name in enumerate(STATIC_BLOCK_NAMES)},
    }
    rkhs_norms = {
        f"{name}_{country + 1}": rkhs_norm_squared(coefficients[country], K_metric)
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_rate = rkhs_norm_squared(theta[5], K_metric)
    rkhs_norm_total = sum(rkhs_norms.values()) + rkhs_norm_rate

    sensitivity_df = None
    if sensitivity_time is not None:
        K_sensitivity, K_tilde_sensitivity = integrated_kernel_matrices(
            jnp.asarray([sensitivity_time], dtype=dtype),
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )

        def rate_at_sensitivity_time(p_value):
            solved = parametric_solver.solve(
                theta,
                None,
                p=p_value,
                **solve_options,
            )
            coordinates = path_coordinates(
                solved.x,
                K_sensitivity,
                K_tilde_sensitivity,
                p_value,
            )
            return coordinates["interest_rate"][0]

        rate_value, rate_grads = jax.value_and_grad(rate_at_sensitivity_time)(p)
        jax.block_until_ready(rate_grads)
        parameter_grads = rate_grads["parameters"]
        rows = [
            {"name": name, "country": 0, "value": float(value)}
            for name, value in (
                ("alpha", parameter_grads.alpha),
                ("gamma", parameter_grads.gamma),
                ("delta", parameter_grads.delta),
                ("r", parameter_grads.r),
                ("eta", parameter_grads.eta),
                ("trade_elasticity", parameter_grads.trade_elasticity),
                ("variety_elasticity", parameter_grads.variety_elasticity),
            )
        ]
        for name, values in (
            ("population", parameter_grads.populations),
            ("intermediate_productivity", parameter_grads.intermediate_productivities),
            ("consumption_productivity", parameter_grads.consumption_productivities),
            ("investment_productivity", parameter_grads.investment_productivities),
            (
                "consumption_value_added_share",
                parameter_grads.consumption_value_added_shares,
            ),
            (
                "intermediate_value_added_share",
                parameter_grads.intermediate_value_added_shares,
            ),
            (
                "investment_value_added_share",
                parameter_grads.investment_value_added_shares,
            ),
            ("k_0", rate_grads["k_0"]),
            ("a_0", rate_grads["a_0"]),
        ):
            rows.extend(
                {
                    "name": name,
                    "country": country + 1,
                    "value": float(values[country]),
                }
                for country in range(num_countries)
            )
        rows.extend(
            {
                "name": "trade_cost",
                "country": importer + 1,
                "exporter": exporter + 1,
                "value": float(parameter_grads.trade_costs[importer, exporter]),
            }
            for importer in range(num_countries)
            for exporter in range(num_countries)
        )
        sensitivity_df = pd.DataFrame(rows)

    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "r": parameters.r,
        "eta": parameters.eta,
        "trade_elasticity": parameters.trade_elasticity,
        "variety_elasticity": parameters.variety_elasticity,
        "num_countries": num_countries,
        "num_train_points": n_train,
        "train_T": float(train_data[-1]),
        "quality_T": data_set.quality_T,
        "quality_validation_point_count": quality_count,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": str(metric.F.dtype),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "differential_residual_count": 4 * num_countries * n_train,
        "country_algebraic_residual_count": num_static * num_countries * n_train,
        "global_algebraic_residual_count": n_train,
        "dae_residual_count": residual_count,
        "unknown_count": num_functions * n_train + num_initials,
        "penalized_coefficient_count": (
            num_functions * n_train if lm_set.optimizer == "ridge_lm" else 0
        ),
        "unpenalized_initial_value_count": (
            num_initials if lm_set.optimizer == "ridge_lm" else 0
        ),
        "dynamic_rkhs_function_count": 4 * num_countries,
        "country_algebraic_rkhs_function_count": num_static * num_countries,
        "common_algebraic_rkhs_function_count": 1,
        "rkhs_function_count": num_functions,
        "free_initial_value_count": num_initials,
        "metric_selection_dimension": (
            num_functions * n_train + num_initials
            if lm_set.optimizer == "metric_lm"
            else num_initials
        ),
        "metric_epsilon": lm_set.metric_eps,
        "metric_free_scale": float(metric.free_scale),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_full_horizon_validation_residual": (max_full_horizon_validation_residual),
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_rate_consistency_residual": max_rate_consistency_residual,
        "max_budget_identity_gap": max_budget_identity_gap,
        "max_world_trade_balance": max_world_trade_balance,
        "max_trade_share_sum_gap": max_trade_share_sum_gap,
        "equation_diagnostics_pass": equation_diagnostics_pass,
        "tail_max_log_capital_change": tail_max_log_capital_change,
        "tail_max_log_consumption_change": tail_max_log_consumption_change,
        "tail_max_asset_change": tail_max_asset_change,
        "tail_interest_rate_gap": tail_interest_rate_gap,
        "tail_max_current_account": tail_max_current_account,
        "dae_jacobian_rank": dae_jacobian_rank,
        "dae_jacobian_min_singular_value": dae_min_singular_value,
        "normal_condition_number": normal_condition_number,
        "preconditioned_condition_number": preconditioned_condition_number,
        "normal_min_eigenvalue": normal_min_eigenvalue,
        "normal_max_eigenvalue": normal_max_eigenvalue,
        "preconditioned_min_eigenvalue": preconditioned_min_eigenvalue,
        "preconditioned_max_eigenvalue": preconditioned_max_eigenvalue,
        "spectrum_operator": spectrum_operator,
        "rkhs_norm_total": float(rkhs_norm_total),
        "rkhs_norm_rate": float(rkhs_norm_rate),
        "solve_time": solve_time,
        "steps": int(result.steps),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "final_ridge": (
            0.0 if result.lm_state.ridge is None else float(result.lm_state.ridge)
        ),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "solution_accepted": not rejection_reasons,
        "rejection_reason": (
            "accepted" if not rejection_reasons else ",".join(rejection_reasons)
        ),
        "linear_solver": lm_set.linear_solver,
        "linear_solver_resolved": (
            "cholesky" if lm_set.linear_solver == "auto" else lm_set.linear_solver
        ),
        "optimizer": lm_set.optimizer,
        "selection": (
            "minimum_shifted_metric_correction_path"
            if lm_set.optimizer == "metric_lm"
            else "minimum_full_dae_rkhs_norm_ridge"
        ),
        "initialization": (
            "static_guess_then_zero_asset_homotopy"
            if homotopy_used
            else "static_equilibrium_guess_with_seeded_initials"
        ),
        "asset_homotopy_used": homotopy_used,
        "asset_homotopy_solve_steps": homotopy_steps,
        "global_closure": "all_country_budgets_with_ex_post_asset_clearing",
        "numeraire": "population_weighted_geometric_consumption_price",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "dtype": str(k_0_array.dtype),
        "static_initial_guess_max_residual": static_initial_diagnostics["max_residual"],
        "terminal_interest_rate": float(validation_eval["interest_rate"][-1]),
        "terminal_interest_rate_gap": float(
            abs(validation_eval["interest_rate"][-1] - parameters.r)
        ),
        "ex_post_discounted_assets": float(
            jnp.max(jnp.abs(tvc["discounted_assets"][-1]))
        ),
        "ex_post_asset_tvc": float(jnp.max(jnp.abs(tvc["asset_tvc"][-1]))),
        "ex_post_capital_tvc": float(jnp.max(jnp.abs(tvc["capital_tvc"][-1]))),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        **{
            f"max_validation_{name}": float(jnp.max(values[quality_slice]))
            for name, values in validation_residual_norms.items()
        },
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
                **{
                    f"rkhs_norm_{name}_{suffix}": float(rkhs_norms[f"{name}_{suffix}"])
                    for name in coefficient_groups
                },
            }
        )
    if sensitivity_time is not None:
        results["sensitivity_time"] = sensitivity_time
        results["interest_rate_at_sensitivity_time"] = float(rate_value)

    def frame(points, values, log_static):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "consumption_basket_log_price": np.asarray(
                consumption_basket_log_price(log_static, p["parameters"])
            ),
        }
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
                "factor_income",
                "trade_balance",
                "current_account",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
            for index, name in enumerate(STATIC_BLOCK_NAMES):
                data[f"{name}_{suffix}"] = np.asarray(
                    jnp.exp(log_static[:, index, country])
                )
            for name in DYNAMIC_RESIDUAL_NAMES:
                data[f"{name}_residual_{suffix}"] = np.asarray(values[name][:, country])
        for name, residual_norm in multicountry_trade_pointwise_residual_norms(
            values, log_static, p["parameters"]
        ).items():
            data[name] = np.asarray(residual_norm)
        data["asset_clearing"] = np.asarray(values["asset_clearing"])
        return pd.DataFrame(data)

    train_df = frame(train_data, train_eval, train_log_static)
    validation_df = frame(validation_data, validation_eval, validation_log_static)
    test_df = frame(test_data, test_eval, test_log_static)

    def evaluate_parametric(points, theta_value, p_value):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        return evaluate_coordinates(
            theta_value,
            K_eval,
            K_tilde_eval,
            p_value,
        )

    @jax.jit
    def kernel_solution(points):
        return evaluate_parametric(points, theta, p)

    def solve_parametric(p_value):
        return parametric_solver.solve(
            theta,
            None,
            p=p_value,
            **solve_options,
        )

    def parametric_model(points, p_value):
        solved = solve_parametric(p_value)
        return evaluate_parametric(points, solved.x, p_value)

    return results, {
        "model": kernel_solution,
        "parametric_model": parametric_model,
        "evaluate_parametric": evaluate_parametric,
        "solve_parametric": solve_parametric,
        "solver": parametric_solver,
        "continuation_solver": solver,
        "result": result,
        "solve_options": solve_options,
        "residual_fn": residual_fn,
        "theta": theta,
        "theta_0": theta_0,
        "p": p,
        "metric": metric,
        "K": K,
        "K_tilde": K_tilde,
        "train_data": train_data,
        "residual_scale": residual_scale,
        "preconditioner_families": trade_growth_block_eigen_families,
        "build_preconditioner": build_preconditioner,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "sensitivity_df": sensitivity_df,
        "hyperparameters": {
            "parameters": asdict(parameters),
            "kernel_set": asdict(kernel_set),
            "data_set": asdict(data_set),
            "lm_set": asdict(lm_set),
            "tolerance_set": asdict(tolerance_set),
            "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
        },
    }


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_trade_growth_kernel)
