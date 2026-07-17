from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
import scipy.interpolate
from jax import config
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    LMStatus,
    UnderdeterminedLevenbergMarquardt,
    metric_from_cholesky,
    metric_from_diagonal,
)

from kernel_metrics import repeated_blockdiag_metric
from kernels import integrated_kernel_matrices, normalize_kernel
from multicountry_growth_kernel_nlls import make_train_grid
from multicountry_trade_growth import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    consumption_basket_log_price,
    multicountry_trade_path_values,
    multicountry_trade_tvc_diagnostics,
    static_trade_equilibrium,
)

config.update("jax_enable_x64", True)

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


def multicountry_trade_growth_kernel_nlls(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: TradeGrowthParameters = TradeGrowthParameters(),
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
):
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
    if lm_set.linear_solver not in ("cholesky", "qr"):
        raise ValueError("linear_solver must be 'cholesky' or 'qr'")

    train_data = make_train_grid(data_set)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.static_evaluation not in ("pchip", "resolve"):
        raise ValueError("static_evaluation must be 'pchip' or 'resolve'")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points
    )
    n_train = len(train_data)
    kernel = normalize_kernel(kernel_set.kernel)
    k_0_array = jnp.asarray(k_0)
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

    K, K_tilde = integrated_kernel_matrices(
        train_data,
        train_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K = 0.5 * (K + K.T)
    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K + eps * jnp.eye(n_train))
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

    dynamic_count = 4 * num_countries * n_train
    static_count = len(STATIC_BLOCK_NAMES) * num_countries * n_train
    numeraire_count = n_train
    residual_count = dynamic_count + static_count + numeraire_count

    def residual_fn(theta):
        rate_coordinates, _, _, log_static = split_algebraic(theta[-1])
        values = evaluate_coordinates(
            theta,
            K,
            K_tilde,
            rate_coordinates,
            log_static,
        )
        dynamic_residual = jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
            ]
        )
        static_residual = values["static_residuals"].reshape(-1)
        numeraire_residual = consumption_basket_log_price(log_static, parameters)
        return jnp.concatenate(
            [
                dynamic_residual / jnp.sqrt(float(dynamic_count)),
                static_residual / jnp.sqrt(float(static_count)),
                numeraire_residual / jnp.sqrt(float(numeraire_count)),
            ]
        )

    num_dynamic_functions = 4 * num_countries
    metric = repeated_blockdiag_metric(
        alpha_metric,
        n_train,
        num_dynamic_functions,
        additional=(
            metric_from_diagonal(jnp.full(algebraic_size, eps)),
            algebraic_size,
        ),
    )
    initial_algebraic = jnp.concatenate(
        [
            jnp.zeros(n_train + 2 * num_countries),
            jnp.broadcast_to(
                static_reference["log_static"],
                (n_train, len(STATIC_BLOCK_NAMES), num_countries),
            ).reshape(-1),
        ]
    )
    theta_0 = (
        jnp.zeros((num_countries, n_train)),
        jnp.zeros((num_countries, n_train)),
        jnp.zeros((num_countries, n_train)),
        jnp.zeros((num_countries, n_train)),
        initial_algebraic,
    )
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        metric=metric,
    )

    compile_start = time.perf_counter()
    compile_result = solver.solve(theta_0, max_steps=1, atol=lm_set.residual_atol)
    jax.block_until_ready(compile_result)
    compile_time = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=lm_set.max_steps,
        atol=lm_set.residual_atol,
    )
    jax.block_until_ready(result)
    solve_time = time.perf_counter() - solve_start
    theta = result.x
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
    tail_start = int(
        np.searchsorted(
            np.asarray(validation_data),
            0.9 * float(validation_data[-1]),
        )
    )
    plateau_index = int(
        np.argmin(
            np.abs(
                np.asarray(validation_data)
                - (2.0 / 3.0) * float(validation_data[-1])
            )
        )
    )
    dynamic_names = ("capital", "assets", "tobin_q", "marginal_wealth_euler")
    max_train_residual = max(
        max(float(jnp.max(jnp.abs(train_eval[name]))) for name in dynamic_names),
        float(jnp.max(jnp.abs(train_eval["static_residuals"]))),
        float(
            jnp.max(
                jnp.abs(
                    consumption_basket_log_price(train_log_static, parameters)
                )
            )
        ),
    )
    max_validation_residual = max(
        max(float(jnp.max(jnp.abs(validation_eval[name]))) for name in dynamic_names),
        float(jnp.max(jnp.abs(validation_eval["static_residuals"]))),
        float(
            jnp.max(
                jnp.abs(
                    consumption_basket_log_price(validation_log_static, parameters)
                )
            )
        ),
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"]))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"]))
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in (
                "k",
                "q",
                "marginal_wealth",
                "consumption",
                "investment",
            )
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
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

    tvc = multicountry_trade_tvc_diagnostics(validation_data, validation_eval, parameters)
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
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_rate_consistency_residual": max_rate_consistency_residual,
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
        "ex_post_discounted_assets": float(jnp.max(jnp.abs(tvc["discounted_assets"][-1]))),
        "ex_post_asset_tvc": float(jnp.max(jnp.abs(tvc["asset_tvc"][-1]))),
        "ex_post_capital_tvc": float(jnp.max(jnp.abs(tvc["capital_tvc"][-1]))),
        "terminal_interest_rate": float(validation_eval["interest_rate"][-1]),
        "terminal_interest_rate_gap": float(
            abs(validation_eval["interest_rate"][-1] - parameters.rho)
        ),
        "plateau_time": float(validation_data[plateau_index]),
        "plateau_interest_rate": float(
            validation_eval["interest_rate"][plateau_index]
        ),
        "plateau_interest_rate_gap": float(
            abs(validation_eval["interest_rate"][plateau_index] - parameters.rho)
        ),
        "rkhs_norm_total": float(rkhs_norm_total),
        "solve_time": solve_time,
        "wall_time": solve_time,
        "compile_time": compile_time,
        "solver": "nlls-gram",
        "linear_solver_resolved": lm_set.linear_solver,
        "metric_eps": eps,
        "train_loss": float(result.info.loss),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
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
