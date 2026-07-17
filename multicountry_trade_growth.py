from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

STATIC_BLOCK_NAMES = (
    "wage",
    "rental_rate",
    "consumption_price",
    "intermediate_price",
    "investment_price",
    "consumption_revenue",
    "intermediate_revenue",
    "investment_revenue",
)

STATIC_RESIDUAL_NAMES = (
    "capital_market",
    "labor_market",
    "consumption_price",
    "intermediate_price",
    "investment_price",
    "consumption_market",
    "intermediate_market",
    "investment_market",
)

DYNAMIC_RESIDUAL_NAMES = (
    "capital",
    "assets",
    "tobin_q",
    "marginal_wealth_euler",
)


@dataclass(frozen=True)
class TradeGrowthParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    rho: float = -math.log(0.96)
    installation_elasticity: float = 0.76
    trade_elasticity: float = 4.0
    variety_elasticity: float = 2.0
    populations: tuple[float, ...] = (1.0, 1.0)
    intermediate_productivities: tuple[float, ...] = (1.0, 1.0)
    consumption_productivities: tuple[float, ...] = (1.0, 1.0)
    investment_productivities: tuple[float, ...] = (1.0, 1.0)
    consumption_value_added_shares: tuple[float, ...] = (0.75, 0.75)
    intermediate_value_added_shares: tuple[float, ...] = (0.5, 0.5)
    investment_value_added_shares: tuple[float, ...] = (0.5, 0.5)
    trade_costs: tuple[tuple[float, ...], ...] = ((1.0, 1.5), (1.5, 1.0))

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.delta <= 0.0 or self.rho <= 0.0:
            raise ValueError("delta and rho must be positive")
        if not 0.0 < self.installation_elasticity < 1.0:
            raise ValueError("installation_elasticity must lie in (0, 1)")
        if self.trade_elasticity <= 0.0:
            raise ValueError("trade_elasticity must be positive")
        if self.variety_elasticity <= 1.0:
            raise ValueError("variety_elasticity must exceed one")
        price_index_shape = (
            1.0 + (1.0 - self.variety_elasticity) / self.trade_elasticity
        )
        if price_index_shape <= 0.0:
            raise ValueError(
                "trade and variety elasticities must imply a finite price index"
            )
        if len(self.populations) < 2:
            raise ValueError("at least two countries are required")
        country_vectors = (
            self.intermediate_productivities,
            self.consumption_productivities,
            self.investment_productivities,
            self.consumption_value_added_shares,
            self.intermediate_value_added_shares,
            self.investment_value_added_shares,
        )
        if any(len(values) != self.num_countries for values in country_vectors):
            raise ValueError("every country vector must match populations")
        if any(value <= 0.0 for value in self.populations):
            raise ValueError("populations must be positive")
        if any(value <= 0.0 for values in country_vectors[:3] for value in values):
            raise ValueError("sector productivities must be positive")
        if any(
            not 0.0 < value < 1.0 for values in country_vectors[3:] for value in values
        ):
            raise ValueError("value-added shares must lie strictly inside (0, 1)")
        if len(self.trade_costs) != self.num_countries or any(
            len(row) != self.num_countries for row in self.trade_costs
        ):
            raise ValueError("trade_costs must be an N by N matrix")
        if any(value < 1.0 for row in self.trade_costs for value in row):
            raise ValueError("iceberg trade costs must be at least one")
        if any(
            not math.isclose(self.trade_costs[i][i], 1.0)
            for i in range(self.num_countries)
        ):
            raise ValueError("domestic trade costs must equal one")

    @property
    def num_countries(self):
        return len(self.populations)

    @property
    def installation_efficiency(self):
        return self.delta ** (1.0 - self.installation_elasticity)

    @property
    def price_index_constant(self):
        shape = 1.0 + (1.0 - self.variety_elasticity) / self.trade_elasticity
        return math.gamma(shape) ** (1.0 / (1.0 - self.variety_elasticity))


def symmetric_trade_growth_parameters(num_countries: int, trade_cost: float = 1.5):
    if num_countries < 2:
        raise ValueError("num_countries must be at least two")
    if trade_cost < 1.0:
        raise ValueError("trade_cost must be at least one")
    ones = (1.0,) * num_countries
    return TradeGrowthParameters(
        populations=ones,
        intermediate_productivities=ones,
        consumption_productivities=ones,
        investment_productivities=ones,
        consumption_value_added_shares=(0.75,) * num_countries,
        intermediate_value_added_shares=(0.5,) * num_countries,
        investment_value_added_shares=(0.5,) * num_countries,
        trade_costs=tuple(
            tuple(
                1.0 if importer == exporter else trade_cost
                for exporter in range(num_countries)
            )
            for importer in range(num_countries)
        ),
    )


def sector_unit_cost_log(
    log_rental_rate,
    log_wage,
    log_intermediate_price,
    value_added_share,
    productivity,
    parameters,
):
    capital_share = parameters.alpha * value_added_share
    labor_share = (1.0 - parameters.alpha) * value_added_share
    intermediate_share = 1.0 - value_added_share
    return (
        -jnp.log(productivity)
        + capital_share * (log_rental_rate - jnp.log(capital_share))
        + labor_share * (log_wage - jnp.log(labor_share))
        + intermediate_share * (log_intermediate_price - jnp.log(intermediate_share))
    )


def unpack_static_trade_values(log_static, parameters):
    if log_static.shape[-2:] != (
        len(STATIC_BLOCK_NAMES),
        parameters.num_countries,
    ):
        raise ValueError("log_static must have trailing shape (8, N)")
    return {
        name: jnp.exp(log_static[..., index, :])
        for index, name in enumerate(STATIC_BLOCK_NAMES)
    }


def static_trade_values(
    log_static,
    k,
    tobin_q,
    marginal_wealth,
    parameters,
):
    static = unpack_static_trade_values(log_static, parameters)
    dtype = jnp.result_type(log_static, k, tobin_q, marginal_wealth)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    intermediate_productivities = jnp.asarray(
        parameters.intermediate_productivities, dtype=dtype
    )
    consumption_productivities = jnp.asarray(
        parameters.consumption_productivities, dtype=dtype
    )
    investment_productivities = jnp.asarray(
        parameters.investment_productivities, dtype=dtype
    )
    consumption_value_added_shares = jnp.asarray(
        parameters.consumption_value_added_shares, dtype=dtype
    )
    intermediate_value_added_shares = jnp.asarray(
        parameters.intermediate_value_added_shares, dtype=dtype
    )
    investment_value_added_shares = jnp.asarray(
        parameters.investment_value_added_shares, dtype=dtype
    )
    log_trade_costs = jnp.log(jnp.asarray(parameters.trade_costs, dtype=dtype))

    log_wage = log_static[..., 0, :]
    log_rental_rate = log_static[..., 1, :]
    log_consumption_price = log_static[..., 2, :]
    log_intermediate_price = log_static[..., 3, :]
    log_investment_price = log_static[..., 4, :]
    log_consumption_revenue = log_static[..., 5, :]
    log_intermediate_revenue = log_static[..., 6, :]
    log_investment_revenue = log_static[..., 7, :]

    predicted_log_consumption_price = sector_unit_cost_log(
        log_rental_rate,
        log_wage,
        log_intermediate_price,
        consumption_value_added_shares,
        consumption_productivities,
        parameters,
    )
    predicted_log_investment_price = sector_unit_cost_log(
        log_rental_rate,
        log_wage,
        log_intermediate_price,
        investment_value_added_shares,
        investment_productivities,
        parameters,
    )
    log_intermediate_unit_cost = sector_unit_cost_log(
        log_rental_rate,
        log_wage,
        log_intermediate_price,
        intermediate_value_added_shares,
        jnp.ones(parameters.num_countries, dtype=dtype),
        parameters,
    )
    log_competitiveness = jnp.log(
        intermediate_productivities
    ) - parameters.trade_elasticity * (
        log_intermediate_unit_cost[..., None, :] + log_trade_costs
    )
    log_price_denominator = jax.nn.logsumexp(log_competitiveness, axis=-1)
    predicted_log_intermediate_price = (
        jnp.log(jnp.asarray(parameters.price_index_constant, dtype=dtype))
        - log_price_denominator / parameters.trade_elasticity
    )
    trade_shares = jax.nn.softmax(log_competitiveness, axis=-1)

    consumption = jnp.exp(
        -(jnp.log(marginal_wealth) + log_consumption_price) / parameters.gamma
    )
    investment_rate = jnp.exp(
        (
            jnp.log(parameters.installation_efficiency)
            + jnp.log(parameters.installation_elasticity)
            + jnp.log(tobin_q)
            - log_investment_price
        )
        / (1.0 - parameters.installation_elasticity)
    )
    investment = k * investment_rate

    predicted_log_consumption_revenue = (
        jnp.log(populations) + log_consumption_price + jnp.log(consumption)
    )
    predicted_log_investment_revenue = (
        jnp.log(populations) + log_investment_price + jnp.log(investment)
    )
    consumption_revenue = static["consumption_revenue"]
    intermediate_revenue = static["intermediate_revenue"]
    investment_revenue = static["investment_revenue"]
    value_added = (
        consumption_value_added_shares * consumption_revenue
        + intermediate_value_added_shares * intermediate_revenue
        + investment_value_added_shares * investment_revenue
    )
    intermediate_expenditure = (
        (1.0 - consumption_value_added_shares) * consumption_revenue
        + (1.0 - intermediate_value_added_shares) * intermediate_revenue
        + (1.0 - investment_value_added_shares) * investment_revenue
    )
    intermediate_demand = jnp.einsum(
        "...ij,...i->...j", trade_shares, intermediate_expenditure
    )
    predicted_log_intermediate_revenue = jnp.log(intermediate_demand)
    sector_revenues = jnp.stack(
        (consumption_revenue, intermediate_revenue, investment_revenue), axis=-2
    )
    sector_value_added_shares = jnp.stack(
        (
            consumption_value_added_shares,
            intermediate_value_added_shares,
            investment_value_added_shares,
        ),
        axis=-2,
    )
    sector_capital = (
        parameters.alpha
        * sector_value_added_shares
        * sector_revenues
        / static["rental_rate"][..., None, :]
    )
    sector_labor = (
        (1.0 - parameters.alpha)
        * sector_value_added_shares
        * sector_revenues
        / static["wage"][..., None, :]
    )
    sector_intermediates = (
        (1.0 - sector_value_added_shares)
        * sector_revenues
        / static["intermediate_price"][..., None, :]
    )

    residual_blocks = {
        "capital_market": log_rental_rate
        + jnp.log(k)
        + jnp.log(populations)
        - jnp.log(parameters.alpha * value_added),
        "labor_market": log_wage
        + jnp.log(populations)
        - jnp.log((1.0 - parameters.alpha) * value_added),
        "consumption_price": log_consumption_price - predicted_log_consumption_price,
        "intermediate_price": log_intermediate_price - predicted_log_intermediate_price,
        "investment_price": log_investment_price - predicted_log_investment_price,
        "consumption_market": log_consumption_revenue
        - predicted_log_consumption_revenue,
        "intermediate_market": log_intermediate_revenue
        - predicted_log_intermediate_revenue,
        "investment_market": log_investment_revenue - predicted_log_investment_revenue,
    }
    static_residuals = jnp.stack(
        tuple(residual_blocks[name] for name in STATIC_RESIDUAL_NAMES), axis=-2
    )
    trade_balance = (intermediate_revenue - intermediate_expenditure) / populations
    factor_income = value_added / populations
    final_expenditure = (consumption_revenue + investment_revenue) / populations

    return {
        **static,
        "consumption": consumption,
        "consumption_output": consumption * populations,
        "investment_rate": investment_rate,
        "investment": investment,
        "investment_output": investment * populations,
        "intermediate_output": intermediate_revenue / static["intermediate_price"],
        "value_added": value_added,
        "intermediate_expenditure": intermediate_expenditure,
        "sector_revenues": sector_revenues,
        "sector_capital": sector_capital,
        "sector_labor": sector_labor,
        "sector_intermediates": sector_intermediates,
        "intermediate_unit_cost": jnp.exp(log_intermediate_unit_cost),
        "trade_shares": trade_shares,
        "trade_balance": trade_balance,
        "factor_income": factor_income,
        "final_expenditure": final_expenditure,
        "world_gdp": jnp.sum(value_added, axis=-1),
        "world_trade_balance": jnp.sum(
            intermediate_revenue - intermediate_expenditure, axis=-1
        ),
        "budget_trade_gap": factor_income - final_expenditure - trade_balance,
        "static_residuals": static_residuals,
        **{f"residual_{name}": value for name, value in residual_blocks.items()},
    }


def static_trade_residuals(
    log_static,
    k,
    tobin_q,
    marginal_wealth,
    parameters,
):
    return static_trade_values(
        log_static,
        k,
        tobin_q,
        marginal_wealth,
        parameters,
    )["static_residuals"]


def static_trade_equilibrium(
    k,
    tobin_q,
    marginal_wealth,
    parameters,
    log_initial=None,
    residual_tol: float | None = None,
    rank_tol: float = 1e-10,
    max_nfev: int = 500,
):
    k = jnp.asarray(k)
    tobin_q = jnp.asarray(tobin_q, dtype=k.dtype)
    marginal_wealth = jnp.asarray(marginal_wealth, dtype=k.dtype)
    expected_shape = (parameters.num_countries,)
    if k.shape != expected_shape:
        raise ValueError("k must contain one value per country")
    if tobin_q.shape != expected_shape or marginal_wealth.shape != expected_shape:
        raise ValueError(
            "tobin_q and marginal_wealth must contain one value per country"
        )
    if bool(jnp.any(k <= 0.0)) or bool(jnp.any(tobin_q <= 0.0)):
        raise ValueError("k and tobin_q must be positive")
    if bool(jnp.any(marginal_wealth <= 0.0)):
        raise ValueError("marginal_wealth must be positive")
    if residual_tol is None:
        residual_tol = 1e-9 if np.dtype(k.dtype).itemsize == 8 else 5e-6
    if residual_tol <= 0.0:
        raise ValueError("residual_tol must be positive")
    static_shape = (len(STATIC_BLOCK_NAMES), parameters.num_countries)
    if log_initial is None:
        log_initial = jnp.zeros(static_shape, dtype=k.dtype)
    else:
        log_initial = jnp.asarray(log_initial, dtype=k.dtype)
        if log_initial.shape != static_shape:
            raise ValueError("log_initial must have shape (8, N)")

    def residual_flat(log_static_flat):
        return static_trade_residuals(
            log_static_flat.reshape(static_shape),
            k,
            tobin_q,
            marginal_wealth,
            parameters,
        ).reshape(-1)

    residual_compiled = jax.jit(residual_flat)
    jacobian_compiled = jax.jit(jax.jacfwd(residual_flat))

    def residual_numpy(log_static_flat):
        return np.asarray(
            residual_compiled(jnp.asarray(log_static_flat, dtype=k.dtype))
        )

    def jacobian_numpy(log_static_flat):
        return np.asarray(
            jacobian_compiled(jnp.asarray(log_static_flat, dtype=k.dtype))
        )

    result = scipy.optimize.least_squares(
        residual_numpy,
        np.asarray(log_initial).reshape(-1),
        jac=jacobian_numpy,
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    log_solution = jnp.asarray(result.x, dtype=k.dtype).reshape(static_shape)
    values = static_trade_values(
        log_solution,
        k,
        tobin_q,
        marginal_wealth,
        parameters,
    )
    residual = np.asarray(values["static_residuals"]).reshape(-1)
    jacobian = jacobian_numpy(result.x)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    max_residual = float(np.max(np.abs(residual)))
    min_singular_value = float(np.min(singular_values))
    if not result.success or max_residual > residual_tol:
        raise RuntimeError(
            "static trade equilibrium failed: "
            f"{result.message}; max residual={max_residual:.3e}"
        )
    if min_singular_value <= rank_tol:
        raise RuntimeError(
            "static trade equilibrium is locally rank deficient: "
            f"minimum singular value={min_singular_value:.3e}"
        )
    return {
        **values,
        "log_static": log_solution,
    }, {
        "max_residual": max_residual,
        "residual_tolerance": residual_tol,
        "dtype": str(k.dtype),
        "minimum_jacobian_singular_value": min_singular_value,
        "jacobian_condition_number": float(
            np.max(singular_values) / min_singular_value
        ),
        "function_evaluations": int(result.nfev),
        "success": bool(result.success),
    }


def multicountry_trade_allocations(
    k,
    tobin_q,
    marginal_wealth,
    interest_rate,
    log_static,
    parameters,
):
    values = static_trade_values(
        log_static,
        k,
        tobin_q,
        marginal_wealth,
        parameters,
    )
    installation = (
        parameters.installation_efficiency
        * values["investment_rate"] ** parameters.installation_elasticity
    )
    capital_drift = k * (installation - parameters.delta)
    rate = jnp.asarray(interest_rate, dtype=k.dtype)[..., None]
    tobin_q_drift = (
        rate
        + parameters.delta
        - (1.0 - parameters.installation_elasticity) * installation
    ) * tobin_q - values["rental_rate"]
    return {
        **values,
        "capital_drift": capital_drift,
        "tobin_q_drift": tobin_q_drift,
        "marginal_wealth_drift": marginal_wealth * (parameters.rho - rate),
    }


def world_asset_clearing(a, parameters, scale=1.0):
    populations = jnp.asarray(parameters.populations, dtype=a.dtype)
    return jnp.sum(populations * a, axis=-1) / scale


def consumption_basket_log_price(log_static, parameters):
    populations = jnp.asarray(parameters.populations, dtype=log_static.dtype)
    weights = populations / jnp.sum(populations)
    return jnp.sum(weights * log_static[..., 2, :], axis=-1)


def multicountry_trade_path_values(
    k,
    a,
    tobin_q,
    marginal_wealth,
    interest_rate,
    log_static,
    dlog_k_dt,
    da_dt,
    dlog_q_dt,
    dlog_marginal_wealth_dt,
    parameters,
):
    allocations = multicountry_trade_allocations(
        k,
        tobin_q,
        marginal_wealth,
        interest_rate,
        log_static,
        parameters,
    )
    populations = jnp.asarray(parameters.populations, dtype=k.dtype)
    income_scale = jnp.sum(
        populations * allocations["factor_income"], axis=-1
    ) / jnp.sum(populations)
    budget_drift = (
        interest_rate[:, None] * a
        + allocations["factor_income"]
        - allocations["final_expenditure"]
    )
    world_flow = jnp.sum(
        populations * (allocations["factor_income"] - allocations["final_expenditure"]),
        axis=-1,
    )
    return {
        **allocations,
        "k": k,
        "a": a,
        "q": tobin_q,
        "marginal_wealth": marginal_wealth,
        "interest_rate": interest_rate,
        "current_account": da_dt,
        "capital": dlog_k_dt - allocations["capital_drift"] / k,
        "assets": (da_dt - budget_drift) / income_scale[:, None],
        "tobin_q": dlog_q_dt - allocations["tobin_q_drift"] / tobin_q,
        "marginal_wealth_euler": dlog_marginal_wealth_dt
        - (parameters.rho - interest_rate[:, None]),
        "world_flow": world_flow / income_scale,
        "asset_clearing": world_asset_clearing(a, parameters, income_scale),
        "interest_rate_country_gap": parameters.rho
        - dlog_marginal_wealth_dt
        - interest_rate[:, None],
    }


def multicountry_trade_dae_residual(
    values,
    log_static,
    parameters,
):
    dynamic_residual = jnp.concatenate(
        [values[name].reshape(-1) for name in DYNAMIC_RESIDUAL_NAMES]
    )
    static_residual = values["static_residuals"].reshape(-1)
    numeraire_residual = consumption_basket_log_price(log_static, parameters)
    return jnp.concatenate(
        [
            dynamic_residual
            / jnp.sqrt(
                jnp.asarray(dynamic_residual.size, dtype=dynamic_residual.dtype)
            ),
            static_residual
            / jnp.sqrt(jnp.asarray(static_residual.size, dtype=static_residual.dtype)),
            numeraire_residual
            / jnp.sqrt(
                jnp.asarray(numeraire_residual.size, dtype=numeraire_residual.dtype)
            ),
        ]
    )


def multicountry_trade_max_dae_residual(
    values,
    log_static,
    parameters,
    time_selection=None,
):
    def select(value):
        return value if time_selection is None else value[time_selection]

    residuals = [
        *[select(values[name]).reshape(-1) for name in DYNAMIC_RESIDUAL_NAMES],
        select(values["static_residuals"]).reshape(-1),
        select(consumption_basket_log_price(log_static, parameters)).reshape(-1),
    ]
    return jnp.max(jnp.abs(jnp.concatenate(residuals)))


def multicountry_trade_pointwise_residual_norms(
    values,
    log_static,
    parameters,
):
    num_points = values[DYNAMIC_RESIDUAL_NAMES[0]].shape[0]
    ode_residual = jnp.concatenate(
        [values[name].reshape(num_points, -1) for name in DYNAMIC_RESIDUAL_NAMES],
        axis=1,
    )
    algebraic_residual = jnp.concatenate(
        [
            values["static_residuals"].reshape(num_points, -1),
            consumption_basket_log_price(log_static, parameters)[:, None],
        ],
        axis=1,
    )
    ode_sum_squares = jnp.sum(jnp.square(ode_residual), axis=1)
    algebraic_sum_squares = jnp.sum(jnp.square(algebraic_residual), axis=1)
    return {
        "ode_residual_rms": jnp.sqrt(ode_sum_squares / ode_residual.shape[1]),
        "algebraic_residual_rms": jnp.sqrt(
            algebraic_sum_squares / algebraic_residual.shape[1]
        ),
        "dae_residual_rms": jnp.sqrt(
            (ode_sum_squares + algebraic_sum_squares)
            / (ode_residual.shape[1] + algebraic_residual.shape[1])
        ),
    }


def multicountry_trade_tvc_diagnostics(t, values, parameters):
    t = jnp.asarray(t)
    interest_rate = values["interest_rate"]
    increments = 0.5 * (interest_rate[1:] + interest_rate[:-1]) * jnp.diff(t)
    integrated_rate = jnp.concatenate(
        [jnp.zeros(1, dtype=t.dtype), jnp.cumsum(increments)]
    )
    bond_discount = jnp.exp(-integrated_rate)
    utility_discount = jnp.exp(-parameters.rho * t)
    marginal_wealth = values["marginal_wealth"]
    return {
        "discounted_assets": bond_discount[:, None] * values["a"],
        "asset_tvc": utility_discount[:, None] * marginal_wealth * values["a"],
        "capital_tvc": utility_discount[:, None]
        * marginal_wealth
        * values["q"]
        * values["k"],
    }
