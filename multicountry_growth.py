from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class MulticountryParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    rho: float = -math.log(0.96)
    installation_elasticity: float = 0.76
    populations: tuple[float, ...] = (1.0, 1.0)
    productivities: tuple[float, ...] = (1.0, 1.0)

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.delta <= 0.0 or self.rho <= 0.0:
            raise ValueError("delta and rho must be positive")
        if not 0.0 < self.installation_elasticity < 1.0:
            raise ValueError("installation_elasticity must lie in (0, 1)")
        if len(self.populations) < 2:
            raise ValueError("at least two countries are required")
        if len(self.populations) != len(self.productivities):
            raise ValueError("populations and productivities must have equal lengths")
        if any(value <= 0.0 for value in self.populations):
            raise ValueError("populations must be positive")
        if any(value <= 0.0 for value in self.productivities):
            raise ValueError("productivities must be positive")

    @property
    def num_countries(self):
        return len(self.populations)

    @property
    def installation_efficiency(self):
        return self.delta ** (1.0 - self.installation_elasticity)


def production(k, z, parameters):
    return z * k**parameters.alpha


def marginal_product(k, z, parameters):
    return parameters.alpha * z * k ** (parameters.alpha - 1.0)


def investment_rate_from_tobin_q(tobin_q, parameters):
    elasticity = parameters.installation_elasticity
    return (parameters.installation_efficiency * elasticity * tobin_q) ** (
        1.0 / (1.0 - elasticity)
    )


def consumption_from_marginal_wealth(marginal_wealth, parameters):
    return marginal_wealth ** (-1.0 / parameters.gamma)


def relative_welfare_weights(marginal_wealth):
    return marginal_wealth[..., :1] / marginal_wealth


def multicountry_allocations(
    k,
    tobin_q,
    marginal_wealth,
    interest_rate,
    parameters,
):
    productivities = jnp.asarray(parameters.productivities, dtype=k.dtype)
    investment_rate = investment_rate_from_tobin_q(tobin_q, parameters)
    investment = k * investment_rate
    output = production(k, productivities, parameters)
    mpk = marginal_product(k, productivities, parameters)
    resources = output - investment
    consumption = consumption_from_marginal_wealth(marginal_wealth, parameters)
    capital_drift = k * (
        parameters.installation_efficiency
        * investment_rate**parameters.installation_elasticity
        - parameters.delta
    )
    q_intercept = (
        parameters.delta
        - (1.0 - parameters.installation_elasticity)
        * parameters.installation_efficiency
        * investment_rate**parameters.installation_elasticity
        - mpk / tobin_q
    )
    rate = jnp.asarray(interest_rate, dtype=k.dtype)[..., None]
    return {
        "investment_rate": investment_rate,
        "investment": investment,
        "output": output,
        "mpk": mpk,
        "resources": resources,
        "consumption": consumption,
        "capital_drift": capital_drift,
        "q_intercept": q_intercept,
        "tobin_q_drift": tobin_q * (rate + q_intercept),
        "marginal_wealth_drift": marginal_wealth * (parameters.rho - rate),
    }


def multicountry_reference_marginal_wealth(parameters, dtype=jnp.float64):
    productivities = jnp.asarray(parameters.productivities, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    mpk = parameters.delta + parameters.rho / parameters.installation_elasticity
    k = (parameters.alpha * productivities / mpk) ** (1.0 / (1.0 - parameters.alpha))
    investment = parameters.delta * k
    resources = production(k, productivities, parameters) - investment
    consumption = jnp.sum(populations * resources) / jnp.sum(populations)
    return jnp.full(
        parameters.num_countries,
        consumption ** (-parameters.gamma),
        dtype=dtype,
    )


def multicountry_steady_state(marginal_wealth, parameters):
    marginal_wealth = jnp.atleast_1d(marginal_wealth)
    if marginal_wealth.shape[-1] != parameters.num_countries:
        raise ValueError("marginal_wealth must contain one value per country")
    dtype = marginal_wealth.dtype
    productivities = jnp.asarray(parameters.productivities, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    tobin_q = jnp.full(
        parameters.num_countries,
        1.0 / parameters.installation_elasticity,
        dtype=dtype,
    )
    mpk = parameters.delta + parameters.rho / parameters.installation_elasticity
    k = (parameters.alpha * productivities / mpk) ** (1.0 / (1.0 - parameters.alpha))
    allocations = multicountry_allocations(
        k,
        tobin_q,
        marginal_wealth,
        jnp.asarray(parameters.rho, dtype=dtype),
        parameters,
    )
    assets = (allocations["consumption"] - allocations["resources"]) / parameters.rho
    return {
        "k": k,
        "q": tobin_q,
        "a": assets,
        "c": allocations["consumption"],
        "x": allocations["investment"],
        "output": allocations["output"],
        "resources": allocations["resources"],
        "marginal_wealth": marginal_wealth,
        "relative_welfare_weights": relative_welfare_weights(marginal_wealth),
        "interest_rate": jnp.asarray(parameters.rho, dtype=dtype),
        "resource_gap": jnp.sum(
            populations * (allocations["resources"] - allocations["consumption"])
        ),
        "asset_clearing_gap": jnp.sum(populations * assets),
    }


def multicountry_reference_steady_state(parameters, dtype=jnp.float64):
    return multicountry_steady_state(
        multicountry_reference_marginal_wealth(parameters, dtype),
        parameters,
    )


def multicountry_output_scale(parameters, dtype=jnp.float64):
    steady = multicountry_reference_steady_state(parameters, dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    return jnp.sum(populations * steady["output"]) / jnp.sum(populations)


def multicountry_path_values(
    k,
    a,
    tobin_q,
    marginal_wealth,
    interest_rate,
    dlog_k_dt,
    da_dt,
    dlog_q_dt,
    dlog_marginal_wealth_dt,
    parameters,
):
    allocations = multicountry_allocations(
        k,
        tobin_q,
        marginal_wealth,
        interest_rate,
        parameters,
    )
    populations = jnp.asarray(parameters.populations, dtype=k.dtype)
    flow_scale = multicountry_output_scale(parameters, k.dtype)
    budget_drift = (
        interest_rate[:, None] * a
        + allocations["resources"]
        - allocations["consumption"]
    )
    residuals = {
        "capital": dlog_k_dt - allocations["capital_drift"] / k,
        "assets": (da_dt - budget_drift) / flow_scale,
        "tobin_q": dlog_q_dt - allocations["tobin_q_drift"] / tobin_q,
        "marginal_wealth_euler": dlog_marginal_wealth_dt
        - (parameters.rho - interest_rate[:, None]),
        "resource": jnp.sum(
            populations * (allocations["resources"] - allocations["consumption"]),
            axis=-1,
        )
        / flow_scale,
        "resource_drift": jnp.sum(
            populations
            * (
                (allocations["mpk"] * k - allocations["investment"]) * dlog_k_dt
                - allocations["investment"]
                * dlog_q_dt
                / (1.0 - parameters.installation_elasticity)
                + allocations["consumption"]
                * dlog_marginal_wealth_dt
                / parameters.gamma
            ),
            axis=-1,
        )
        / flow_scale,
        "asset_clearing": jnp.sum(populations * a, axis=-1) / flow_scale,
        "interest_rate_country_gap": parameters.rho
        - dlog_marginal_wealth_dt
        - interest_rate[:, None],
    }
    return {
        "k": k,
        "a": a,
        "q": tobin_q,
        "c": allocations["consumption"],
        "x": allocations["investment"],
        "output": allocations["output"],
        "resources": allocations["resources"],
        "current_account": da_dt,
        "interest_rate": interest_rate,
        "marginal_wealth": marginal_wealth,
        "relative_welfare_weights": relative_welfare_weights(marginal_wealth),
        **residuals,
    }


def multicountry_dae_path_values(
    k,
    a,
    tobin_q,
    marginal_wealth,
    consumption,
    investment_rate,
    interest_rate,
    dlog_k_dt,
    da_dt,
    dlog_q_dt,
    dlog_marginal_wealth_dt,
    dlog_consumption_dt,
    dlog_investment_rate_dt,
    parameters,
):
    productivities = jnp.asarray(parameters.productivities, dtype=k.dtype)
    populations = jnp.asarray(parameters.populations, dtype=k.dtype)
    flow_scale = multicountry_output_scale(parameters, k.dtype)
    rate = interest_rate[:, None]
    investment = k * investment_rate
    output = production(k, productivities, parameters)
    mpk = marginal_product(k, productivities, parameters)
    resources = output - investment
    capital_drift = k * (
        parameters.installation_efficiency
        * investment_rate**parameters.installation_elasticity
        - parameters.delta
    )
    budget_drift = rate * a + resources - consumption
    q_drift = (
        tobin_q
        * (
            rate
            + parameters.delta
            - (1.0 - parameters.installation_elasticity)
            * parameters.installation_efficiency
            * investment_rate**parameters.installation_elasticity
        )
        - mpk
    )
    resource = (
        jnp.sum(
            populations * (resources - consumption),
            axis=-1,
        )
        / flow_scale
    )
    resource_drift = (
        jnp.sum(
            populations
            * (
                (mpk * k - investment) * dlog_k_dt
                - investment * dlog_investment_rate_dt
                - consumption * dlog_consumption_dt
            ),
            axis=-1,
        )
        / flow_scale
    )
    return {
        "k": k,
        "a": a,
        "q": tobin_q,
        "c": consumption,
        "s": investment_rate,
        "x": investment,
        "output": output,
        "resources": resources,
        "current_account": da_dt,
        "interest_rate": interest_rate,
        "marginal_wealth": marginal_wealth,
        "relative_welfare_weights": relative_welfare_weights(marginal_wealth),
        "capital": dlog_k_dt - capital_drift / k,
        "assets": (da_dt - budget_drift) / flow_scale,
        "tobin_q": dlog_q_dt - q_drift / tobin_q,
        "marginal_wealth_euler": dlog_marginal_wealth_dt - (parameters.rho - rate),
        "consumption_foc": marginal_wealth * consumption**parameters.gamma - 1.0,
        "investment_foc": parameters.installation_efficiency
        * parameters.installation_elasticity
        * tobin_q
        * investment_rate ** (parameters.installation_elasticity - 1.0)
        - 1.0,
        "resource": resource,
        "resource_drift": resource_drift,
        "asset_clearing": jnp.sum(populations * a, axis=-1) / flow_scale,
    }


def multicountry_tvc_diagnostics(t, values, parameters):
    discount = jnp.exp(-jnp.trapezoid(values["interest_rate"], t))
    utility_discount = jnp.exp(-parameters.rho * t[-1])
    terminal_assets = values["a"][-1]
    terminal_consumption = values["c"][-1]
    terminal_q = values["q"][-1]
    terminal_k = values["k"][-1]
    marginal_utility = terminal_consumption ** (-parameters.gamma)
    return {
        "discounted_assets": jnp.abs(discount * terminal_assets),
        "asset_tvc": jnp.abs(utility_discount * marginal_utility * terminal_assets),
        "capital_tvc": jnp.abs(
            utility_discount * marginal_utility * terminal_q * terminal_k
        ),
    }


def multicountry_tail_diagnostics(values, parameters):
    steady = multicountry_steady_state(values["marginal_wealth"][-1], parameters)
    scale = multicountry_output_scale(parameters, values["k"].dtype)
    return {
        "k_log_gap": jnp.abs(jnp.log(values["k"][-1] / steady["k"])),
        "q_log_gap": jnp.abs(jnp.log(values["q"][-1] / steady["q"])),
        "asset_scaled_gap": jnp.abs(values["a"][-1] - steady["a"]) / scale,
        "marginal_wealth_drift": jnp.abs(parameters.rho - values["interest_rate"][-1]),
        "interest_rate_gap": jnp.abs(values["interest_rate"][-1] - parameters.rho),
    }
