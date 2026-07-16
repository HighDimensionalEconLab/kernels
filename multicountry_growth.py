from __future__ import annotations

import math
from dataclasses import dataclass

import jax
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


def relative_consumption_weights(log_relative_weights, parameters):
    log_relative_weights = jnp.atleast_1d(log_relative_weights)
    if log_relative_weights.shape[-1] != parameters.num_countries - 1:
        raise ValueError("one relative weight is required for each country after the first")
    log_weights = jnp.concatenate(
        [jnp.zeros(1, dtype=log_relative_weights.dtype), log_relative_weights]
    )
    weights = jnp.exp(log_weights)
    consumption_weights = jnp.exp(log_weights / parameters.gamma)
    return weights, consumption_weights


def multicountry_allocations(k, tobin_q, log_relative_weights, parameters):
    populations = jnp.asarray(parameters.populations, dtype=k.dtype)
    productivities = jnp.asarray(parameters.productivities, dtype=k.dtype)
    weights, consumption_weights = relative_consumption_weights(
        log_relative_weights, parameters
    )
    investment_rate = investment_rate_from_tobin_q(tobin_q, parameters)
    investment = k * investment_rate
    output = production(k, productivities, parameters)
    mpk = marginal_product(k, productivities, parameters)
    resources = output - investment
    aggregate_resources = jnp.sum(populations * resources)
    weight_sum = jnp.sum(populations * consumption_weights)
    consumption = consumption_weights * aggregate_resources / weight_sum
    marginal_wealth = (weight_sum / aggregate_resources) ** parameters.gamma

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
    rate_denominator = aggregate_resources / parameters.gamma + jnp.sum(
        populations
        * investment
        / (1.0 - parameters.installation_elasticity)
    )
    rate_numerator = (
        jnp.sum(populations * (mpk - investment_rate) * capital_drift)
        + parameters.rho * aggregate_resources / parameters.gamma
        - jnp.sum(
            populations
            * investment
            * q_intercept
            / (1.0 - parameters.installation_elasticity)
        )
    )
    interest_rate = rate_numerator / rate_denominator
    tobin_q_drift = tobin_q * (interest_rate + q_intercept)
    resource_drift = jnp.sum(
        populations
        * (
            (mpk - investment_rate) * capital_drift
            - investment
            * (interest_rate + q_intercept)
            / (1.0 - parameters.installation_elasticity)
        )
    )
    interest_rate_from_resources = (
        parameters.rho
        + parameters.gamma * resource_drift / aggregate_resources
    )
    return {
        "weights": weights,
        "consumption_weights": consumption_weights,
        "investment_rate": investment_rate,
        "investment": investment,
        "output": output,
        "mpk": mpk,
        "resources": resources,
        "aggregate_resources": aggregate_resources,
        "weight_sum": weight_sum,
        "consumption": consumption,
        "marginal_wealth": marginal_wealth,
        "capital_drift": capital_drift,
        "q_intercept": q_intercept,
        "interest_rate": interest_rate,
        "tobin_q_drift": tobin_q_drift,
        "resource_drift": resource_drift,
        "interest_rate_from_resources": interest_rate_from_resources,
        "rate_denominator": rate_denominator,
    }


def reconstruct_assets(a_independent, parameters):
    populations = jnp.asarray(parameters.populations, dtype=a_independent.dtype)
    final_asset = -jnp.sum(
        populations[:-1] * a_independent, axis=-1, keepdims=True
    ) / populations[-1]
    return jnp.concatenate([a_independent, final_asset], axis=-1)


def reconstruct_asset_drifts(da_independent_dt, parameters):
    populations = jnp.asarray(
        parameters.populations, dtype=da_independent_dt.dtype
    )
    final_drift = -jnp.sum(
        populations[:-1] * da_independent_dt, axis=-1, keepdims=True
    ) / populations[-1]
    return jnp.concatenate([da_independent_dt, final_drift], axis=-1)


def multicountry_steady_state(log_relative_weights, parameters):
    dtype = jnp.asarray(log_relative_weights).dtype
    productivities = jnp.asarray(parameters.productivities, dtype=dtype)
    tobin_q = jnp.full(
        parameters.num_countries,
        1.0 / parameters.installation_elasticity,
        dtype=dtype,
    )
    mpk_ss = parameters.delta + parameters.rho / parameters.installation_elasticity
    k = (parameters.alpha * productivities / mpk_ss) ** (
        1.0 / (1.0 - parameters.alpha)
    )
    allocations = multicountry_allocations(
        k, tobin_q, log_relative_weights, parameters
    )
    assets = (
        allocations["consumption"] - allocations["resources"]
    ) / parameters.rho
    return {
        "k": k,
        "q": tobin_q,
        "a": assets,
        "c": allocations["consumption"],
        "x": allocations["investment"],
        "output": allocations["output"],
        "resources": allocations["resources"],
        "marginal_wealth": allocations["marginal_wealth"],
        "interest_rate": jnp.asarray(parameters.rho, dtype=dtype),
    }


def multicountry_output_scale(parameters, dtype=jnp.float64):
    steady = multicountry_steady_state(
        jnp.zeros(parameters.num_countries - 1, dtype=dtype), parameters
    )
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    return jnp.sum(populations * steady["output"]) / jnp.sum(populations)


def multicountry_path_values(
    k,
    a_independent,
    tobin_q,
    log_relative_weights,
    dlog_k_dt,
    da_independent_dt,
    dlog_q_dt,
    parameters,
):
    allocations = jax.vmap(
        lambda k_t, q_t: multicountry_allocations(
            k_t, q_t, log_relative_weights, parameters
        )
    )(k, tobin_q)
    assets = reconstruct_assets(a_independent, parameters)
    asset_drifts = reconstruct_asset_drifts(da_independent_dt, parameters)
    budget_drift = (
        allocations["interest_rate"][:, None] * assets
        + allocations["resources"]
        - allocations["consumption"]
    )
    flow_scale = multicountry_output_scale(parameters, k.dtype)
    residuals = {
        "capital": dlog_k_dt
        - allocations["capital_drift"] / k,
        "tobin_q": dlog_q_dt
        - allocations["tobin_q_drift"] / tobin_q,
        "assets": (
            da_independent_dt - budget_drift[:, :-1]
        ) / flow_scale,
        "omitted_asset": (asset_drifts[:, -1] - budget_drift[:, -1])
        / flow_scale,
        "resource": jnp.sum(
            jnp.asarray(parameters.populations, dtype=k.dtype)
            * (allocations["resources"] - allocations["consumption"]),
            axis=-1,
        )
        / flow_scale,
        "asset_clearing": jnp.sum(
            jnp.asarray(parameters.populations, dtype=k.dtype) * assets,
            axis=-1,
        )
        / flow_scale,
        "interest_rate_index_gap": allocations["interest_rate"]
        - allocations["interest_rate_from_resources"],
    }
    return {
        "k": k,
        "a": assets,
        "q": tobin_q,
        "c": allocations["consumption"],
        "x": allocations["investment"],
        "output": allocations["output"],
        "resources": allocations["resources"],
        "current_account": asset_drifts,
        "interest_rate": allocations["interest_rate"],
        "marginal_wealth": allocations["marginal_wealth"],
        "rate_denominator": allocations["rate_denominator"],
        **residuals,
    }


def two_country_transformed_drift(state, log_relative_weight, parameters):
    if parameters.num_countries != 2:
        raise ValueError("the reduced transformed system requires two countries")
    steady = multicountry_steady_state(
        jnp.atleast_1d(log_relative_weight), parameters
    )
    scale = multicountry_output_scale(parameters, state.dtype)
    k = steady["k"] * jnp.exp(state[:2])
    tobin_q = steady["q"] * jnp.exp(state[2:4])
    a_1 = scale * state[4]
    allocations = multicountry_allocations(
        k, tobin_q, jnp.atleast_1d(log_relative_weight), parameters
    )
    budget_1 = (
        allocations["interest_rate"] * a_1
        + allocations["resources"][0]
        - allocations["consumption"][0]
    )
    return jnp.concatenate(
        [
            allocations["capital_drift"] / k,
            allocations["tobin_q_drift"] / tobin_q,
            jnp.atleast_1d(budget_1 / scale),
        ]
    )


def two_country_transformed_steady_state(log_relative_weight, parameters):
    steady = multicountry_steady_state(
        jnp.atleast_1d(log_relative_weight), parameters
    )
    scale = multicountry_output_scale(parameters, steady["k"].dtype)
    return jnp.concatenate([jnp.zeros(4, dtype=scale.dtype), steady["a"][:1] / scale])


def two_country_full_linearization(log_relative_weight, parameters):
    steady_state = two_country_transformed_steady_state(
        log_relative_weight, parameters
    )
    full_state = jnp.concatenate(
        [steady_state, jnp.atleast_1d(log_relative_weight)]
    )

    def full_drift(state):
        return jnp.concatenate(
            [
                two_country_transformed_drift(state[:5], state[5], parameters),
                jnp.zeros(1, dtype=state.dtype),
            ]
        )

    return jax.jacfwd(full_drift)(full_state)


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
        "asset_tvc": jnp.abs(
            utility_discount * marginal_utility * terminal_assets
        ),
        "capital_tvc": jnp.abs(
            utility_discount * marginal_utility * terminal_q * terminal_k
        ),
    }


def multicountry_tail_diagnostics(values, log_relative_weights, parameters):
    steady = multicountry_steady_state(log_relative_weights, parameters)
    scale = multicountry_output_scale(parameters, values["k"].dtype)
    return {
        "k_log_gap": jnp.abs(jnp.log(values["k"][-1] / steady["k"])),
        "q_log_gap": jnp.abs(jnp.log(values["q"][-1] / steady["q"])),
        "asset_scaled_gap": jnp.abs(values["a"][-1] - steady["a"]) / scale,
        "interest_rate_gap": jnp.abs(
            values["interest_rate"][-1] - parameters.rho
        ),
    }
