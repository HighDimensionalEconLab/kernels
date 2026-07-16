from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ModelParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    rho: float = -math.log(0.96)
    installation_elasticity: float = 0.76
    world_interest_rate: float | None = None

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if not 0.0 < self.installation_elasticity < 1.0:
            raise ValueError("installation_elasticity must lie in (0, 1)")
        if self.interest_rate <= 0.0:
            raise ValueError("world_interest_rate must be positive")

    @property
    def interest_rate(self):
        if self.world_interest_rate is None:
            return self.rho
        return self.world_interest_rate

    @property
    def installation_efficiency(self):
        return self.delta ** (1.0 - self.installation_elasticity)


def production(k, z, parameters):
    return z * k**parameters.alpha


def marginal_product(k, z, parameters):
    return parameters.alpha * z * k ** (parameters.alpha - 1.0)


def investment_rate_from_tobin_q(tobin_q, parameters):
    installation_elasticity = parameters.installation_elasticity
    return (parameters.installation_efficiency * installation_elasticity * tobin_q) ** (
        1.0 / (1.0 - installation_elasticity)
    )


def investment_expenditure(k, tobin_q, parameters):
    return k * investment_rate_from_tobin_q(tobin_q, parameters)


def consumption_from_marginal_wealth(marginal_wealth, parameters):
    return marginal_wealth ** (-1.0 / parameters.gamma)


def capital_drift(k, tobin_q, parameters):
    investment_rate = investment_rate_from_tobin_q(tobin_q, parameters)
    return k * (
        parameters.installation_efficiency
        * investment_rate**parameters.installation_elasticity
        - parameters.delta
    )


def current_account(k, a, tobin_q, c, z, parameters):
    return (
        parameters.interest_rate * a
        + production(k, z, parameters)
        - c
        - investment_expenditure(k, tobin_q, parameters)
    )


def tobin_q_drift(k, tobin_q, z, parameters):
    installation_elasticity = parameters.installation_elasticity
    investment_rate = investment_rate_from_tobin_q(tobin_q, parameters)
    return (
        parameters.interest_rate
        + parameters.delta
        - (1.0 - installation_elasticity)
        * parameters.installation_efficiency
        * investment_rate**installation_elasticity
    ) * tobin_q - marginal_product(k, z, parameters)


def marginal_wealth_drift(marginal_wealth, parameters):
    return (parameters.rho - parameters.interest_rate) * marginal_wealth


def steady_state(parameters):
    installation_elasticity = parameters.installation_elasticity
    investment_rate = parameters.delta
    tobin_q = 1.0 / installation_elasticity
    marginal_product_ss = (
        parameters.delta + parameters.interest_rate / installation_elasticity
    )
    k = (parameters.alpha / marginal_product_ss) ** (1.0 / (1.0 - parameters.alpha))
    output = production(k, 1.0, parameters)
    investment = parameters.delta * k
    domestic_resources = output - investment
    return {
        "k": k,
        "q": tobin_q,
        "s": investment_rate,
        "x": investment,
        "output": output,
        "g": domestic_resources,
        "marginal_product": marginal_product_ss,
    }


def stable_tail_terms(parameters):
    steady = steady_state(parameters)
    installation_elasticity = parameters.installation_elasticity
    capital_loading = (
        parameters.delta * installation_elasticity / (1.0 - installation_elasticity)
    )
    q_loading = (1.0 - parameters.alpha) * (
        parameters.interest_rate + installation_elasticity * parameters.delta
    )
    discriminant = parameters.interest_rate**2 + 4.0 * capital_loading * q_loading
    stable_root = 0.5 * (parameters.interest_rate - jnp.sqrt(discriminant))
    unstable_root = 0.5 * (parameters.interest_rate + jnp.sqrt(discriminant))
    stable_slope = stable_root / capital_loading
    resource_loading = (
        parameters.alpha * steady["output"]
        - parameters.delta * steady["k"]
        - parameters.delta
        * steady["k"]
        * stable_slope
        / (1.0 - installation_elasticity)
    )
    return {
        "capital_loading": capital_loading,
        "q_loading": q_loading,
        "stable_root": stable_root,
        "unstable_root": unstable_root,
        "stable_slope": stable_slope,
        "resource_loading": resource_loading,
    }


def path_residuals(k, a, tobin_q, c, z, dlog_k_dt, da_dt, dlog_q_dt, parameters):
    investment_rate = investment_rate_from_tobin_q(tobin_q, parameters)
    return {
        "capital": dlog_k_dt
        - (
            parameters.installation_efficiency
            * investment_rate**parameters.installation_elasticity
            - parameters.delta
        ),
        "assets": (da_dt - current_account(k, a, tobin_q, c, z, parameters))
        / steady_state(parameters)["output"],
        "tobin_q": dlog_q_dt - tobin_q_drift(k, tobin_q, z, parameters) / tobin_q,
    }


def tail_diagnostics(k, a, tobin_q, c, parameters):
    steady = steady_state(parameters)
    tail = stable_tail_terms(parameters)
    h = jnp.log(k / steady["k"])
    v = jnp.log(tobin_q / steady["q"])
    stable = v - tail["stable_slope"] * h
    present_value = (
        c
        - parameters.interest_rate * a
        - steady["g"]
        - parameters.interest_rate
        * tail["resource_loading"]
        * h
        / (parameters.interest_rate - tail["stable_root"])
    ) / steady["output"]
    return {"stable": stable, "present_value": present_value}


def present_value_gap(t, k, tobin_q, c, a_0, parameters):
    steady = steady_state(parameters)
    tail = stable_tail_terms(parameters)
    investment = investment_expenditure(k, tobin_q, parameters)
    domestic_resources = production(k, 1.0, parameters) - investment
    discount = jnp.exp(-parameters.interest_rate * t)
    h_terminal = jnp.log(k[-1] / steady["k"])
    continuation = steady["g"] / parameters.interest_rate + tail[
        "resource_loading"
    ] * h_terminal / (parameters.interest_rate - tail["stable_root"])
    wealth = (
        a_0
        + jnp.trapezoid(discount * domestic_resources, t)
        + discount[-1] * continuation
    )
    return c - parameters.interest_rate * wealth


def tvc_diagnostics(k, a, tobin_q, c, t, parameters):
    marginal_wealth = c ** (-parameters.gamma)
    discount_assets = jnp.exp(-parameters.interest_rate * t) * a
    utility_discount = jnp.exp(-parameters.rho * t)
    return {
        "discounted_assets": jnp.abs(discount_assets),
        "asset_tvc": jnp.abs(utility_discount * marginal_wealth * a),
        "capital_tvc": jnp.abs(utility_discount * marginal_wealth * tobin_q * k),
    }
