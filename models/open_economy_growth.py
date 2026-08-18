from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ModelParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    r: float = -math.log(0.96)
    eta: float = 0.76
    r_star: float | None = None

    @property
    def interest_rate(self):
        if self.r_star is None:
            return self.r
        return self.r_star

    @property
    def chi(self):
        return self.delta ** (1.0 - self.eta)


def production(k, z, parameters):
    return z * k**parameters.alpha


def marginal_product(k, z, parameters):
    return parameters.alpha * z * k ** (parameters.alpha - 1.0)


def investment_rate_from_q(q, parameters):
    eta = parameters.eta
    return (parameters.chi * eta * q) ** (1.0 / (1.0 - eta))


def investment_expenditure(k, q, parameters):
    return k * investment_rate_from_q(q, parameters)


def consumption_from_mu(mu, parameters):
    return mu ** (-1.0 / parameters.gamma)


def capital_drift(k, q, parameters):
    investment_rate = investment_rate_from_q(q, parameters)
    return k * (parameters.chi * investment_rate**parameters.eta - parameters.delta)


def current_account(k, a, q, c, z, parameters):
    return (
        parameters.interest_rate * a
        + production(k, z, parameters)
        - c
        - investment_expenditure(k, q, parameters)
    )


def q_drift(k, q, z, parameters):
    eta = parameters.eta
    investment_rate = investment_rate_from_q(q, parameters)
    return (
        parameters.interest_rate
        + parameters.delta
        - (1.0 - eta) * parameters.chi * investment_rate**eta
    ) * q - marginal_product(k, z, parameters)


def mu_drift(mu, parameters):
    return (parameters.r - parameters.interest_rate) * mu


def consumption_growth_rate(parameters):
    return (parameters.interest_rate - parameters.r) / parameters.gamma


def consumption_annuity_rate(parameters):
    return parameters.interest_rate - consumption_growth_rate(parameters)


def steady_state(parameters):
    eta = parameters.eta
    investment_rate = parameters.delta
    q = 1.0 / eta
    marginal_product_ss = parameters.delta + parameters.interest_rate / eta
    k = (parameters.alpha / marginal_product_ss) ** (1.0 / (1.0 - parameters.alpha))
    output = production(k, 1.0, parameters)
    investment = parameters.delta * k
    domestic_resources = output - investment
    return {
        "k": k,
        "q": q,
        "s": investment_rate,
        "x": investment,
        "output": output,
        "g": domestic_resources,
        "marginal_product": marginal_product_ss,
    }


def stable_tail_terms(parameters):
    steady = steady_state(parameters)
    eta = parameters.eta
    capital_loading = parameters.delta * eta / (1.0 - eta)
    q_loading = (1.0 - parameters.alpha) * (
        parameters.interest_rate + eta * parameters.delta
    )
    discriminant = parameters.interest_rate**2 + 4.0 * capital_loading * q_loading
    stable_root = 0.5 * (parameters.interest_rate - jnp.sqrt(discriminant))
    unstable_root = 0.5 * (parameters.interest_rate + jnp.sqrt(discriminant))
    stable_slope = stable_root / capital_loading
    resource_loading = (
        parameters.alpha * steady["output"]
        - parameters.delta * steady["k"]
        - parameters.delta * steady["k"] * stable_slope / (1.0 - eta)
    )
    return {
        "capital_loading": capital_loading,
        "q_loading": q_loading,
        "stable_root": stable_root,
        "unstable_root": unstable_root,
        "stable_slope": stable_slope,
        "resource_loading": resource_loading,
    }


def path_residuals(
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
):
    investment_rate = investment_rate_from_q(q, parameters)
    return {
        "capital": dlog_k_dt
        - (parameters.chi * investment_rate**parameters.eta - parameters.delta),
        "assets": da_dt - current_account(k, a, q, c, z, parameters),
        "tobin_q": dlog_q_dt - q_drift(k, q, z, parameters) / q,
        "marginal_wealth": dlog_mu_dt - mu_drift(mu, parameters) / mu,
        "consumption_foc": mu * c**parameters.gamma - 1.0,
    }


def tail_diagnostics(k, a, q, c, parameters):
    steady = steady_state(parameters)
    tail = stable_tail_terms(parameters)
    h = jnp.log(k / steady["k"])
    v = jnp.log(q / steady["q"])
    stable = v - tail["stable_slope"] * h
    tail_resources = steady["g"] / parameters.interest_rate + tail[
        "resource_loading"
    ] * h / (parameters.interest_rate - tail["stable_root"])
    present_value = (
        c - consumption_annuity_rate(parameters) * (a + tail_resources)
    ) / steady["output"]
    return {"stable": stable, "present_value": present_value}


def present_value_gap(t, k, q, c, a_0, parameters):
    steady = steady_state(parameters)
    tail = stable_tail_terms(parameters)
    investment = investment_expenditure(k, q, parameters)
    domestic_resources = production(k, 1.0, parameters) - investment
    discount = jnp.exp(-parameters.interest_rate * t)
    h_terminal = jnp.log(k[-1] / steady["k"])
    continuation = steady["g"] / parameters.interest_rate + tail[
        "resource_loading"
    ] * h_terminal / (parameters.interest_rate - tail["stable_root"])
    resource_wealth = (
        a_0
        + jnp.trapezoid(discount * domestic_resources, t)
        + discount[-1] * continuation
    )
    consumption_wealth = jnp.trapezoid(discount * c, t) + discount[-1] * c[
        -1
    ] / consumption_annuity_rate(parameters)
    return consumption_annuity_rate(parameters) * (consumption_wealth - resource_wealth)


def tvc_diagnostics(k, a, q, c, t, parameters):
    mu = c ** (-parameters.gamma)
    discount_assets = jnp.exp(-parameters.interest_rate * t) * a
    utility_discount = jnp.exp(-parameters.r * t)
    return {
        "discounted_assets": jnp.abs(discount_assets),
        "asset_tvc": jnp.abs(utility_discount * mu * a),
        "capital_tvc": jnp.abs(utility_discount * mu * q * k),
    }
