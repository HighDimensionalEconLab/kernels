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
    r_star: float = -math.log(0.96)


def production(k, z, p):
    return z * k**p.alpha


def marginal_product(k, z, p):
    return p.alpha * z * k ** (p.alpha - 1.0)


# chi = delta^(1 - eta) normalizes adjustment costs so the steady-state
# investment rate is exactly delta.
def investment_rate_from_q(q, p):
    eta = p.eta
    chi = p.delta ** (1.0 - eta)
    return (chi * eta * q) ** (1.0 / (1.0 - eta))


def investment_expenditure(k, q, p):
    return k * investment_rate_from_q(q, p)


def consumption_from_mu(mu, p):
    return mu ** (-1.0 / p.gamma)


def capital_drift(k, q, p):
    chi = p.delta ** (1.0 - p.eta)
    investment_rate = investment_rate_from_q(q, p)
    return k * (chi * investment_rate**p.eta - p.delta)


def current_account(k, a, q, c, z, p):
    return p.r_star * a + production(k, z, p) - c - investment_expenditure(k, q, p)


def q_drift(k, q, z, p):
    eta = p.eta
    chi = p.delta ** (1.0 - eta)
    investment_rate = investment_rate_from_q(q, p)
    return (
        p.r_star + p.delta - (1.0 - eta) * chi * investment_rate**eta
    ) * q - marginal_product(k, z, p)


def mu_drift(mu, p):
    return (p.r - p.r_star) * mu


def consumption_growth_rate(p):
    return (p.r_star - p.r) / p.gamma


def consumption_annuity_rate(p):
    return p.r_star - consumption_growth_rate(p)


def steady_state(p):
    eta = p.eta
    investment_rate = p.delta
    q = 1.0 / eta
    marginal_product_ss = p.delta + p.r_star / eta
    k = (p.alpha / marginal_product_ss) ** (1.0 / (1.0 - p.alpha))
    output = production(k, 1.0, p)
    investment = p.delta * k
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


def stable_tail_terms(p):
    steady = steady_state(p)
    eta = p.eta
    capital_loading = p.delta * eta / (1.0 - eta)
    q_loading = (1.0 - p.alpha) * (p.r_star + eta * p.delta)
    discriminant = p.r_star**2 + 4.0 * capital_loading * q_loading
    stable_root = 0.5 * (p.r_star - jnp.sqrt(discriminant))
    unstable_root = 0.5 * (p.r_star + jnp.sqrt(discriminant))
    stable_slope = stable_root / capital_loading
    resource_loading = (
        p.alpha * steady["output"]
        - p.delta * steady["k"]
        - p.delta * steady["k"] * stable_slope / (1.0 - eta)
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
    p,
):
    chi = p.delta ** (1.0 - p.eta)
    investment_rate = investment_rate_from_q(q, p)
    return {
        "capital": dlog_k_dt - (chi * investment_rate**p.eta - p.delta),
        "assets": da_dt - current_account(k, a, q, c, z, p),
        "tobin_q": dlog_q_dt - q_drift(k, q, z, p) / q,
        "marginal_wealth": dlog_mu_dt - mu_drift(mu, p) / mu,
        "consumption_foc": mu * c**p.gamma - 1.0,
    }


def tail_diagnostics(k, a, q, c, p):
    steady = steady_state(p)
    tail = stable_tail_terms(p)
    h = jnp.log(k / steady["k"])
    v = jnp.log(q / steady["q"])
    stable = v - tail["stable_slope"] * h
    tail_resources = steady["g"] / p.r_star + tail["resource_loading"] * h / (
        p.r_star - tail["stable_root"]
    )
    present_value = (c - consumption_annuity_rate(p) * (a + tail_resources)) / steady[
        "output"
    ]
    return {"stable": stable, "present_value": present_value}


def present_value_gap(t, k, q, c, a_0, p):
    steady = steady_state(p)
    tail = stable_tail_terms(p)
    investment = investment_expenditure(k, q, p)
    domestic_resources = production(k, 1.0, p) - investment
    discount = jnp.exp(-p.r_star * t)
    h_terminal = jnp.log(k[-1] / steady["k"])
    continuation = steady["g"] / p.r_star + tail["resource_loading"] * h_terminal / (
        p.r_star - tail["stable_root"]
    )
    resource_wealth = (
        a_0
        + jnp.trapezoid(discount * domestic_resources, t)
        + discount[-1] * continuation
    )
    consumption_wealth = jnp.trapezoid(discount * c, t) + discount[-1] * c[
        -1
    ] / consumption_annuity_rate(p)
    return consumption_annuity_rate(p) * (consumption_wealth - resource_wealth)


def tvc_diagnostics(k, a, q, c, t, p):
    mu = c ** (-p.gamma)
    discount_assets = jnp.exp(-p.r_star * t) * a
    utility_discount = jnp.exp(-p.r * t)
    return {
        "discounted_assets": jnp.abs(discount_assets),
        "asset_tvc": jnp.abs(utility_discount * mu * a),
        "capital_tvc": jnp.abs(utility_discount * mu * q * k),
    }
