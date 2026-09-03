from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from models.utilities import ParameterPyTree

# Shared calibration every open economy driver defaults to, so all solve the
# identical model.
OPEN_ECONOMY_GROWTH_CONSTANTS = {
    "alpha": 0.33,
    "gamma": 2.0,
    "delta": -math.log(0.94),
    "r": -math.log(0.96),
    "eta": 0.76,
    "r_star": -math.log(0.96),
}


# Deep structural parameters and the initial conditions together, all
# implicitly differentiable through the solver's `p` argument.
class StructuralParameters(ParameterPyTree):
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    r: jax.Array
    eta: jax.Array
    r_star: jax.Array
    k_0: jax.Array
    a_0: jax.Array


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


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 160.0
    train_points: int = 192
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 15
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401


# Ex-post acceptance bars shared by the NLLS, NN, and uno kernel drivers.
@dataclass(frozen=True)
class ToleranceSettings:
    gate_train: bool = True
    train_residual: float = 1e-5
    gate_validation: bool = True
    validation_residual: float = 1e-2


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


def present_value_gap(t, k, q, c, p):
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
        p.a_0
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
