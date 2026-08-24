from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from models.kernels import (
    IntegratedPath,
    integrated_kernel_matrices,
    integrated_kernel_paths,
)


@dataclass(frozen=True)
class ModelParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    r: float = -math.log(0.96)
    eta: float = 0.76
    r_star: float = -math.log(0.96)


# Differentiable duck-type of ModelParameters: a pytree whose leaves can
# carry tracers, for implicit-AD comparative statics through the solver's
# `p` argument. The model functions only use attribute access, so either
# container works.
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ModelParameterValues:
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    r: jax.Array
    eta: jax.Array
    r_star: jax.Array


def model_parameter_values(p, dtype=jnp.float64):
    return ModelParameterValues(
        alpha=jnp.asarray(p.alpha, dtype=dtype),
        gamma=jnp.asarray(p.gamma, dtype=dtype),
        delta=jnp.asarray(p.delta, dtype=dtype),
        r=jnp.asarray(p.r, dtype=dtype),
        eta=jnp.asarray(p.eta, dtype=dtype),
        r_star=jnp.asarray(p.r_star, dtype=dtype),
    )


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


# Alpha blocks first (the metric penalizes the leading flat coordinates).
class Theta(NamedTuple):
    alpha_mu: jax.Array
    alpha_c: jax.Array
    alpha_k: jax.Array
    alpha_a: jax.Array
    alpha_q: jax.Array
    c_offset: jax.Array
    mu_offset: jax.Array
    q_offset: jax.Array


def residual_fn(theta, args, p):
    parameters_p = p["parameters"]
    steady_p = steady_state(parameters_p)
    # The asset block is the one path with no unpenalized offset, so its
    # scale enters the minimum-norm selection metric; the scale must be
    # differentiation-inert or every parameter sensitivity picks up a
    # selection-drift term from the metric moving with p. The other
    # normalizations are absorbed by their free offsets.
    output_scale_p = jax.lax.stop_gradient(steady_p["output"])
    log_consumption_scale_p = jnp.log(steady_p["g"])
    paths = integrated_kernel_paths(
        {
            "log_mu": IntegratedPath(
                theta.alpha_mu,
                -parameters_p.gamma * log_consumption_scale_p + theta.mu_offset,
            ),
            "log_c": IntegratedPath(
                theta.alpha_c, log_consumption_scale_p + theta.c_offset
            ),
            "log_k": IntegratedPath(theta.alpha_k, jnp.log(p["k_0"])),
            "a_raw": IntegratedPath(theta.alpha_a, jnp.zeros((), theta.alpha_a.dtype)),
            "log_q": IntegratedPath(
                theta.alpha_q, jnp.log(steady_p["q"]) + theta.q_offset
            ),
        },
        args["K"],
        args["K_tilde"],
    )
    k = jnp.exp(paths["log_k"].value)
    a = p["a_0"] + output_scale_p * paths["a_raw"].value
    q = jnp.exp(paths["log_q"].value)
    c = jnp.exp(paths["log_c"].value)
    mu = jnp.exp(paths["log_mu"].value)
    dlog_k_dt = paths["log_k"].d_dt
    da_dt = output_scale_p * paths["a_raw"].d_dt
    dlog_q_dt = paths["log_q"].d_dt
    dlog_mu_dt = paths["log_mu"].d_dt
    residuals = path_residuals(
        k,
        a,
        q,
        c,
        mu,
        jnp.ones_like(k),
        dlog_k_dt,
        da_dt,
        dlog_q_dt,
        dlog_mu_dt,
        p["parameters"],
    )
    residual = jnp.concatenate(
        [
            residuals["capital"],
            residuals["assets"],
            residuals["tobin_q"],
            residuals["marginal_wealth"],
            residuals["consumption_foc"],
        ]
    )
    return residual / jnp.sqrt(residual.size)


# Paths, model quantities, and DAE residuals on an evaluation grid through
# fresh cross-kernel matrices against the training grid.
def evaluate_grid(theta, points, train_data, kernel_set, p):
    K_eval, K_tilde_eval = integrated_kernel_matrices(
        points,
        train_data,
        kernel_set.kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    parameters_p = p["parameters"]
    steady_p = steady_state(parameters_p)
    output_scale_p = jax.lax.stop_gradient(steady_p["output"])
    log_consumption_scale_p = jnp.log(steady_p["g"])
    paths = integrated_kernel_paths(
        {
            "log_mu": IntegratedPath(
                theta.alpha_mu,
                -parameters_p.gamma * log_consumption_scale_p + theta.mu_offset,
            ),
            "log_c": IntegratedPath(
                theta.alpha_c, log_consumption_scale_p + theta.c_offset
            ),
            "log_k": IntegratedPath(theta.alpha_k, jnp.log(p["k_0"])),
            "a_raw": IntegratedPath(theta.alpha_a, jnp.zeros((), theta.alpha_a.dtype)),
            "log_q": IntegratedPath(
                theta.alpha_q, jnp.log(steady_p["q"]) + theta.q_offset
            ),
        },
        K_eval,
        K_tilde_eval,
    )
    k = jnp.exp(paths["log_k"].value)
    a = p["a_0"] + output_scale_p * paths["a_raw"].value
    q = jnp.exp(paths["log_q"].value)
    c = jnp.exp(paths["log_c"].value)
    mu = jnp.exp(paths["log_mu"].value)
    dlog_k_dt = paths["log_k"].d_dt
    da_dt = output_scale_p * paths["a_raw"].d_dt
    dlog_q_dt = paths["log_q"].d_dt
    dlog_mu_dt = paths["log_mu"].d_dt
    z = jnp.ones_like(k)
    residuals = path_residuals(
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
        parameters_p,
    )
    return {
        "t": points,
        "k": k,
        "a": a,
        "q": q,
        "c": c,
        "mu": mu,
        "x": investment_expenditure(k, q, parameters_p),
        "output": production(k, z, parameters_p),
        "current_account": da_dt,
        **residuals,
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
