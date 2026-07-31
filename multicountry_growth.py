from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MulticountryParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    r: float = -math.log(0.96)
    eta: float = 0.76
    populations: tuple[float, ...] = (1.0, 1.0)
    productivities: tuple[float, ...] = (1.0, 1.0)

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.delta <= 0.0 or self.r <= 0.0:
            raise ValueError("delta and r must be positive")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("eta must lie in (0, 1)")
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
    def chi(self):
        return self.delta ** (1.0 - self.eta)


# Identical initial conditions across countries make the multicountry
# dynamics degenerate (no cross-country variation to solve for), so the
# drivers DRAW heterogeneous defaults from these seeded distributions
# whenever k_0/a_0 are not passed explicitly.
@dataclass(frozen=True)
class InitialConditionSettings:
    seed: int = 0
    # k_0 = exp(k_0_log_sigma * z), z ~ N(0, 1): strictly positive with
    # ~k_0_log_sigma relative dispersion around 1.
    k_0_log_sigma: float = 0.25
    # a_0 = a_0_sigma * z, population-demeaned so the weighted NFA positions
    # clear exactly at t = 0. Debt positions want real dispersion: 0.6 puts
    # typical draws near the +-0.5 the hand-calibrated experiments used and
    # reaches past +-1 in the tails.
    a_0_sigma: float = 0.6


# Differentiable duck-type of MulticountryParameters: a pytree whose leaves
# can carry tracers, for implicit-AD comparative statics through the
# solver's `p` argument. The model functions only use attribute access, so
# either container works; this one skips the float-only validation.
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MulticountryParameterValues:
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    r: jax.Array
    eta: jax.Array
    populations: jax.Array
    productivities: jax.Array

    @property
    def num_countries(self):
        return self.populations.shape[-1]

    @property
    def chi(self):
        return self.delta ** (1.0 - self.eta)


def multicountry_parameter_values(parameters, dtype=jnp.float64):
    return MulticountryParameterValues(
        alpha=jnp.asarray(parameters.alpha, dtype=dtype),
        gamma=jnp.asarray(parameters.gamma, dtype=dtype),
        delta=jnp.asarray(parameters.delta, dtype=dtype),
        r=jnp.asarray(parameters.r, dtype=dtype),
        eta=jnp.asarray(parameters.eta, dtype=dtype),
        populations=jnp.asarray(parameters.populations, dtype=dtype),
        productivities=jnp.asarray(parameters.productivities, dtype=dtype),
    )


def draw_initial_conditions(initial_set, parameters, dtype=jnp.float64):
    key_k, key_a = jax.random.split(jax.random.key(initial_set.seed))
    num_countries = parameters.num_countries
    k_0 = jnp.exp(
        initial_set.k_0_log_sigma
        * jax.random.normal(key_k, (num_countries,), dtype=dtype)
    )
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    raw = initial_set.a_0_sigma * jax.random.normal(
        key_a, (num_countries,), dtype=dtype
    )
    a_0 = raw - jnp.sum(populations * raw) / jnp.sum(populations)
    return tuple(float(value) for value in k_0), tuple(float(value) for value in a_0)


def production(k, z, parameters):
    return z * k**parameters.alpha


def marginal_product(k, z, parameters):
    return parameters.alpha * z * k ** (parameters.alpha - 1.0)


def investment_rate_from_q(q, parameters):
    elasticity = parameters.eta
    return (parameters.chi * elasticity * q) ** (1.0 / (1.0 - elasticity))


def consumption_from_mu(mu, parameters):
    return mu ** (-1.0 / parameters.gamma)


def relative_welfare_weights(mu):
    return mu[..., :1] / mu


def weighted_gini(values, populations):
    # Population-weighted Gini across countries (each country a mass of
    # identical members at its per-capita value):
    # sum_ij w_i w_j |v_i - v_j| / (2 mu) over the trailing country axis,
    # vectorized over any leading (time) axes. Scale-invariant, so for
    # consumption -- whose cross-country ratios are constant under the
    # common Euler drift -- it is constant along an exact solution path,
    # while for output or capital it measures sigma-convergence.
    # Differentiable away from exact value ties.
    values = jnp.asarray(values)
    weights = jnp.asarray(populations, dtype=values.dtype)
    weights = weights / jnp.sum(weights)
    mean = jnp.sum(weights * values, axis=-1)
    differences = jnp.abs(values[..., :, None] - values[..., None, :])
    dispersion = jnp.einsum("i,j,...ij->...", weights, weights, differences)
    return dispersion / (2.0 * mean)


def multicountry_allocations(
    k,
    q,
    mu,
    interest_rate,
    parameters,
):
    productivities = jnp.asarray(parameters.productivities, dtype=k.dtype)
    investment_rate = investment_rate_from_q(q, parameters)
    investment = k * investment_rate
    output = production(k, productivities, parameters)
    mpk = marginal_product(k, productivities, parameters)
    resources = output - investment
    consumption = consumption_from_mu(mu, parameters)
    capital_drift = k * (
        parameters.chi * investment_rate**parameters.eta - parameters.delta
    )
    q_intercept = (
        parameters.delta
        - (1.0 - parameters.eta) * parameters.chi * investment_rate**parameters.eta
        - mpk / q
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
        "tobin_q_drift": q * (rate + q_intercept),
        "marginal_wealth_drift": mu * (parameters.r - rate),
    }


def multicountry_reference_mu(parameters, dtype=jnp.float64):
    productivities = jnp.asarray(parameters.productivities, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    mpk = parameters.delta + parameters.r / parameters.eta
    k = (parameters.alpha * productivities / mpk) ** (1.0 / (1.0 - parameters.alpha))
    investment = parameters.delta * k
    resources = production(k, productivities, parameters) - investment
    consumption = jnp.sum(populations * resources) / jnp.sum(populations)
    return jnp.full(
        parameters.num_countries,
        consumption ** (-parameters.gamma),
        dtype=dtype,
    )


def multicountry_steady_state(mu, parameters):
    mu = jnp.atleast_1d(mu)
    if mu.shape[-1] != parameters.num_countries:
        raise ValueError("marginal_wealth must contain one value per country")
    dtype = mu.dtype
    productivities = jnp.asarray(parameters.productivities, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    q = jnp.full(
        parameters.num_countries,
        1.0 / parameters.eta,
        dtype=dtype,
    )
    mpk = parameters.delta + parameters.r / parameters.eta
    k = (parameters.alpha * productivities / mpk) ** (1.0 / (1.0 - parameters.alpha))
    allocations = multicountry_allocations(
        k,
        q,
        mu,
        jnp.asarray(parameters.r, dtype=dtype),
        parameters,
    )
    assets = (allocations["consumption"] - allocations["resources"]) / parameters.r
    return {
        "k": k,
        "q": q,
        "a": assets,
        "c": allocations["consumption"],
        "x": allocations["investment"],
        "output": allocations["output"],
        "resources": allocations["resources"],
        "marginal_wealth": mu,
        "relative_welfare_weights": relative_welfare_weights(mu),
        "interest_rate": jnp.asarray(parameters.r, dtype=dtype),
        "resource_gap": jnp.sum(
            populations * (allocations["resources"] - allocations["consumption"])
        ),
        "asset_clearing_gap": jnp.sum(populations * assets),
    }


def multicountry_reference_steady_state(parameters, dtype=jnp.float64):
    return multicountry_steady_state(
        multicountry_reference_mu(parameters, dtype),
        parameters,
    )


def multicountry_output_scale(parameters, dtype=jnp.float64):
    steady = multicountry_reference_steady_state(parameters, dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    return jnp.sum(populations * steady["output"]) / jnp.sum(populations)


def multicountry_path_values(
    k,
    a,
    q,
    mu,
    interest_rate,
    dlog_k_dt,
    da_dt,
    dlog_q_dt,
    dlog_mu_dt,
    parameters,
):
    allocations = multicountry_allocations(
        k,
        q,
        mu,
        interest_rate,
        parameters,
    )
    populations = jnp.asarray(parameters.populations, dtype=k.dtype)
    budget_drift = (
        interest_rate[:, None] * a
        + allocations["resources"]
        - allocations["consumption"]
    )
    residuals = {
        "capital": dlog_k_dt - allocations["capital_drift"] / k,
        "assets": da_dt - budget_drift,
        "tobin_q": dlog_q_dt - allocations["tobin_q_drift"] / q,
        "marginal_wealth_euler": dlog_mu_dt - (parameters.r - interest_rate[:, None]),
        "resource": jnp.sum(
            populations * (allocations["resources"] - allocations["consumption"]),
            axis=-1,
        ),
        "resource_drift": jnp.sum(
            populations
            * (
                (allocations["mpk"] * k - allocations["investment"]) * dlog_k_dt
                - allocations["investment"] * dlog_q_dt / (1.0 - parameters.eta)
                + allocations["consumption"] * dlog_mu_dt / parameters.gamma
            ),
            axis=-1,
        ),
        "asset_clearing": jnp.sum(populations * a, axis=-1),
        "interest_rate_country_gap": parameters.r - dlog_mu_dt - interest_rate[:, None],
    }
    return {
        "k": k,
        "a": a,
        "q": q,
        "c": allocations["consumption"],
        "x": allocations["investment"],
        "output": allocations["output"],
        "resources": allocations["resources"],
        "current_account": da_dt,
        "interest_rate": interest_rate,
        "marginal_wealth": mu,
        "relative_welfare_weights": relative_welfare_weights(mu),
        **residuals,
    }


def multicountry_tvc_diagnostics(t, values, parameters):
    discount = jnp.exp(-jnp.trapezoid(values["interest_rate"], t))
    utility_discount = jnp.exp(-parameters.r * t[-1])
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
        "marginal_wealth_drift": jnp.abs(parameters.r - values["interest_rate"][-1]),
        "interest_rate_gap": jnp.abs(values["interest_rate"][-1] - parameters.r),
    }


def multicountry_long_run_diagnostics(
    t,
    values,
    a_0,
    parameters,
    check_T=100.0,
    window=10.0,
):
    # Long-run validity of the solved paths, checked at t ~ check_T (well
    # before the training horizon): flatness of consumption, capital, and
    # debt; per-country convergence of capital to the THEORY steady state
    # (k_ss_i ~ A_i^(1/(1-alpha)) -- equal across countries only under equal
    # productivities, the general invariant is MPK equalization); interest rate -> r;
    # and the exact one-good sharing structure -- consumption RATIOS are
    # constant in t (common Euler drift), consumption LEVELS are
    # proportional to initial total wealth a_0_i + PV(resources_i) (with
    # decreasing returns V != qk, so the PV is integrated numerically along
    # the solved rate path with a steady-state tail), and long-run
    # consumption therefore ranks with initial wealth. The a_0-only rank
    # comparisons hold when k_0 and productivities are homogeneous and are
    # reported as ungated diagnostics.
    a_0 = jnp.asarray(a_0, dtype=values["k"].dtype)
    i_check = int(jnp.argmin(jnp.abs(t - check_T)))
    i_before = int(jnp.argmin(jnp.abs(t - (check_T - window))))
    steady = multicountry_steady_state(values["marginal_wealth"][-1], parameters)
    scale = multicountry_output_scale(parameters, values["k"].dtype)
    productivities = jnp.asarray(parameters.productivities, dtype=values["k"].dtype)
    mpk_steady = parameters.delta + parameters.r / parameters.eta
    mpk_check = marginal_product(values["k"][i_check], productivities, parameters)
    rate = values["interest_rate"]
    consumption_ratios = values["c"] / values["c"][:, :1]
    cumulative_rate = jnp.concatenate(
        [
            jnp.zeros(1, dtype=rate.dtype),
            jnp.cumsum(0.5 * (rate[1:] + rate[:-1]) * jnp.diff(t)),
        ]
    )
    discount = jnp.exp(-cumulative_rate)
    resource_pv = (
        jnp.trapezoid(discount[:, None] * values["resources"], t, axis=0)
        + discount[-1] * steady["resources"] / parameters.r
    )
    initial_wealth = a_0 + resource_pv
    consumption_wealth_shares = values["c"][0] / initial_wealth
    a_0_order = np.argsort(np.asarray(a_0))
    # Exact rank equality is tie-fragile at large N: the proportionality
    # spread bounds how close two wealth levels can be before their
    # consumption ordering is numerically undetermined. The gated quantity
    # is therefore the largest relative wealth separation among misordered
    # pairs -- zero when the ranking is exact, and at tie scale when an
    # exact-rank mismatch is a numerical tie rather than a violation.
    wealth_scaled = initial_wealth / jnp.mean(initial_wealth)
    wealth_differences = wealth_scaled[:, None] - wealth_scaled[None, :]
    consumption_differences = steady["c"][:, None] - steady["c"][None, :]
    wealth_rank_violation = jnp.max(
        jnp.where(
            wealth_differences * consumption_differences < 0.0,
            jnp.abs(wealth_differences),
            0.0,
        )
    )
    return {
        "c_flatness": jnp.max(
            jnp.abs(values["c"][i_check] / values["c"][i_before] - 1.0)
        ),
        "k_flatness": jnp.max(
            jnp.abs(jnp.log(values["k"][i_check] / values["k"][i_before]))
        ),
        "a_flatness": jnp.max(
            jnp.abs(values["a"][i_check] - values["a"][i_before]) / scale
        ),
        "k_theory_gap": jnp.max(jnp.abs(jnp.log(values["k"][i_check] / steady["k"]))),
        "mpk_equalization_gap": jnp.max(jnp.abs(mpk_check - mpk_steady)) / mpk_steady,
        "interest_rate_gap": jnp.abs(rate[i_check] - parameters.r) / parameters.r,
        "consumption_ratio_drift": jnp.max(
            jnp.abs(consumption_ratios / consumption_ratios[0] - 1.0)
        ),
        "wealth_proportionality_gap": jnp.max(
            jnp.abs(
                consumption_wealth_shares / jnp.mean(consumption_wealth_shares) - 1.0
            )
        ),
        "initial_wealth": initial_wealth,
        "wealth_rank_violation": wealth_rank_violation,
        "c_rank_matches_wealth": bool(
            np.array_equal(
                np.argsort(np.asarray(steady["c"])),
                np.argsort(np.asarray(initial_wealth)),
            )
        ),
        "c_rank_matches_a_0": bool(
            np.array_equal(np.argsort(np.asarray(steady["c"])), a_0_order)
        ),
        "terminal_a_rank_matches_a_0": bool(
            np.array_equal(np.argsort(np.asarray(values["a"][-1])), a_0_order)
        ),
    }
