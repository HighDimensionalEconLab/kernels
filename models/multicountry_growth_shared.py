from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from models.utilities import ParameterPyTree

# Validity of the inputs, not an acceptance gate: checked in exact float64 on
# the raw values, so it holds at any working dtype.
INITIAL_CLEARING_TOL = 1e-10
# Initial debt service may not eat past this share of a country's own initial
# output: z_i f(k_0_i) + r a_0_i must stay above margin * z_i f(k_0_i), or
# positive consumption plus investment is infeasible from date zero.
DEBT_SERVICE_MARGIN = 0.05


# Shared calibration every multicountry driver defaults to.
MULTICOUNTRY_GROWTH_CONSTANTS = {
    "alpha": 0.33,
    "gammas": (2.0, 2.0),
    "delta": -math.log(0.94),
    "r": -math.log(0.96),
    "eta": 0.76,
    "populations": (1.0, 1.0),
    "productivities": (1.0, 1.0),
}


# Deep structural parameters and the initial conditions together (vector
# leaves over countries), all implicitly differentiable through the solver's
# `p` argument. The country count is compile-time structure and stays
# host-side (len(populations) at the drivers).
class StructuralParameters(ParameterPyTree):
    alpha: jax.Array
    gammas: jax.Array
    delta: jax.Array
    r: jax.Array
    eta: jax.Array
    populations: jax.Array
    productivities: jax.Array
    # Optional ICs let steady-state consumers build a p from the constants.
    k_0: jax.Array | None = None
    a_0: jax.Array | None = None


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
    # Long-run validity checks are evaluated at t ~ long_run_check_T (flat
    # consumption/capital/debt, capital at the theory steady state, interest
    # rate ~ r), with flatness measured against
    # t = long_run_check_T - long_run_window.
    long_run_check_T: float = 100.0
    long_run_window: float = 10.0


# Identical initial conditions across countries make the multicountry
# dynamics degenerate (no cross-country variation to solve for), so the
# drivers DRAW heterogeneous defaults from these seeded distributions
# whenever k_0/a_0 are not passed explicitly.
@dataclass(frozen=True)
class InitialConditionSettings:
    # The default seed must keep the drawn bond positions well away from the
    # +-f(k*)/2 mirror configurations the replication overlays them with,
    # or two of the three plotted configurations coincide.
    seed: int = 24
    # k_0 = exp(k_0_log_sigma * z), z ~ N(0, 1): strictly positive with
    # ~k_0_log_sigma relative dispersion around 1.
    k_0_log_sigma: float = 0.25
    # a_0 = a_0_sigma * z, population-demeaned so the weighted NFA positions
    # clear exactly at t = 0. Debt positions want real dispersion: 0.6 puts
    # typical draws near the +-0.5 the mirror experiments use and reaches
    # past +-1 in the tails.
    a_0_sigma: float = 0.6


def draw_initial_conditions(initial_set, populations, dtype=jnp.float64):
    key_k, key_a = jax.random.split(jax.random.key(initial_set.seed))
    num_countries = len(populations)
    k_0 = jnp.exp(
        initial_set.k_0_log_sigma
        * jax.random.normal(key_k, (num_countries,), dtype=dtype)
    )
    populations = jnp.asarray(populations, dtype=dtype)
    raw = initial_set.a_0_sigma * jax.random.normal(
        key_a, (num_countries,), dtype=dtype
    )
    a_0 = raw - jnp.sum(populations * raw) / jnp.sum(populations)
    return tuple(float(value) for value in k_0), tuple(float(value) for value in a_0)


def validate_initial_conditions(p):
    populations = np.asarray(p.populations, dtype=np.float64)
    a_0_np = np.asarray(p.a_0, dtype=np.float64)
    scale = float(steady_state(p, jnp.float64)["output_scale"])
    clearing_gap = float(np.sum(populations * a_0_np)) / scale
    if abs(clearing_gap) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")
    productivities = np.asarray(p.productivities, dtype=np.float64)
    r = float(np.asarray(p.r, dtype=np.float64))
    alpha = float(np.asarray(p.alpha, dtype=np.float64))
    output_0 = productivities * np.asarray(p.k_0, dtype=np.float64) ** alpha
    margin = output_0 + r * a_0_np - DEBT_SERVICE_MARGIN * output_0
    if np.any(margin <= 0.0):
        worst = int(np.argmin(margin))
        raise ValueError(
            f"country {worst + 1} initial debt service exceeds output: "
            f"z f(k_0) + r a_0 = {output_0[worst] + r * a_0_np[worst]:.4f} "
            f"must exceed {DEBT_SERVICE_MARGIN:.0%} of its initial output "
            f"{output_0[worst]:.4f}"
        )


def production(k, z, p):
    return z * k**p.alpha


def marginal_product(k, z, p):
    return p.alpha * z * k ** (p.alpha - 1.0)


# chi = delta^(1 - eta) normalizes adjustment costs so investment equals
# replacement (rate delta) exactly at q = 1/eta.
def investment_rate_from_q(q, p):
    eta = p.eta
    chi = p.delta ** (1.0 - eta)
    return (chi * eta * q) ** (1.0 / (1.0 - eta))


def investment_expenditure(k, q, p):
    return k * investment_rate_from_q(q, p)


def consumption_from_mu(mu, p):
    gammas = jnp.asarray(p.gammas, dtype=mu.dtype)
    return mu ** (-1.0 / gammas)


def current_account(k, a, q, c, z, rate, p):
    return rate * a + production(k, z, p) - c - investment_expenditure(k, q, p)


def q_drift(k, q, z, rate, p):
    eta = p.eta
    chi = p.delta ** (1.0 - eta)
    investment_rate = investment_rate_from_q(q, p)
    return (
        rate + p.delta - (1.0 - eta) * chi * investment_rate**eta
    ) * q - marginal_product(k, z, p)


def relative_welfare_weights(mu):
    return mu[..., :1] / mu


def weighted_gini(values, populations):
    # Population-weighted Gini across countries (each country a mass of
    # identical members at its per-capita value):
    # sum_ij w_i w_j |v_i - v_j| / (2 mu) over the trailing country axis,
    # vectorized over any leading (time) axes. Scale-invariant, so for
    # consumption -- whose cross-country ratios are constant under a common
    # gamma -- it is constant along an exact solution path, while for output
    # or capital it measures sigma-convergence. Differentiable away from
    # exact value ties.
    values = jnp.asarray(values)
    weights = jnp.asarray(populations, dtype=values.dtype)
    weights = weights / jnp.sum(weights)
    mean = jnp.sum(weights * values, axis=-1)
    differences = jnp.abs(values[..., :, None] - values[..., None, :])
    dispersion = jnp.einsum("i,j,...ij->...", weights, weights, differences)
    return dispersion / (2.0 * mean)


def steady_state(p, dtype=jnp.float64):
    # Capital block of the steady state at r* = r, shared by the whole
    # continuum; the asset distribution stays undetermined up to clearing.
    # The world per-capita aggregates are the parameter-only path scales:
    # output for the asset paths, resources ("g") for consumption.
    productivities = jnp.asarray(p.productivities, dtype=dtype)
    populations = jnp.asarray(p.populations, dtype=dtype)
    eta = p.eta
    q = jnp.full(populations.shape[-1], 1.0 / eta, dtype=dtype)
    marginal_product_ss = p.delta + p.r / eta
    k = (p.alpha * productivities / marginal_product_ss) ** (1.0 / (1.0 - p.alpha))
    output = production(k, productivities, p)
    investment = p.delta * k
    domestic_resources = output - investment
    total_population = jnp.sum(populations)
    return {
        "k": k,
        "q": q,
        "x": investment,
        "output": output,
        "g": domestic_resources,
        "marginal_product": marginal_product_ss,
        "output_scale": jnp.sum(populations * output) / total_population,
        "consumption_scale": jnp.sum(populations * domestic_resources)
        / total_population,
    }


def resolved_steady_state(mu, p):
    # The member of the steady-state continuum selected by the terminal
    # marginal-wealth levels: consumption from the FOC, assets from the
    # stationary budgets.
    mu = jnp.atleast_1d(mu)
    dtype = mu.dtype
    populations = jnp.asarray(p.populations, dtype=dtype)
    steady = steady_state(p, dtype)
    consumption = consumption_from_mu(mu, p)
    assets = (consumption - steady["g"]) / p.r
    return {
        "k": steady["k"],
        "q": steady["q"],
        "a": assets,
        "c": consumption,
        "x": steady["x"],
        "output": steady["output"],
        "resources": steady["g"],
        "mu": mu,
        "relative_welfare_weights": relative_welfare_weights(mu),
        "resource_gap": jnp.sum(populations * (steady["g"] - consumption)),
        "asset_clearing_gap": jnp.sum(populations * assets),
    }


# The world rate, three cross-sectional world statistics, and every country's
# consumption and capital -- the observable stack both solvers' sensitivity
# callers differentiate, so the two tables are comparable cell by cell. The
# input dict needs the solvers' path keys (interest_rate, a, k, q, c);
# column order matches sensitivity_observable_labels.
def sensitivity_observables(values, p):
    populations = p.populations
    weights = populations / jnp.sum(populations)
    x = investment_expenditure(values["k"], values["q"], p)
    output = production(values["k"], p.productivities, p)
    # World saving is world investment: with no outside world the aggregate
    # current account is identically zero.
    saving_rate = jnp.sum(weights * x, axis=-1) / jnp.sum(weights * output, axis=-1)
    steady = steady_state(p, values["a"].dtype)
    nfa_dispersion = (
        jnp.sqrt(jnp.sum(weights * values["a"] ** 2, axis=-1)) / steady["output_scale"]
    )
    return jnp.column_stack(
        [
            values["interest_rate"],
            saving_rate,
            nfa_dispersion,
            weighted_gini(output, populations),
            values["c"],
            values["k"],
        ]
    )


def sensitivity_observable_labels(num_countries):
    return (
        ["rate", "saving_rate", "nfa_dispersion", "output_gini"]
        + [f"c_{country + 1}" for country in range(num_countries)]
        + [f"k_{country + 1}" for country in range(num_countries)]
    )


# Flat results fields d_<observable>_T<t>_d_<parameter>[_<country>] plus the
# observable levels <observable>_T<t>, and a long-form frame with one row per
# derivative entry, shared by the kernel and NN sensitivity callers so both
# publish identical names. Every country block also carries a
# _common entry, the derivative along (1, ..., 1): the country count drops
# out, so it is the one country-block statistic comparable across N. For a_0
# that direction is infeasible and the demeaning projects it away, leaving a
# measured zero. The two-country a_0_transfer entry is the derivative along
# the clearing-preserving transfer (1/L_1, -1/L_2): tau world units of bonds
# moved from country 2 to country 1.
def sensitivity_fields_frame(
    times,
    observable_labels,
    primal_observables,
    scalar_blocks,
    country_blocks,
    populations,
):
    num_countries = len(populations)
    fields = {}
    rows = []
    for index, time_value in enumerate(times):
        suffix = f"T{time_value:g}".replace(".", "p")
        for column, label in enumerate(observable_labels):
            fields[f"{label}_{suffix}"] = float(primal_observables[index, column])
            for name, block in scalar_blocks.items():
                value = float(block[index, column])
                fields[f"d_{label}_{suffix}_d_{name}"] = value
                rows.append(
                    {
                        "time": time_value,
                        "observable": label,
                        "parameter": name,
                        "country": 0,
                        "value": value,
                    }
                )
            for name, block in country_blocks.items():
                for country in range(num_countries):
                    value = float(block[index, column, country])
                    fields[f"d_{label}_{suffix}_d_{name}_{country + 1}"] = value
                    rows.append(
                        {
                            "time": time_value,
                            "observable": label,
                            "parameter": name,
                            "country": country + 1,
                            "value": value,
                        }
                    )
                common = float(np.sum(block[index, column]))
                fields[f"d_{label}_{suffix}_d_{name}_common"] = common
                rows.append(
                    {
                        "time": time_value,
                        "observable": label,
                        "parameter": f"{name}_common",
                        "country": 0,
                        "value": common,
                    }
                )
            if num_countries == 2 and "a_0" in country_blocks:
                transfer = float(
                    country_blocks["a_0"][index, column, 0] / populations[0]
                    - country_blocks["a_0"][index, column, 1] / populations[1]
                )
                fields[f"d_{label}_{suffix}_d_a_0_transfer"] = transfer
                rows.append(
                    {
                        "time": time_value,
                        "observable": label,
                        "parameter": "a_0_transfer",
                        "country": 0,
                        "value": transfer,
                    }
                )
    return fields, pd.DataFrame(rows)


def tvc_diagnostics(t, values, p):
    # Certificates at the end of the supplied grid (t[-1] = train_T in the
    # drivers): assets discounted at the solved rate path, and the asset and
    # capital transversality products at the utility discount rate with
    # per-country marginal utilities c_i^{-gamma_i}.
    gammas = jnp.asarray(p.gammas, dtype=values["c"].dtype)
    discount = jnp.exp(-jnp.trapezoid(values["interest_rate"], t))
    utility_discount = jnp.exp(-p.r * t[-1])
    marginal_utility = values["c"][-1] ** (-gammas)
    return {
        "discounted_assets": jnp.abs(discount * values["a"][-1]),
        "asset_tvc": jnp.abs(utility_discount * marginal_utility * values["a"][-1]),
        "capital_tvc": jnp.abs(
            utility_discount * marginal_utility * values["q"][-1] * values["k"][-1]
        ),
    }


def tail_diagnostics(values, p):
    steady = resolved_steady_state(values["mu"][-1], p)
    scale = steady_state(p, values["k"].dtype)["output_scale"]
    return {
        "k_log_gap": jnp.abs(jnp.log(values["k"][-1] / steady["k"])),
        "q_log_gap": jnp.abs(jnp.log(values["q"][-1] / steady["q"])),
        "asset_scaled_gap": jnp.abs(values["a"][-1] - steady["a"]) / scale,
        "interest_rate_gap": jnp.abs(values["interest_rate"][-1] - p.r),
    }


def long_run_diagnostics(
    t,
    values,
    a_0,
    p,
    check_T=100.0,
    window=10.0,
):
    # Host-side diagnostics (argsort, python bools): call on concrete
    # arrays only -- not jittable or traceable, unlike tvc/tail above.
    # Long-run validity at t ~ check_T (well before the training horizon).
    # The gamma-robust block holds for any curvature profile: flatness of
    # consumption, capital, and debt; capital at the theory steady state
    # (k_ss_i ~ z_i^(1/(1-alpha)) -- equal across countries only under equal
    # productivities, the general invariant is MPK equalization); and the
    # interest rate at r. The invariance identities -- constant consumption
    # ratios, consumption proportional to initial wealth a_0_i +
    # PV(resources_i), and wealth-rank preservation -- are exact only under
    # a common gamma, so they are computed only when every gamma_i is equal.
    a_0 = jnp.asarray(a_0, dtype=values["k"].dtype)
    i_check = int(jnp.argmin(jnp.abs(t - check_T)))
    i_before = int(jnp.argmin(jnp.abs(t - (check_T - window))))
    steady = resolved_steady_state(values["mu"][-1], p)
    scale = steady_state(p, values["k"].dtype)["output_scale"]
    productivities = jnp.asarray(p.productivities, dtype=values["k"].dtype)
    mpk_steady = p.delta + p.r / p.eta
    mpk_check = marginal_product(values["k"][i_check], productivities, p)
    rate = values["interest_rate"]
    diagnostics = {
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
        "interest_rate_gap": jnp.abs(rate[i_check] - p.r) / p.r,
        "common_gamma": bool(np.all(np.asarray(p.gammas) == np.asarray(p.gammas)[0])),
    }
    if not diagnostics["common_gamma"]:
        return diagnostics

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
        + discount[-1] * steady["resources"] / p.r
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
    diagnostics.update(
        {
            "consumption_ratio_drift": jnp.max(
                jnp.abs(consumption_ratios / consumption_ratios[0] - 1.0)
            ),
            "wealth_proportionality_gap": jnp.max(
                jnp.abs(
                    consumption_wealth_shares / jnp.mean(consumption_wealth_shares)
                    - 1.0
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
    )
    return diagnostics
