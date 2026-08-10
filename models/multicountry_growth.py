from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Validity of the inputs, not an acceptance gate: checked in exact float64 on
# the raw values, so it holds at any working dtype.
INITIAL_CLEARING_TOL = 1e-10
# Initial debt service may not eat past this share of a country's own initial
# output: z_i f(k_0_i) + r a_0_i must stay above margin * z_i f(k_0_i), or
# positive consumption plus investment is infeasible from date zero.
DEBT_SERVICE_MARGIN = 0.05


@dataclass(frozen=True)
class MulticountryParameters:
    alpha: float = 0.33
    gammas: tuple[float, ...] = (2.0, 2.0)
    delta: float = -math.log(0.94)
    r: float = -math.log(0.96)
    eta: float = 0.76
    populations: tuple[float, ...] = (1.0, 1.0)
    productivities: tuple[float, ...] = (1.0, 1.0)

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.delta <= 0.0 or self.r <= 0.0:
            raise ValueError("delta and r must be positive")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("eta must lie in (0, 1)")
        if len(self.populations) < 2:
            raise ValueError("at least two countries are required")
        if not (len(self.gammas) == len(self.populations) == len(self.productivities)):
            raise ValueError(
                "gammas, populations, and productivities must have equal lengths"
            )
        if any(value <= 0.0 for value in self.gammas):
            raise ValueError("gammas must be positive")
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


# Differentiable duck-type of MulticountryParameters: a pytree whose leaves
# can carry tracers, for implicit-AD comparative statics through the
# solver's `p` argument. The model functions only use attribute access, so
# either container works; this one skips the float-only validation.
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MulticountryParameterValues:
    alpha: jax.Array
    gammas: jax.Array
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
        gammas=jnp.asarray(parameters.gammas, dtype=dtype),
        delta=jnp.asarray(parameters.delta, dtype=dtype),
        r=jnp.asarray(parameters.r, dtype=dtype),
        eta=jnp.asarray(parameters.eta, dtype=dtype),
        populations=jnp.asarray(parameters.populations, dtype=dtype),
        productivities=jnp.asarray(parameters.productivities, dtype=dtype),
    )


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


def validate_initial_conditions(k_0, a_0, parameters):
    num_countries = parameters.num_countries
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive capital stock per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one NFA position per country")
    populations = np.asarray(parameters.populations, dtype=np.float64)
    a_0_np = np.asarray(a_0, dtype=np.float64)
    scale = float(steady_state(parameters, jnp.float64)["output_scale"])
    clearing_gap = float(np.sum(populations * a_0_np)) / scale
    if abs(clearing_gap) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")
    productivities = np.asarray(parameters.productivities, dtype=np.float64)
    output_0 = productivities * np.asarray(k_0, dtype=np.float64) ** parameters.alpha
    margin = output_0 + parameters.r * a_0_np - DEBT_SERVICE_MARGIN * output_0
    if np.any(margin <= 0.0):
        worst = int(np.argmin(margin))
        raise ValueError(
            f"country {worst + 1} initial debt service exceeds output: "
            f"z f(k_0) + r a_0 = {output_0[worst] + parameters.r * a_0_np[worst]:.4f} "
            f"must exceed {DEBT_SERVICE_MARGIN:.0%} of its initial output "
            f"{output_0[worst]:.4f}"
        )


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
    gammas = jnp.asarray(parameters.gammas, dtype=mu.dtype)
    return mu ** (-1.0 / gammas)


def capital_drift(k, q, parameters):
    investment_rate = investment_rate_from_q(q, parameters)
    return k * (parameters.chi * investment_rate**parameters.eta - parameters.delta)


def current_account(k, a, q, c, z, rate, parameters):
    return (
        rate * a
        + production(k, z, parameters)
        - c
        - investment_expenditure(k, q, parameters)
    )


def q_drift(k, q, z, rate, parameters):
    eta = parameters.eta
    investment_rate = investment_rate_from_q(q, parameters)
    return (
        rate + parameters.delta - (1.0 - eta) * parameters.chi * investment_rate**eta
    ) * q - marginal_product(k, z, parameters)


def mu_drift(mu, rate, parameters):
    return (parameters.r - rate) * mu


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


def steady_state(parameters, dtype=jnp.float64):
    # Capital block of the steady state at r* = r, shared by the whole
    # continuum; the asset distribution stays undetermined up to clearing.
    # The world per-capita aggregates are the parameter-only path scales:
    # output for the asset paths, resources ("g") for consumption.
    productivities = jnp.asarray(parameters.productivities, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    eta = parameters.eta
    q = jnp.full(parameters.num_countries, 1.0 / eta, dtype=dtype)
    marginal_product_ss = parameters.delta + parameters.r / eta
    k = (parameters.alpha * productivities / marginal_product_ss) ** (
        1.0 / (1.0 - parameters.alpha)
    )
    output = production(k, productivities, parameters)
    investment = parameters.delta * k
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


def resolved_steady_state(mu, parameters):
    # The member of the steady-state continuum selected by the terminal
    # marginal-wealth levels: consumption from the FOC, assets from the
    # stationary budgets.
    mu = jnp.atleast_1d(mu)
    if mu.shape[-1] != parameters.num_countries:
        raise ValueError("mu must contain one value per country")
    dtype = mu.dtype
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    steady = steady_state(parameters, dtype)
    consumption = consumption_from_mu(mu, parameters)
    assets = (consumption - steady["g"]) / parameters.r
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


def path_residuals(
    k,
    a,
    q,
    c,
    mu,
    z,
    rate,
    dlog_k_dt,
    da_dt,
    dlog_q_dt,
    dlog_mu_dt,
    parameters,
):
    gammas = jnp.asarray(parameters.gammas, dtype=c.dtype)
    populations = jnp.asarray(parameters.populations, dtype=a.dtype)
    rate_column = rate[..., None]
    investment_rate = investment_rate_from_q(q, parameters)
    return {
        "capital": dlog_k_dt
        - (parameters.chi * investment_rate**parameters.eta - parameters.delta),
        "assets": da_dt - current_account(k, a, q, c, z, rate_column, parameters),
        "tobin_q": dlog_q_dt - q_drift(k, q, z, rate_column, parameters) / q,
        "marginal_wealth": dlog_mu_dt - mu_drift(mu, rate_column, parameters) / mu,
        "consumption_foc": mu * c**gammas - 1.0,
        # World bond clearing, per capita so the row size is N-free.
        "asset_clearing": jnp.sum(populations * a, axis=-1) / jnp.sum(populations),
    }


# Flat results fields d_<observable>_T<t>_d_<parameter>[_<country>] plus the
# observable levels <observable>_T<t>, and a long-form frame with one row per
# derivative entry, shared by the kernel and NN sensitivity blocks so both
# drivers publish identical names. Every country block also carries a
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


def tvc_diagnostics(t, values, parameters):
    # Certificates at the end of the supplied grid (t[-1] = train_T in the
    # drivers): assets discounted at the solved rate path, and the asset and
    # capital transversality products at the utility discount rate with
    # per-country marginal utilities c_i^{-gamma_i}.
    gammas = jnp.asarray(parameters.gammas, dtype=values["c"].dtype)
    discount = jnp.exp(-jnp.trapezoid(values["interest_rate"], t))
    utility_discount = jnp.exp(-parameters.r * t[-1])
    marginal_utility = values["c"][-1] ** (-gammas)
    return {
        "discounted_assets": jnp.abs(discount * values["a"][-1]),
        "asset_tvc": jnp.abs(utility_discount * marginal_utility * values["a"][-1]),
        "capital_tvc": jnp.abs(
            utility_discount * marginal_utility * values["q"][-1] * values["k"][-1]
        ),
    }


def tail_diagnostics(values, parameters):
    steady = resolved_steady_state(values["mu"][-1], parameters)
    scale = steady_state(parameters, values["k"].dtype)["output_scale"]
    return {
        "k_log_gap": jnp.abs(jnp.log(values["k"][-1] / steady["k"])),
        "q_log_gap": jnp.abs(jnp.log(values["q"][-1] / steady["q"])),
        "asset_scaled_gap": jnp.abs(values["a"][-1] - steady["a"]) / scale,
        "interest_rate_gap": jnp.abs(values["interest_rate"][-1] - parameters.r),
    }


def long_run_diagnostics(
    t,
    values,
    a_0,
    parameters,
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
    if check_T > float(t[-1]) or check_T - window < float(t[0]):
        raise ValueError("the long-run check window must lie inside the grid")
    a_0 = jnp.asarray(a_0, dtype=values["k"].dtype)
    i_check = int(jnp.argmin(jnp.abs(t - check_T)))
    i_before = int(jnp.argmin(jnp.abs(t - (check_T - window))))
    steady = resolved_steady_state(values["mu"][-1], parameters)
    scale = steady_state(parameters, values["k"].dtype)["output_scale"]
    productivities = jnp.asarray(parameters.productivities, dtype=values["k"].dtype)
    mpk_steady = parameters.delta + parameters.r / parameters.eta
    mpk_check = marginal_product(values["k"][i_check], productivities, parameters)
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
        "interest_rate_gap": jnp.abs(rate[i_check] - parameters.r) / parameters.r,
        "common_gamma": bool(
            np.all(np.asarray(parameters.gammas) == np.asarray(parameters.gammas)[0])
        ),
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
