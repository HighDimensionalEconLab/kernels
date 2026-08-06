# Transition-statistics example: solve the reduced 10-country model on
# drawn heterogeneous initial conditions and differentiate aggregate
# statistics through the whole solve, in both AD directions.
#
# Time panels (t = 0, 5, ..., 80), each paired with its d/d gamma path
# from ONE forward tangent pass:
#   - world saving = investment rate X/Y (boom decaying to the steady
#     replacement share; smoothing cuts early saving, so d s / d gamma < 0
#     early);
#   - the world interest rate (converging to r as growth flattens);
#   - the population-weighted output Gini (sigma-convergence: capital and
#     output converge as MPKs equalize);
#   - NFA dispersion (population-weighted RMS of the net positions,
#     relative to steady-state output): external positions diverge to a
#     permanent plateau even as output converges -- while the consumption
#     Gini stays frozen by risk sharing.
#
# High-dimensional redistribution gradients, one reverse pass per scalar
# through jax.jacrev of a stacked functional: the consumption Gini and the
# terminal NFA dispersion respond country-by-country to the initial bond
# and capital positions, while r(t=10) is provably NEUTRAL to feasible
# bond redistributions (identical homothetic preferences) -- its a_0
# gradient is the machine-zero row of the same Jacobian.
#
#   JAX_PLATFORMS=cpu uv run python -m models.multicountry_growth_transitions

from __future__ import annotations

import dataclasses
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    steady_state,
    weighted_gini,
)
from models.multicountry_growth_kernel import multicountry_growth_kernel

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def multicountry_growth_transitions(
    num_countries: int = 10,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    time_stop: float = 80.0,
    time_step: float = 5.0,
    rate_time: float = 10.0,
    output_dir: str = ".results",
):
    parameters = MulticountryParameters(
        gammas=(2.0,) * num_countries,
        populations=(1.0,) * num_countries,
        productivities=(1.0,) * num_countries,
    )
    results, extras = multicountry_growth_kernel(
        parameters=parameters,
        initial_set=initial_set,
    )
    if not results["solution_accepted"]:
        raise SystemExit(f"solve rejected: {results['rejection_reason']}")
    output_scale = steady_state(parameters)["output_scale"]

    grid = jnp.arange(0.0, time_stop + 0.5 * time_step, time_step)
    p = extras["p"]
    parametric_model = extras["parametric_model"]

    def aggregate_paths(p_value):
        values = parametric_model(grid, p_value)
        weights = p_value["parameters"].populations
        weights = weights / jnp.sum(weights)
        return {
            "saving_rate": jnp.sum(weights * values["x"], axis=-1)
            / jnp.sum(weights * values["output"], axis=-1),
            "interest_rate": values["interest_rate"],
            "output_gini": weighted_gini(
                values["output"], p_value["parameters"].populations
            ),
            "nfa_dispersion": jnp.sqrt(jnp.sum(weights * values["a"] ** 2, axis=-1))
            / output_scale,
        }

    # d/d gamma for every panel at once: the tangent pytree is zero except
    # for a unit common-gamma direction (every country's gamma_i moved
    # together), and one jax.jvp pass differentiates the full dictionary of
    # paths through the solve.
    zero_tangent = jax.tree_util.tree_map(jnp.zeros_like, p)
    gamma_tangent = {
        **zero_tangent,
        "parameters": dataclasses.replace(
            zero_tangent["parameters"],
            gammas=jnp.ones_like(p["parameters"].gammas),
        ),
    }
    paths, gamma_paths = jax.jvp(aggregate_paths, (p,), (gamma_tangent,))

    frame = pd.DataFrame(
        {
            "t": np.asarray(grid),
            **{name: np.asarray(paths[name]) for name in paths},
            **{f"d_{name}_d_gamma": np.asarray(gamma_paths[name]) for name in paths},
        }
    )
    print(frame.to_string(index=False))

    # The redistribution Jacobian: three scalar statistics stacked so ONE
    # jacrev call (one primal solve, one reverse pass per row) yields the
    # gradient of each with respect to EVERY country-specific input in p.
    # Feasible bond gradients are the demeaning-projected ones, so the r
    # row measures redistribution neutrality directly.
    def scalar_statistics(p_value):
        values = parametric_model(
            jnp.asarray([0.0, rate_time, time_stop], dtype=grid.dtype), p_value
        )
        weights = p_value["parameters"].populations
        weights = weights / jnp.sum(weights)
        consumption_gini = weighted_gini(
            values["c"][0], p_value["parameters"].populations
        )
        terminal_dispersion = (
            jnp.sqrt(jnp.sum(weights * values["a"][-1] ** 2)) / output_scale
        )
        return jnp.stack(
            [consumption_gini, terminal_dispersion, values["interest_rate"][1]]
        )

    jacobian = jax.jacrev(scalar_statistics)(p)
    statistic_names = ("consumption_gini", "terminal_nfa_dispersion", "interest_rate")
    gradients = pd.DataFrame(
        {
            "country": np.arange(1, num_countries + 1),
            "initial_wealth": [
                results[f"initial_wealth_{country + 1}"]
                for country in range(num_countries)
            ],
            "a_0": [results[f"a_0_{country + 1}"] for country in range(num_countries)],
            "k_0": [results[f"k_0_{country + 1}"] for country in range(num_countries)],
            **{
                f"d_{name}_d_a_0": np.asarray(jacobian["a_0"][row])
                for row, name in enumerate(statistic_names)
            },
            **{
                f"d_{name}_d_k_0": np.asarray(jacobian["k_0"][row])
                for row, name in enumerate(statistic_names)
            },
        }
    ).sort_values("initial_wealth", ignore_index=True)
    print(gradients.to_string(index=False))
    neutrality = float(np.max(np.abs(gradients["d_interest_rate_d_a_0"])))
    inequality_scale = float(np.max(np.abs(gradients["d_consumption_gini_d_a_0"])))
    print(
        f"redistribution neutrality: max |d r({rate_time:.0f})/d a_0| = "
        f"{neutrality:.2e} vs max |d Gini_c/d a_0| = {inequality_scale:.2e}"
    )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = (
        ("saving_rate", "World saving = investment rate"),
        ("interest_rate", "World interest rate"),
        ("output_gini", "Output Gini (sigma-convergence)"),
        ("nfa_dispersion", "NFA dispersion / steady output"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(11, 14))
    for row, (name, label) in enumerate(panels):
        axes[row, 0].plot(frame["t"], frame[name], marker="o", color="#2166ac")
        axes[row, 0].set_ylabel(label)
        axes[row, 1].plot(
            frame["t"], frame[f"d_{name}_d_gamma"], marker="o", color="#b2182b"
        )
        axes[row, 1].axhline(0.0, color="gray", linewidth=0.8)
        axes[row, 1].set_ylabel("d/d gamma")
    for axis in axes[-1]:
        axis.set_xlabel("t")
    axes[0, 0].set_title("Transition")
    axes[0, 1].set_title("Comparative static (implicit AD)")
    figure.tight_layout()
    transitions_path = out / "multicountry_growth_transitions.png"
    figure.savefig(transitions_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    positions = np.arange(num_countries)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    bar_panels = (
        ("consumption_gini", "d consumption Gini"),
        ("terminal_nfa_dispersion", f"d NFA dispersion(t={time_stop:.0f})"),
        ("interest_rate", f"d r(t={rate_time:.0f})  [neutral]"),
    )
    for axis, (name, label) in zip(axes, bar_panels, strict=True):
        axis.bar(
            positions - 0.2,
            gradients[f"d_{name}_d_a_0"],
            width=0.4,
            color="#2166ac",
            label="w.r.t. a_0 (bonds)",
        )
        axis.bar(
            positions + 0.2,
            gradients[f"d_{name}_d_k_0"],
            width=0.4,
            color="#b2182b",
            label="w.r.t. k_0",
        )
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_title(label)
        axis.set_xticks(positions)
        axis.set_xticklabels(gradients["country"])
        axis.set_xlabel("country (sorted by initial wealth)")
    axes[0].legend()
    axes[-1].annotate(
        f"bond bars are invisible by theorem:\nmax |d r/d a_0| = {neutrality:.1e}",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize=9,
    )
    figure.suptitle(
        "Redistribution gradients: inequality responds country-by-country, "
        "the world rate does not",
        y=1.04,
    )
    figure.tight_layout()
    gradients_path = out / "multicountry_growth_redistribution_gradients.png"
    figure.savefig(gradients_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    frame.to_json(
        out / "multicountry_growth_transitions.json", orient="records", indent=2
    )
    gradients.to_json(
        out / "multicountry_growth_redistribution_gradients.json",
        orient="records",
        indent=2,
    )
    print(f"wrote {transitions_path} and {gradients_path}")

    summary = {
        "num_countries": num_countries,
        "initial_seed": initial_set.seed,
        "gamma": parameters.gammas[0],
        "saving_rate_start": float(frame["saving_rate"].iloc[0]),
        "saving_rate_end": float(frame["saving_rate"].iloc[-1]),
        "output_gini_start": float(frame["output_gini"].iloc[0]),
        "output_gini_end": float(frame["output_gini"].iloc[-1]),
        "nfa_dispersion_start": float(frame["nfa_dispersion"].iloc[0]),
        "nfa_dispersion_end": float(frame["nfa_dispersion"].iloc[-1]),
        "rate_neutrality_max_a_0_gradient": neutrality,
        "consumption_gini_max_a_0_gradient": inequality_scale,
        "solve_accepted": results["solution_accepted"],
    }
    return summary, frame, gradients


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_growth_transitions)
