# Consumption-Gini example: solve the reduced 10-country model on drawn
# heterogeneous initial conditions, compute the population-weighted
# consumption Gini on a coarse time grid, and its comparative static with
# respect to the CRRA coefficient gamma by forward-mode implicit AD
# through the whole solve (one tangent pass covers the entire grid).
#
# One-good risk sharing keeps consumption ratios constant along the exact
# path and the Gini is scale-invariant, so the THEORY predicts a
# time-constant Gini and a time-constant comparative static -- gamma moves
# the Gini only through the general-equilibrium revaluation of initial
# wealth a_0 + PV(resources), not through the sharing rule itself. The
# plots make both flat lines visible; the printed ranges measure how far
# the solved paths deviate from exactness.
#
#   JAX_PLATFORMS=cpu uv run python -m multicountry_growth_gini

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

from multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    consumption_gini,
)
from multicountry_growth_kernel import multicountry_growth_kernel

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def multicountry_growth_gini(
    num_countries: int = 10,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    time_stop: float = 80.0,
    time_step: float = 5.0,
    output_dir: str = ".results",
):
    parameters = MulticountryParameters(
        populations=(1.0,) * num_countries,
        productivities=(1.0,) * num_countries,
    )
    results, extras = multicountry_growth_kernel(
        parameters=parameters,
        initial_set=initial_set,
    )
    if not results["solution_accepted"]:
        raise SystemExit(f"solve rejected: {results['rejection_reason']}")

    grid = jnp.arange(0.0, time_stop + 0.5 * time_step, time_step)
    p = extras["p"]
    parametric_model = extras["parametric_model"]

    def gini_path(p_value):
        values = parametric_model(grid, p_value)
        return consumption_gini(values["c"], p_value["parameters"].populations)

    # The flat Gini is a statement about RATIOS, not levels: every country
    # shares the same Euler drift (r(t) - rho)/gamma, so consumption levels
    # all move together while relative consumption never changes. Print the
    # common level movement and the measured ratio drift to make the
    # distinction visible.
    solution = extras["model"](grid)
    consumption_levels = np.asarray(solution["c"])
    ratios = consumption_levels / consumption_levels[:, :1]
    populations = np.asarray(parameters.populations)
    aggregate = consumption_levels @ populations / populations.sum()
    print(
        f"aggregate consumption: {aggregate[0]:.6f} at t=0 -> "
        f"{aggregate[-1]:.6f} at t={float(grid[-1]):.0f} "
        f"({100.0 * (aggregate[-1] / aggregate[0] - 1.0):+.2f}%), "
        f"per-country growth spread "
        f"{np.ptp(consumption_levels[-1] / consumption_levels[0]):.2e}, "
        f"max ratio drift {np.max(np.abs(ratios / ratios[0] - 1.0)):.2e}"
    )

    # One forward tangent pass differentiates the whole Gini path: the
    # tangent pytree is zero except for a unit gamma direction.
    zero_tangent = jax.tree_util.tree_map(jnp.zeros_like, p)
    gamma_tangent = {
        **zero_tangent,
        "parameters": dataclasses.replace(
            zero_tangent["parameters"],
            gamma=jnp.ones_like(p["parameters"].gamma),
        ),
    }
    gini_values, gini_gamma_sensitivity = jax.jvp(gini_path, (p,), (gamma_tangent,))

    frame = pd.DataFrame(
        {
            "t": np.asarray(grid),
            "gini": np.asarray(gini_values),
            "d_gini_d_gamma": np.asarray(gini_gamma_sensitivity),
        }
    )
    print(frame.to_string(index=False))
    gini_range = float(frame["gini"].max() - frame["gini"].min())
    sensitivity_range = float(
        frame["d_gini_d_gamma"].max() - frame["d_gini_d_gamma"].min()
    )
    print(
        f"gini mean {frame['gini'].mean():.6f} (range {gini_range:.2e}), "
        f"d_gini/d_gamma mean {frame['d_gini_d_gamma'].mean():.6f} "
        f"(range {sensitivity_range:.2e})"
    )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["t"], frame["gini"], marker="o", color="#2166ac")
    axes[0].set_ylim(0.0, 1.25 * frame["gini"].max())
    axes[0].set_title(
        f"Consumption Gini, N={num_countries} (range {gini_range:.1e} across t)"
    )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Gini")
    center = frame["d_gini_d_gamma"].mean()
    pad = max(0.5 * abs(center), 5.0 * sensitivity_range, 1e-6)
    axes[1].plot(frame["t"], frame["d_gini_d_gamma"], marker="o", color="#b2182b")
    axes[1].set_ylim(center - pad, center + pad)
    axes[1].set_title(
        f"d Gini / d gamma via implicit AD (range {sensitivity_range:.1e} across t)"
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("d Gini / d gamma")
    figure.suptitle(
        "Risk sharing keeps consumption ratios constant: both are flat in t",
        y=1.02,
    )
    figure.tight_layout()
    figure_path = out / "multicountry_growth_gini.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    frame.to_json(out / "multicountry_growth_gini.json", orient="records", indent=2)
    print(f"wrote {figure_path}")

    summary = {
        "num_countries": num_countries,
        "initial_seed": initial_set.seed,
        "gamma": parameters.gamma,
        "gini_mean": float(frame["gini"].mean()),
        "gini_range": gini_range,
        "d_gini_d_gamma_mean": float(frame["d_gini_d_gamma"].mean()),
        "d_gini_d_gamma_range": sensitivity_range,
        "solve_accepted": results["solution_accepted"],
    }
    return summary, frame


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_growth_gini)
