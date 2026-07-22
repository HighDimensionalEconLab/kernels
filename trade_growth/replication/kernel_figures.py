from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate

from replication.utilities import style

PREFIX = "multicountry_trade_growth_kernel"
SCENARIOS = (
    ("three_country_initial_nfa", "Initial debt", "#b2182b"),
    ("three_country_zero_nfa", "Zero initial NFA", "#2166ac"),
    ("three_country_reversed_nfa", "Initial assets", "#1b7837"),
    ("three_country_large_debt", "Twice initial debt", "#762a83"),
)
PANELS = (
    ("a_1", "Country 1 net foreign assets"),
    ("consumption_1", "Country 1 consumption"),
    ("k_1", "Country 1 capital"),
    ("trade_balance_1", "Country 1 trade balance"),
)


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    frames = {
        name: pd.read_parquet(out / f"{PREFIX}_{name}_validation_df.parquet")
        for name, _, _ in SCENARIOS
    }

    style(figsize=(13, 10), fontsize=14, ticksize=12)
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for axis, (column, ylabel) in zip(axes.flat, PANELS, strict=True):
        for name, label, color in SCENARIOS:
            frame = frames[name]
            axis.plot(frame["t"], frame[column], color=color, label=label)
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.tight_layout()
    path = out / "multicountry_trade_growth_nfa_transitions.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")

    style(figsize=(12, 5), fontsize=14, ticksize=12)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    frame = frames["three_country_initial_nfa"]
    discount = np.exp(
        -scipy.integrate.cumulative_trapezoid(
            frame["interest_rate"], frame["t"], initial=0.0
        )
    )
    for country, color in zip(
        range(1, 4), ("#b2182b", "#2166ac", "#1b7837"), strict=True
    ):
        axes[0].semilogy(
            frame["t"],
            np.maximum(np.abs(discount * frame[f"a_{country}"]), 1e-16),
            color=color,
            label=f"Country {country}",
        )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Absolute discounted NFA")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    for column, label, color in (
        ("k", "Capital", "#b2182b"),
        ("a", "NFA", "#2166ac"),
        ("consumption", "Consumption", "#1b7837"),
        ("trade_balance", "Trade balance", "#762a83"),
    ):
        axes[1].semilogy(
            frame["t"],
            np.maximum(np.abs(frame[f"{column}_2"] - frame[f"{column}_3"]), 1e-16),
            color=color,
            label=label,
        )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Absolute country 2/3 path gap")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    path = out / "multicountry_trade_growth_ex_post_diagnostics.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
