from __future__ import annotations

import json
import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "multicountry_trade_growth_kernel"
SCENARIOS = (
    ("three_country_horizon_240", "Initial debt", "#b2182b"),
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
HORIZONS = (
    (80, "three_country_initial_nfa"),
    (120, "three_country_horizon_120"),
    (160, "three_country_horizon_160"),
    (240, "three_country_horizon_240"),
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

    summaries = []
    for horizon, name in HORIZONS:
        summary = json.loads((out / f"{PREFIX}_{name}.json").read_text())
        summaries.append((horizon, summary))

    style(figsize=(12, 5), fontsize=14, ticksize=12)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    horizons = [horizon for horizon, _ in summaries]
    axes[0].plot(
        horizons,
        [summary["terminal_a_1"] for _, summary in summaries],
        marker="o",
        color="#b2182b",
    )
    axes[0].set_xlabel("Solve horizon")
    axes[0].set_ylabel("Terminal country 1 NFA")
    axes[0].grid(alpha=0.2)
    for key, label, color in (
        ("ex_post_discounted_assets", "Discounted assets", "#2166ac"),
        ("ex_post_asset_tvc", "Asset TVC", "#1b7837"),
        ("ex_post_capital_tvc", "Capital TVC", "#762a83"),
    ):
        axes[1].semilogy(
            horizons,
            [summary[key] for _, summary in summaries],
            marker="o",
            color=color,
            label=label,
        )
    axes[1].set_xlabel("Solve horizon")
    axes[1].set_ylabel("Ex-post terminal diagnostic")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    path = out / "multicountry_trade_growth_horizon_diagnostics.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
