from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "multicountry_growth_kernel"
SCENARIOS = (
    ("initial_debt", "Country 1 debt", "#b2182b"),
    ("baseline", "Drawn initial NFA", "#2166ac"),
    ("initial_assets", "Country 1 assets", "#1b7837"),
)
PANELS = (
    ("k", "Capital"),
    ("q", "Installed-capital value"),
    ("c", "Consumption"),
    ("a", "Net foreign assets"),
    ("current_account", "Current account"),
    ("interest_rate", "World interest rate"),
)


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    frames = {
        name: pd.read_parquet(out / f"{PREFIX}_{name}_test_df.parquet")
        for name, _, _ in SCENARIOS
    }

    style(figsize=(15, 15), fontsize=15, ticksize=13)
    figure, axes = plt.subplots(3, 2, figsize=(15, 15))
    for axis, (column, ylabel) in zip(axes.flat, PANELS, strict=True):
        for name, label, color in SCENARIOS:
            frame = frames[name]
            if column == "interest_rate":
                axis.plot(frame["t"], frame[column], color=color, label=label)
                continue
            for country, linestyle in ((1, "-"), (2, "--")):
                axis.plot(
                    frame["t"],
                    frame[f"{column}_{country}"],
                    color=color,
                    linestyle=linestyle,
                    label=label if country == 1 else "_nolegend_",
                )
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes.flat[0].plot([], [], color="black", linestyle="-", label="Country 1")
    axes.flat[0].plot([], [], color="black", linestyle="--", label="Country 2")
    axes.flat[0].legend()
    figure.tight_layout()
    path = out / "multicountry_growth_transitions.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
