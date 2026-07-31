from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "open_economy_growth_nn"
SCENARIOS = (
    ("initial_debt", "Initial debt", "#b2182b"),
    ("baseline", "Zero initial NFA", "#2166ac"),
    ("initial_assets", "Initial assets", "#1b7837"),
)
PANELS = (
    ("k", "Capital"),
    ("q", "Installed-capital value"),
    ("c", "Consumption"),
    ("a", "Net foreign assets"),
    ("current_account", "Current account"),
    ("x", "Investment expenditure"),
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
            frame = frame.loc[frame["seed"] == 0]
            axis.plot(frame["t"], frame[column], color=color, label=label)
            axis.plot(
                frame["t"],
                frame[f"{column}_benchmark"],
                color=color,
                linestyle="--",
                label="_nolegend_",
            )
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes.flat[0].plot([], [], color="black", linestyle="--", label="BVP benchmark")
    axes.flat[0].legend()
    figure.tight_layout()
    path = out / "open_economy_growth_nn_transitions.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
