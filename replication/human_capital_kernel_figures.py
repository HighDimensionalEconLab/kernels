from __future__ import annotations

import json
import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "human_capital_kernel"

PANELS = [
    ("k", "black", r"$\hat{x}_k(t)$", "Physical Capital: $x_k(t)$"),
    ("h", "black", r"$\hat{x}_h(t)$", "Human Capital: $x_h(t)$"),
    ("c", "tab:blue", r"$\hat{y}_c(t)$", "Consumption: $y_c(t)$"),
    ("i_k", "tab:blue", r"$\hat{y}_k(t)$", "Physical Capital Investment: $y_k(t)$"),
    ("i_h", "tab:blue", r"$\hat{y}_h(t)$", "Human Capital Investment: $y_h(t)$"),
]


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")
    hp = json.loads((out / f"{PREFIX}_baseline_hyperparameters.json").read_text())
    train_T = float(hp["data_set"]["train_T"])

    style(figsize=(15, 13), fontsize=17, ticksize=16)
    t = test_df["t"].to_numpy()
    plt.figure(figsize=(15, 13))
    for index, (column, color, label, ylabel) in enumerate(PANELS, start=1):
        plt.subplot(3, 2, index)
        plt.plot(t, test_df[column].to_numpy(), color=color, label=label)
        plt.axvline(x=train_T, color="grey", linestyle=":")
        plt.ylabel(ylabel)
        plt.xlabel("Time")
        plt.legend()

    # One panel for both co-states: the paths coincide, which is the algebraic
    # condition mu_k = mu_h holding along the solved trajectory.
    plt.subplot(3, 2, 6)
    plt.plot(t, test_df["mu_k"].to_numpy(), color="grey", label=r"$\hat{\mu}_k(t)$")
    plt.plot(
        t,
        test_df["mu_h"].to_numpy(),
        color="black",
        linestyle="--",
        label=r"$\hat{\mu}_h(t)$",
    )
    plt.axvline(x=train_T, color="grey", linestyle=":")
    plt.ylabel(r"Co-state Variables: $\mu_k(t), \mu_h(t)$")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()

    path = out / "neoclassical_human_capital.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
