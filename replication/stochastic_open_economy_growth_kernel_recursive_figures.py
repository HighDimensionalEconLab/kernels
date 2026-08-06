from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replication.utilities import style

PREFIX = "stochastic_open_economy_growth_kernel_recursive"

PANELS = [
    ("k", "Capital"),
    ("a", "Net foreign assets"),
    ("c", "Consumption"),
    ("q", "Installed-capital value"),
    ("interest_rate", "World interest rate $r^{\\star}(t)$"),
]


def run(output_dir: str = ".results", fontsize: int = 14, ticksize: int = 13):
    out = pathlib.Path(output_dir)
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")

    style(figsize=(15, 10), fontsize=fontsize, ticksize=ticksize)
    fig, axes = plt.subplots(2, 3)
    axes = axes.ravel()

    for ax, (column, label) in zip(axes, PANELS):
        for _, group in test_df.groupby("trajectory"):
            ax.plot(group["t"], group[column], color="black", alpha=0.12, lw=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel(label)

    ax_resid = axes[-1]
    pooled = (
        test_df.assign(
            c_resid=np.abs(test_df["c_euler"]), q_resid=np.abs(test_df["q_euler"])
        )
        .groupby("t")[["c_resid", "q_resid"]]
        .quantile([0.5, 0.9])
        .unstack()
    )
    ax_resid.plot(
        pooled.index,
        pooled[("c_resid", 0.5)],
        color="black",
        label="$\\mathcal{R}^{c}$, median",
    )
    ax_resid.plot(
        pooled.index,
        pooled[("c_resid", 0.9)],
        color="black",
        ls="--",
        label="$\\mathcal{R}^{c}$, p90",
    )
    ax_resid.plot(
        pooled.index,
        pooled[("q_resid", 0.5)],
        color="tab:blue",
        label="$\\mathcal{R}^{q}$, median",
    )
    ax_resid.plot(
        pooled.index,
        pooled[("q_resid", 0.9)],
        color="tab:blue",
        ls="--",
        label="$\\mathcal{R}^{q}$, p90",
    )
    ax_resid.set_yscale("log")
    ax_resid.set_xlabel("Time")
    ax_resid.set_ylabel("$|$Euler residual$|$")
    ax_resid.legend(frameon=False, ncol=2)

    fig.tight_layout()
    path = out / "stochastic_open_economy_paths.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
