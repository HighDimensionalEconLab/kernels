from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "human_capital_kernel"

PANELS = [
    ("k", "k", r"$\hat{x}_k(t)$: Kernel Approximation", "Physical Capital: $x_k(t)$"),
    ("h", "k", r"$\hat{x}_h(t)$: Kernel Approximation", "Human Capital: $x_h(t)$"),
    ("c", "b", r"$\hat{y}_c(t)$: Kernel Approximation", "Consumption: $y_c(t)$"),
    (
        "i_k",
        "b",
        r"$\hat{y}_k(t)$: Kernel Approximation",
        "Physical Capital Investment: $y_k(t)$",
    ),
    (
        "i_h",
        "b",
        r"$\hat{y}_h(t)$: Kernel Approximation",
        "Human Capital Investment: $y_h(t)$",
    ),
    (
        "mu_k",
        "grey",
        r"$\hat{\mu}_k(t)$: Kernel Approximation",
        r"Co-state Variable: $\mu_k(t)$",
    ),
    (
        "mu_h",
        "grey",
        r"$\hat{\mu}_h(t)$: Kernel Approximation",
        r"Co-state Variable: $\mu_h(t)$",
    ),
]


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    results = pd.read_parquet(out / f"{PREFIX}_baseline_results.parquet")
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")
    train_T = float(results["train_T"].iloc[0])

    style(figsize=(15, 18), fontsize=17, ticksize=16)
    t = test_df["t"].to_numpy()
    plt.figure(figsize=(15, 18))
    for index, (column, color, label, ylabel) in enumerate(PANELS, start=1):
        plt.subplot(4, 2, index)
        plt.plot(t, test_df[column].to_numpy(), color=color, label=label)
        plt.axvline(
            x=train_T, color=color, linestyle=":", label="Extrapolation/Interpolation"
        )
        plt.ylabel(ylabel)
        plt.xlabel("Time")
        plt.legend()
    plt.tight_layout()

    path = out / "neoclassical_human_capital.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
