from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import QUANTILES, style

PREFIX = "neoclassical_growth_nn"


def band(frame, column):
    grouped = frame.groupby("t")[column]
    return (
        grouped.quantile(QUANTILES[0]),
        grouped.median(),
        grouped.quantile(QUANTILES[-1]),
    )


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    results = pd.read_parquet(out / f"{PREFIX}_baseline_results.parquet")
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")
    accepted = results.loc[results["solution_accepted"], "seed"]
    test_df = test_df[test_df["seed"].isin(accepted)]
    train_T = float(results["train_T"].iloc[0])

    style(figsize=(15, 10), fontsize=14, ticksize=14)
    plt.figure(figsize=(15, 10))
    paths = (
        (1, "k", "k_benchmark", "C0", "Capital: $x(t)$"),
        (3, "c", "c_benchmark", "C1", "Consumption: $y(t)$"),
    )
    for position, column, benchmark_column, color, label in paths:
        ax = plt.subplot(2, 2, position)
        lo, mid, hi = band(test_df, column)
        ax.plot(mid.index, mid.to_numpy(), color=color, label="NN median")
        ax.fill_between(
            lo.index,
            lo.to_numpy(),
            hi.to_numpy(),
            color=color,
            alpha=0.25,
            label="10th--90th percentile",
        )
        benchmark = test_df.groupby("t")[benchmark_column].first()
        ax.plot(
            benchmark.index,
            benchmark.to_numpy(),
            linestyle="--",
            color="k",
            label="Benchmark solution",
        )
        ax.axvline(
            train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.legend()

    errors = (
        (2, "k_rel_error", "C0", r"$\varepsilon_x(t)$: Relative errors for $x(t)$"),
        (4, "c_rel_error", "C1", r"$\varepsilon_y(t)$: Relative errors for $y(t)$"),
    )
    for position, column, color, label in errors:
        ax = plt.subplot(2, 2, position)
        lo, mid, hi = band(test_df, column)
        ax.plot(mid.index, mid.to_numpy(), color=color, label="NN median")
        ax.fill_between(
            lo.index,
            lo.to_numpy(),
            hi.to_numpy(),
            color=color,
            alpha=0.25,
            label="10th--90th percentile",
        )
        ax.axvline(
            train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.legend()

    plt.tight_layout()
    path = out / "neoclassical_growth_nn.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
