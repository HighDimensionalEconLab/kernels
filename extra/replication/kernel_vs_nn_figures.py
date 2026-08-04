from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "kernel_vs_nn"

ERROR_SERIES = [
    # (experiment, points column, error column, label, color)
    (
        "kernel_points_sweep",
        "train_points",
        "k_rel_error_mean",
        "Kernel (ridgeless RKHS)",
        "k",
    ),
    (
        "nn_direct_points_sweep",
        "train_points",
        "test_mean_abs_k_rel_error",
        "NN: direct MLP",
        "C0",
    ),
    (
        "nn_integrated_mlp_points_sweep",
        "train_points",
        "test_mean_abs_k_rel_error",
        "NN: integrated MLP",
        "C1",
    ),
    (
        "nn_integrated_rff_points_sweep",
        "train_points",
        "test_mean_abs_k_rel_error",
        "NN: integrated RFF",
        "C2",
    ),
]

BASIN_SERIES = [
    # (prefix, experiment, label, color)
    (PREFIX, "basin_kernel", "Kernel (ridgeless RKHS)", "k"),
    (PREFIX, "basin_nn_direct", "NN: direct MLP", "C0"),
]


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)

    # --- accuracy vs number of training points --------------------------------
    style(figsize=(15, 7), fontsize=14, ticksize=14)
    plt.figure(figsize=(15, 7))
    ax = plt.gca()
    for experiment, points_column, error_column, label, color in ERROR_SERIES:
        results = pd.read_parquet(out / f"{PREFIX}_{experiment}_results.parquet")
        accepted = results[results["solution_accepted"]]
        median_error = accepted.groupby(points_column)[error_column].median()
        ax.plot(
            median_error.index,
            median_error.to_numpy(),
            marker="o",
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of training points on $[0, 40]$")
    ax.set_ylabel(r"Median $E(|\varepsilon_x(t)|)$ over accepted seeds")
    ax.legend()
    plt.tight_layout()
    path = out / "kernel_vs_nn_errors.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")

    # --- basin of attraction overlay -------------------------------------------
    style(figsize=(15, 7), fontsize=14, ticksize=14)
    plt.figure(figsize=(15, 7))
    ax = plt.gca()
    k_thr = None
    for prefix, experiment, label, color in BASIN_SERIES:
        results = pd.read_parquet(out / f"{prefix}_{experiment}_results.parquet")
        frac = results.groupby("k_0")["basin_correct"].mean()
        ax.plot(frac.index, frac.to_numpy(), marker="o", color=color, label=label)
        k_thr = float(results["k_thr"].iloc[0])
    ax.axvline(x=k_thr, color="k", linestyle=":", label=r"Threshold $\bar{k}$")
    ax.set_xlabel("Initial capital: $k_0$")
    ax.set_ylabel("Fraction converging to the correct basin")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    path = out / "kernel_vs_nn_basin.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
