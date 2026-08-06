from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replication.utilities import style

PREFIX = "stochastic_growth_kernel_recursive"


def run(output_dir: str = ".results", fontsize: int = 14, ticksize: int = 13):
    out = pathlib.Path(output_dir)
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")

    style(figsize=(15, 5), fontsize=fontsize, ticksize=ticksize)
    fig, (ax_paths, ax_error) = plt.subplots(1, 2)

    for _, group in test_df.groupby("trajectory"):
        ax_paths.plot(group["t"], group["k"], color="black", alpha=0.15, lw=0.8)
    ax_paths.set_xlabel("Time")
    ax_paths.set_ylabel("Capital: $k(t)$")
    ax_paths.set_title("Simulated capital under the kernel policy")

    # Common-noise pathwise gap to the reference NN policy, pooled across
    # trajectories at each saved step.
    quantiles = (
        test_df.assign(
            k_err=np.abs(test_df["k_rel_error"]), c_err=np.abs(test_df["c_rel_error"])
        )
        .groupby("t")[["k_err", "c_err"]]
        .quantile([0.5, 0.9])
        .unstack()
    )
    ax_error.plot(
        quantiles.index, quantiles[("k_err", 0.5)], color="black", label="$k$, median"
    )
    ax_error.plot(
        quantiles.index,
        quantiles[("k_err", 0.9)],
        color="black",
        ls="--",
        label="$k$, p90",
    )
    ax_error.plot(
        quantiles.index, quantiles[("c_err", 0.5)], color="tab:blue", label="$c$, median"
    )
    ax_error.plot(
        quantiles.index,
        quantiles[("c_err", 0.9)],
        color="tab:blue",
        ls="--",
        label="$c$, p90",
    )
    ax_error.set_yscale("log")
    ax_error.set_xlabel("Time")
    ax_error.set_ylabel("Relative gap to reference NN")
    ax_error.set_title("Common-noise pathwise agreement")
    ax_error.legend(frameon=False, ncol=2)

    fig.tight_layout()
    path = out / "stochastic_growth_paths.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
