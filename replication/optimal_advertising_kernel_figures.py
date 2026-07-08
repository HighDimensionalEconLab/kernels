from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "optimal_advertising_kernel"


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    results = pd.read_parquet(out / f"{PREFIX}_baseline_results.parquet")
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")
    train_T = float(results["train_T"].iloc[0])

    style(figsize=(15, 7), fontsize=17, ticksize=16)
    t = test_df["t"].to_numpy()

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(
        t,
        test_df["x"].to_numpy(),
        color="k",
        label=r"$\hat{x}(t)$:  Kernel Approximation",
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.ylabel(r"Market Share: $x(t)$")
    plt.xlabel("Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        t,
        test_df["mu"].to_numpy(),
        color="blue",
        label=r"$\hat{\mu}(t)$:  Kernel Approximation",
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.ylabel(r"Co-state Variable: $\mu(t)$")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()

    path = out / "optimal_advertising.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
