from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import QUANTILES, style

PREFIX = "asset_pricing_nn"


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

    style(figsize=(15, 7), fontsize=17, ticksize=16)
    plt.figure(figsize=(15, 7))
    ax_price = plt.subplot(1, 2, 1)
    lo, mid, hi = band(test_df, "p")
    ax_price.plot(mid.index, mid.to_numpy(), color="C0", label="NN median")
    ax_price.fill_between(
        lo.index,
        lo.to_numpy(),
        hi.to_numpy(),
        color="C0",
        alpha=0.25,
        label="10th--90th percentile",
    )
    benchmark = test_df.groupby("t")["p_benchmark"].first()
    ax_price.plot(
        benchmark.index,
        benchmark.to_numpy(),
        linestyle="--",
        color="k",
        label="No-bubble solution",
    )
    ax_price.axvline(train_T, color="k", linestyle=":")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price")
    ax_price.legend()

    ax_error = plt.subplot(1, 2, 2)
    lo, mid, hi = band(test_df, "p_rel_error")
    ax_error.plot(mid.index, mid.to_numpy(), color="C0", label="NN median")
    ax_error.fill_between(
        lo.index, lo.to_numpy(), hi.to_numpy(), color="C0", alpha=0.25
    )
    ax_error.axvline(train_T, color="k", linestyle=":")
    ax_error.set_yscale("log")
    ax_error.set_xlabel("Time")
    ax_error.set_ylabel("Relative price error")
    ax_error.legend()

    plt.tight_layout()
    path = out / "asset_pricing_nn.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
