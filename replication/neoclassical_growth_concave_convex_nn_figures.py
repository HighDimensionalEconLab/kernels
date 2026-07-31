from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import style

PREFIX = "neoclassical_growth_concave_convex_nn"


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    results = pd.read_parquet(out / f"{PREFIX}_threshold_results.parquet")
    test_df = pd.read_parquet(out / f"{PREFIX}_threshold_test_df.parquet")

    # Unlike the kernel fan, a rejected NN solve is a first-class outcome here
    # (acceptance is the solver's, basin correctness is scored separately), so
    # drop rejects from the fan and say how many rather than aborting.
    accepted_k_0 = results.loc[results["solution_accepted"], "k_0"]
    dropped = len(results) - len(accepted_k_0)
    if dropped:
        print(
            f"{PREFIX}: {dropped}/{len(results)} rejected solves omitted from the fan"
        )
    test_df = test_df[test_df["k_0"].isin(accepted_k_0)]

    style(figsize=(15, 8), fontsize=17, ticksize=16)
    train_T = float(results["train_T"].iloc[0])
    plt.figure(figsize=(15, 8))

    ax_k = plt.subplot(1, 2, 1)
    for _, group in test_df.groupby("k_0"):
        group = group.sort_values("t")
        ax_k.plot(group["t"], group["k"], color="gray")
    ax_k.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    ax_k.set_ylabel("Capital: $x(t)$")
    ax_k.set_xlabel("Time")
    ax_k.legend()

    ax_c = plt.subplot(1, 2, 2)
    for _, group in test_df.groupby("k_0"):
        group = group.sort_values("t")
        ax_c.plot(group["t"], group["c"], color="b")
    ax_c.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    ax_c.set_ylabel("Consumption: $y(t)$")
    ax_c.set_xlabel("Time")
    ax_c.legend()

    plt.tight_layout()
    path = out / "neoclassical_growth_model_concave_convex_threshold_nn.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
