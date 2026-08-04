from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import QUANTILES, style

PREFIX = "neoclassical_growth_nn_recursive"


def quantile_band(df, value_column):
    grouped = df.groupby("kappa")[value_column]
    return (
        grouped.quantile(QUANTILES[0]),
        grouped.median(),
        grouped.quantile(QUANTILES[-1]),
    )


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)

    # --- learned homothetic policy vs the solve_bvp benchmark policy -----------
    policy_df = pd.read_parquet(out / f"{PREFIX}_single_ic_g0_policy_df.parquet")
    results = pd.read_parquet(out / f"{PREFIX}_single_ic_g0_results.parquet")
    ok_seeds = results.loc[results["solution_accepted"], "seed"]
    policy_df = policy_df[policy_df["seed"].isin(ok_seeds)]

    style(figsize=(15, 7), fontsize=14, ticksize=14)
    plt.figure(figsize=(15, 7))
    ax_c = plt.subplot(1, 2, 1)
    ax_dk = plt.subplot(1, 2, 2)
    for ax, col in ((ax_c, "c_tilde_nn"), (ax_dk, "dkappa_nn")):
        lo, mid, hi = quantile_band(policy_df, col)
        ax.plot(
            mid.index,
            mid.to_numpy(),
            color="C0",
            label=r"Recursive NN policy",
        )
        ax.fill_between(lo.index, lo.to_numpy(), hi.to_numpy(), color="C0", alpha=0.25)
    one_seed = policy_df[policy_df["seed"] == policy_df["seed"].iloc[0]]
    ax_c.plot(
        one_seed["kappa"],
        one_seed["c_tilde_benchmark"],
        linestyle="--",
        color="k",
        label="Benchmark policy",
    )
    ax_dk.plot(
        one_seed["kappa"],
        one_seed["dkappa_benchmark"],
        linestyle="--",
        color="k",
        label="Benchmark policy",
    )
    ax_c.set_xlabel(r"Policy input: $k/z$")
    ax_c.set_ylabel(r"Implied normalized consumption: $c/z$")
    ax_c.legend()
    ax_dk.set_xlabel(r"Policy input: $k/z$")
    ax_dk.set_ylabel(r"Implied ratio drift: $d(k/z)/dt$")
    ax_dk.axhline(0.0, color="0.5", linewidth=0.7)
    ax_dk.legend()
    plt.tight_layout()
    path = out / "neoclassical_growth_recursive_policy.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
