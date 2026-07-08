from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from replication.utilities import add_zoom_inset, style

PREFIX = "asset_pricing_kernel"


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    results = pd.read_parquet(out / f"{PREFIX}_baseline_results.parquet")
    test_df = pd.read_parquet(out / f"{PREFIX}_baseline_test_df.parquet")
    train_T = float(results["train_T"].iloc[0])

    style(figsize=(15, 7), fontsize=17, ticksize=16)
    t = test_df["t"].to_numpy()
    p_hat = test_df["p"].to_numpy()
    p_benchmark = test_df["p_benchmark"].to_numpy()

    plt.figure(figsize=(15, 7))
    ax_prices = plt.subplot(1, 2, 1)
    plt.plot(t, p_hat, color="k", label=r"$\hat{\mu}(t)$: Kernel Approximation")
    plt.plot(
        t,
        p_benchmark,
        linestyle="--",
        color="k",
        label=r"$\mu_f(t)$: Closed-Form Solution",
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.ylabel(r"Price: $\mu(t)$")
    plt.xlabel("Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        t,
        test_df["p_rel_error"].to_numpy(),
        color="k",
        label=r"$\varepsilon_{\mu}(t)$: Relative Errors",
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.yscale("log")
    plt.ylim(1e-5, 2e-2)
    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    add_zoom_inset(
        ax_prices,
        t,
        [(p_hat, dict(color="k")), (p_benchmark, dict(linestyle="--", color="k"))],
        p_benchmark,
        (85, 95),
    )
    path = out / "asset_pricing_contiguous.pdf"
    plt.savefig(path, format="pdf")
    plt.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
