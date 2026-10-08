from __future__ import annotations

import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replication.utilities import style

PREFIX = "neoclassical_growth_kernel_ridge_grid"
PANELS = (
    ("test_k_rel_error_abs", r"Mean rel. error of $\hat{x}(t)$ on the test grid"),
    ("max_train_residual", "Max residual on the training dates"),
    (
        "rkhs_norm_squared",
        r"$\sum_j \boldsymbol{\alpha}_j^{\top} \mathbf{K} \boldsymbol{\alpha}_j$",
    ),
)


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    results = pd.read_parquet(out / f"{PREFIX}_results.parquet")
    medians = results.groupby(["train_points", "ridge"]).median(numeric_only=True)
    counts = sorted(results["train_points"].unique())
    ridges = np.array(sorted(results["ridge"].unique()))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(counts)))

    style(figsize=(16, 5))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (column, title) in zip(axes, PANELS):
        for color, n in zip(colors, counts):
            axis.loglog(
                ridges,
                [medians.loc[(n, ridge), column] for ridge in ridges],
                color=color,
                marker="o",
                markersize=5,
                linewidth=1.8,
                label=f"$N = {n}$",
            )
        axis.set_xlabel(r"Ridge weight $\lambda$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    # Slope-one reference through the largest-N residual at the largest ridge.
    anchor = medians.loc[(counts[-1], ridges[-1]), "max_train_residual"]
    axes[1].loglog(
        ridges,
        anchor * ridges / ridges[-1],
        color="k",
        linestyle=":",
        linewidth=1.2,
        label=r"slope $1$",
    )
    axes[0].legend(fontsize=11)
    axes[1].legend(fontsize=11, loc="lower right")
    figure.tight_layout()
    path = out / f"{PREFIX}.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
