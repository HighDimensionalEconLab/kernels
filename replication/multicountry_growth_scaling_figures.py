from __future__ import annotations

import json
import pathlib

import jsonargparse
import matplotlib.pyplot as plt

from replication.utilities import style

PREFIX = "multicountry_growth_scaling"
SERIES = (
    ("dense", "Dense Cholesky", "#b2182b", "o"),
    ("cg", "Preconditioned CG", "#2166ac", "s"),
)


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    summary = json.loads((out / f"{PREFIX}_summary.json").read_text())

    # Collapsed constants are plain floats in the ensemble JSONs; spread
    # statistics arrive as {median, ...} blocks.
    def med(block):
        return block["median"] if isinstance(block, dict) else block

    series: dict[str, dict[int, float]] = {solver: {} for solver, *_ in SERIES}
    for name, experiment in summary.items():
        solver, count = name.rsplit("_N", 1)
        series[solver][int(count)] = float(med(experiment["solve_time"]))

    style(figsize=(9, 6))
    figure, axis = plt.subplots(figsize=(9, 6))
    for solver, label, color, marker in SERIES:
        counts = sorted(series[solver])
        axis.loglog(
            counts,
            [series[solver][count] for count in counts],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            label=label,
        )
    axis.set_xlabel("Countries $N$")
    axis.set_ylabel("Solve time (seconds)")
    all_counts = sorted({count for values in series.values() for count in values})
    axis.set_xticks(all_counts)
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.get_xaxis().set_minor_formatter(plt.NullFormatter())
    axis.tick_params(axis="x", which="minor", bottom=False)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = out / f"{PREFIX}.pdf"
    figure.savefig(path, format="pdf")
    plt.close(figure)
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
