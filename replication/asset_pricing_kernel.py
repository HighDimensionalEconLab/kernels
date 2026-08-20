from __future__ import annotations

import jsonargparse

from models import asset_pricing_kernel
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

SOLVER = asset_pricing_kernel.asset_pricing_kernel
PREFIX = "asset_pricing_kernel"


def run(
    output_dir: str = ".results",
):
    configure_process()
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds=1,
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=1, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
