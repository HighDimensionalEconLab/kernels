from __future__ import annotations

import jsonargparse

from models import asset_pricing_nn
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

SOLVER = asset_pricing_nn.asset_pricing_nn
PREFIX = "asset_pricing_nn"


def run(
    output_dir: str = ".results",
    n_seeds: int = 7,
    verbose: bool = False,
):
    configure_process()
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds=n_seeds,
        options={"verbose": verbose},
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=n_seeds, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
