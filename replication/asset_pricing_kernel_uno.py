from __future__ import annotations

import jsonargparse

from models import asset_pricing_kernel_uno
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "asset_pricing_kernel_uno"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    configure_process()
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        asset_pricing_kernel_uno.asset_pricing_kernel_uno,
        n_seeds=1,
        options={"verbose": verbose},
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=1),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
