from __future__ import annotations

import jsonargparse

import asset_pricing_kernel
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

enable_compilation_cache()

PREFIX = "asset_pricing_kernel"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        asset_pricing_kernel.asset_pricing_kernel,
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
