from __future__ import annotations

import jsonargparse

import asset_pricing_nn
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "asset_pricing_nn"


def run(
    output_dir: str | None = None,
    n_seeds: int = 11,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-9)
    results_df, frames, hyperparameters = ensemble(
        asset_pricing_nn.asset_pricing_nn,
        n_seeds=n_seeds,
        options={"use_float64": use_float64, "verbose": verbose, "stop_set": stop},
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        {
            "n_seeds": n_seeds,
            "residual_dtype": "float64" if use_float64 else "float32",
        },
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
