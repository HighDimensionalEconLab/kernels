from __future__ import annotations

import jsonargparse

import neoclassical_growth_nn
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "neoclassical_growth_nn"

SOLVER = neoclassical_growth_nn.neoclassical_growth_nn


def run(
    output_dir: str | None = None,
    # Multi-start retries inside each solve keep acceptance high, so a small
    # seed ensemble suffices for the cross-seed medians.
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
    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds,
        options={
            "use_float64": use_float64,
            "verbose": verbose,
            "k_0": 1.0,
            "stop_set": stop,
        },
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

    # Explicit set: the prefix is a strict prefix of
    # neoclassical_growth_nn_recursive, whose JSONs share the glob.
    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
