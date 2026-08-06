from __future__ import annotations

import jsonargparse

from models import neoclassical_growth_nn
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
    n_seeds: int = 7,
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

    # Implicit-AD sensitivity experiment for the supplement's sensitivity
    # appendix: train to T = 120 and differentiate k(T), c(T) through a
    # warm-started re-solve; the exact steady-state derivatives ride along
    # as the comparison target. Float64 only: the float32 noise floor
    # swamps that comparison.
    if use_float64:
        print(f"sensitivity: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds,
            options={
                "use_float64": use_float64,
                "verbose": verbose,
                "k_0": 1.0,
                "stop_set": stop,
                "data_set": dict(train_T=120.0, train_points=121),
                "sensitivity_times": [120.0],
            },
        )
        save_experiment(
            "sensitivity",
            results_df,
            frames,
            output_dir,
            {"n_seeds": n_seeds, "residual_dtype": "float64"},
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    # Explicit set: the prefix is a strict prefix of
    # neoclassical_growth_nn_recursive, whose JSONs share the glob.
    write_summary(output_dir, PREFIX, experiments={"baseline", "sensitivity"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
