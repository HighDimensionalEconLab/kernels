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

# The sensitivity experiment lives in neoclassical_growth_nn_sensitivity.py;
# both drivers regenerate the shared summary from whatever is on disk.
SUMMARY_EXPERIMENTS = {"baseline", "sensitivity"}


def run(
    output_dir: str = ".results",
    # Multi-start retries inside each solve keep acceptance high, so a small
    # seed ensemble suffices for the cross-seed medians.
    n_seeds: int = 7,
    verbose: bool = False,
):
    configure_process()
    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds,
        options={
            "verbose": verbose,
            "k_0": 1.0,
        },
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        {
            "n_seeds": n_seeds,
            "residual_dtype": "float64",
        },
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )

    # Explicit set: the prefix is a strict prefix of
    # neoclassical_growth_nn_recursive, whose JSONs share the glob.
    write_summary(output_dir, PREFIX, experiments=SUMMARY_EXPERIMENTS)


if __name__ == "__main__":
    jsonargparse.CLI(run)
