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
SUMMARY_EXPERIMENTS = {
    "baseline",
    "baseline_g02",
    "baseline_g02_learned",
    "sensitivity",
}


def run(
    output_dir: str = ".results",
    # Multi-start retries inside each solve keep acceptance high, so a small
    # seed ensemble suffices for the cross-seed medians.
    n_seeds: int = 7,
    verbose: bool = False,
):
    configure_process()

    experiments = {
        "baseline": {},
        "baseline_g02": {"g": 0.02, "tolerance_set": {"train_loss_threshold": 1e-14}},
        "baseline_g02_learned": {
            "g": 0.02,
            "nn_set": {"learn_g_hat": True},
            "tolerance_set": {"train_loss_threshold": 1e-14},
        },
    }
    for name, experiment in experiments.items():
        print(f"{name}: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds,
            options={
                "verbose": verbose,
                "k_0": 1.0,
                **experiment,
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            dict(n_seeds=n_seeds, residual_dtype="float64"),
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    # Explicit set: the prefix is a strict prefix of
    # neoclassical_growth_nn_recursive, whose JSONs share the glob.
    write_summary(output_dir, PREFIX, experiments=SUMMARY_EXPERIMENTS)


if __name__ == "__main__":
    jsonargparse.CLI(run)
