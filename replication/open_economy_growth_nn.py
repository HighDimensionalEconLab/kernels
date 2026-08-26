from __future__ import annotations

import jsonargparse

from models import open_economy_growth_nn
from models.open_economy_growth_shared import (
    OPEN_ECONOMY_GROWTH_CONSTANTS,
    steady_state,
)
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "open_economy_growth_nn"
SOLVER = open_economy_growth_nn.open_economy_growth_nn


def run(
    output_dir: str = ".results",
    n_seeds: int = 7,
    verbose: bool = False,
):
    configure_process()
    output_ss = float(steady_state(OPEN_ECONOMY_GROWTH_CONSTANTS)["output"])

    def go(name, experiment_n_seeds, **experiment):
        print(f"{name}: {experiment_n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=experiment_n_seeds,
            options={
                "verbose": verbose,
                **experiment,
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            dict(n_seeds=experiment_n_seeds, residual_dtype="float64"),
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("baseline", n_seeds)
    go("initial_debt", 1, a_0=-0.5 * output_ss)
    go("initial_assets", 1, a_0=0.5 * output_ss)

    # Explicit set: the prefix is a strict prefix of
    # open_economy_growth_nn_recursive, whose JSONs share the glob.
    write_summary(
        output_dir,
        PREFIX,
        experiments={"baseline", "initial_debt", "initial_assets"},
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
