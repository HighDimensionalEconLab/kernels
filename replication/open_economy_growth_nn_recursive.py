from __future__ import annotations

import jsonargparse

from models import open_economy_growth_nn_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "open_economy_growth_nn_recursive"
SOLVER = open_economy_growth_nn_recursive.open_economy_growth_nn_recursive


def run(
    output_dir: str = ".results",
    n_seeds: int = 5,
    verbose: bool = False,
):
    configure_process()

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
            {
                "n_seeds": experiment_n_seeds,
                "residual_dtype": "float64",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )
        return results_df

    baseline = go("baseline", n_seeds)
    # Debt/asset experiments sized by the baseline run's own
    # steady-state output.
    output_ss = float(baseline["output_ss"].iloc[0])
    go("initial_debt", 1, a_0=-0.5 * output_ss)
    go("initial_assets", 1, a_0=0.5 * output_ss)
    write_summary(
        output_dir,
        PREFIX,
        experiments={"baseline", "initial_debt", "initial_assets"},
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
