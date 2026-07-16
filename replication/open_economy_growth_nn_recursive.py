from __future__ import annotations

import jsonargparse

import open_economy_growth_nn_recursive
from open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

enable_compilation_cache()

PREFIX = "open_economy_growth_nn_recursive"
SOLVER = open_economy_growth_nn_recursive.open_economy_growth_nn_recursive


def run(
    output_dir: str = ".results",
    n_seeds: int = 3,
    verbose: bool = False,
):
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    meta = dict(n_seeds=n_seeds)

    def go(name, **experiment):
        print(f"{name}: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=n_seeds,
            options={"verbose": verbose, **experiment},
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            meta,
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("baseline")
    go("initial_debt", a_0=-0.5 * output_ss)
    go("initial_assets", a_0=0.5 * output_ss)
    go("above_steady_state", k_0=7.0)
    go(
        "short_diagnostic",
        data_set=dict(
            diagnostic_T=50.0,
            diagnostic_points=51,
            diagnostic_steps=50,
        ),
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
