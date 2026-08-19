from __future__ import annotations

import jsonargparse

from models import multicountry_growth_nn
from replication.multicountry_growth_kernel import LARGE_COUNTRY_PARAMETERS
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn


def run(
    output_dir: str = ".results",
    n_seeds: int = 7,
    verbose: bool = False,
):
    configure_process()

    def go(name, scenario, **options):
        print(f"{name}: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=n_seeds,
            options={
                "verbose": verbose,
                **options,
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            {
                "n_seeds": n_seeds,
                "scenario": scenario,
                "residual_dtype": "float64",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("baseline", "two_country_drawn_initial_conditions")
    go(
        "large_countries",
        "ten_country_drawn_initial_conditions",
        parameters=LARGE_COUNTRY_PARAMETERS,
    )
    write_summary(output_dir, PREFIX, experiments={"baseline", "large_countries"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
