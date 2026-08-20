from __future__ import annotations

import jsonargparse

from models import open_economy_growth_kernel_uno
from models.open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

SOLVER = open_economy_growth_kernel_uno.open_economy_growth_kernel_uno
PREFIX = "open_economy_growth_kernel_uno"


def run(output_dir: str = ".results", verbose: bool = False):
    configure_process()
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    meta = dict(n_seeds=1, residual_dtype="float64")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
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
    go("r_star_005", parameters=dict(r_star=0.05))
    go("initial_debt", a_0=-0.5 * output_ss)
    go("initial_assets", a_0=0.5 * output_ss)
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
