from __future__ import annotations

import jsonargparse

from models import open_economy_growth_kernel_uno
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
        return results_df

    baseline = go("baseline")
    # Debt/asset experiments sized by the baseline run's own
    # steady-state output.
    output_ss = float(baseline["output_ss"].iloc[0])
    # The higher world rate steepens the paths; its between-node bar is looser.
    go("r_star_005", r_star=0.05, tolerance_set=dict(validation_residual=1e-3))
    go("initial_debt", a_0=-0.5 * output_ss)
    go("initial_assets", a_0=0.5 * output_ss)
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
