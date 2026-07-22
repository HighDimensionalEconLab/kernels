from __future__ import annotations

import jax
import jsonargparse

import open_economy_growth_kernel_uno
from open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "open_economy_growth_kernel_uno"


def run(output_dir: str = ".results", verbose: bool = False):
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    meta = dict(n_seeds=1, residual_dtype="float64")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            open_economy_growth_kernel_uno.open_economy_growth_kernel_uno,
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
