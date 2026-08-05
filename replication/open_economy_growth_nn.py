from __future__ import annotations

import jax
import jsonargparse

import open_economy_growth_nn
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

PREFIX = "open_economy_growth_nn"
SOLVER = open_economy_growth_nn.open_economy_growth_nn


def run(
    output_dir: str = ".results",
    n_seeds: int = 11,
    use_float64: bool = True,
    verbose: bool = False,
):
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])

    def go(name, experiment_n_seeds, **experiment):
        print(f"{name}: {experiment_n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=experiment_n_seeds,
            options={
                "use_float64": use_float64,
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
                "residual_dtype": "float64" if use_float64 else "float32",
            },
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
