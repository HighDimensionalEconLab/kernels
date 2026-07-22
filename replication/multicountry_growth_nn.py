from __future__ import annotations

import jax
import jsonargparse

import multicountry_growth_nn
from multicountry_growth import MulticountryParameters, multicountry_output_scale
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn


def run(
    output_dir: str = ".results",
    n_seeds: int = 100,
    use_float64: bool = True,
    verbose: bool = False,
):
    parameters = MulticountryParameters()
    output_scale = float(multicountry_output_scale(parameters))
    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds=n_seeds,
        options={
            "a_0": (-0.5 * output_scale, 0.5 * output_scale),
            "use_float64": use_float64,
            "verbose": verbose,
        },
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        {
            "n_seeds": n_seeds,
            "scenario": "two_country_initial_nfa",
            "residual_dtype": "float64" if use_float64 else "float32",
        },
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
