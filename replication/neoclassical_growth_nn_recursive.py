from __future__ import annotations

import jax
import jsonargparse

import neoclassical_growth_nn_recursive
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_nn_recursive"

SOLVER = neoclassical_growth_nn_recursive.neoclassical_growth_nn_recursive


def run(
    output_dir: str = ".results",
    n_seeds: int = 100,
    use_float64: bool = True,
    verbose: bool = False,
):
    common = dict(use_float64=use_float64, verbose=verbose)
    meta = dict(
        n_seeds=n_seeds,
        residual_dtype="float64" if use_float64 else "float32",
    )

    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER, n_seeds, options={**common, "k_0": 1.0}
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )

    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
