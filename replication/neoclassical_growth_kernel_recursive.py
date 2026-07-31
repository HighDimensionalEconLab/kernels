from __future__ import annotations

import jax
import jsonargparse

import neoclassical_growth_kernel_recursive
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_kernel_recursive"

SOLVER = neoclassical_growth_kernel_recursive.neoclassical_growth_kernel_recursive


def run(output_dir: str = ".results", use_float64: bool = True, verbose: bool = False):
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")

    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds=1,
        options={"use_float64": use_float64, "verbose": verbose},
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
