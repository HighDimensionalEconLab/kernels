from __future__ import annotations

import jax
import jsonargparse

from extra import neoclassical_growth_nn_recursive_adaptive
from replication.utilities import (
    BASIN_K0_GRID,
    enable_compilation_cache,
    ensemble,
    save_experiment,
    sweep,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_nn_recursive_adaptive"
SOLVER = (
    neoclassical_growth_nn_recursive_adaptive.neoclassical_growth_nn_recursive_adaptive
)
BUTTERFLY = dict(A=0.5, b_1=3.0, b_2=2.5)


def run(
    output_dir: str = ".results_extra",
    n_seeds: int = 100,
    use_float64: bool = True,
    verbose: bool = False,
):
    common = dict(use_float64=use_float64, verbose=verbose)
    meta = {
        "n_seeds": n_seeds,
        "residual_dtype": "float64" if use_float64 else "float32",
    }
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds,
        options={**common, "k_0": 1.0},
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
    results_df, frames, hyperparameters = sweep(
        SOLVER,
        "k_0",
        [float(k_0) for k_0 in BASIN_K0_GRID],
        n_seeds=n_seeds,
        options={**common, **BUTTERFLY},
    )
    save_experiment(
        "butterfly_basin",
        results_df,
        {"test_df": frames["test_df"]},
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
