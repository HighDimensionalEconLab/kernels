from __future__ import annotations

import jax
import jsonargparse

import neoclassical_growth_nn_recursive
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

PREFIX = "neoclassical_growth_nn_recursive"

SOLVER = neoclassical_growth_nn_recursive.neoclassical_growth_nn_recursive

BUTTERFLY = dict(A=0.5, b_1=3.0, b_2=2.5)


def run(
    output_dir: str = ".results",
    # Adaptive solves recompile for each independently constructed seed. Three
    # seeds preserve a robustness band without making the core pipeline a
    # 10-minute compilation sweep; larger ensembles belong in extra/sweeps.
    n_seeds: int = 3,
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

    # Butterfly basin-of-attraction outcomes over the shared k_0 grid
    # (settle-based classification inside the solver).
    print(f"butterfly_basin: {n_seeds} seeds x {len(BASIN_K0_GRID)} ICs")
    results_df, frames, hyperparameters = sweep(
        SOLVER,
        "k_0",
        [float(k_0) for k_0 in BASIN_K0_GRID],
        n_seeds=n_seeds,
        options={**common, **BUTTERFLY},
    )
    frac = results_df.groupby("k_0")["basin_correct"].mean()
    print(f"  P(correct) by k_0: min={frac.min():.2f} mean={frac.mean():.2f}")
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
