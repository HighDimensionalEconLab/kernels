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
    g: float = 0.0,
    z_0: float | list[float] = 1.0,
    k_0: float | list[float] = 1.0,
    num_test_trajectories: int = 1,
    test_log_variance: float = 0.0,
    test_T: float = 40.0,
    test_points: int = 41,
    data_seed: int = 0,
    verbose: bool = False,
):
    common = dict(
        use_float64=use_float64,
        verbose=verbose,
        g=g,
        z_0=z_0,
        k_0=k_0,
        data_set={
            "num_test_trajectories": num_test_trajectories,
            "test_T": test_T,
            "test_points": test_points,
        },
        ic_set={
            "data_seed": data_seed,
            "test_log_variance": test_log_variance,
        },
    )
    meta = dict(
        n_seeds=n_seeds,
        residual_dtype="float64" if use_float64 else "float32",
    )

    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(SOLVER, n_seeds, options=common)
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
