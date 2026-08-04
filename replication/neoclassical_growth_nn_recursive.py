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

SINGLE_IC = {
    "num_train_trajectories": 1,
    "num_test_trajectories": 1,
    "train_log_z_variance": 0.0,
    "train_log_k_variance": 0.0,
    "test_log_z_variance": 0.0,
    "test_log_k_variance": 0.0,
}
MULTI_IC = {
    "num_train_trajectories": 5,
    "num_test_trajectories": 10,
    "train_log_z_variance": 0.01,
    "train_log_k_variance": 0.01,
    "test_log_z_variance": 0.0025,
    "test_log_k_variance": 0.0025,
}
DIAGNOSTIC_COLUMNS = [
    "max_train_euler",
    "max_test_euler",
    "max_log_tvc_residual",
    "test_mean_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "steps",
    "train_time",
]


def run(
    output_dir: str = ".results",
    n_seeds: int = 100,
    use_float64: bool = True,
    test_T: float = 40.0,
    test_points: int = 41,
    data_seed: int = 0,
    verbose: bool = False,
):
    def options(g, ic_set):
        return dict(
            use_float64=use_float64,
            verbose=verbose,
            g=g,
            data_set={"test_T": test_T, "test_points": test_points},
            ic_set={"data_seed": data_seed, **ic_set},
        )

    experiments = {
        "single_ic_g0": options(0.0, SINGLE_IC),
        "multi_ic_g0": options(0.0, MULTI_IC),
        "multi_ic_g02": options(0.02, MULTI_IC),
    }
    meta = dict(
        n_seeds=n_seeds,
        residual_dtype="float64" if use_float64 else "float32",
    )
    for name, experiment_options in experiments.items():
        print(f"{name}: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER, n_seeds, options=experiment_options
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
        accepted = int(results_df["solution_accepted"].sum())
        print(f"  accepted {accepted}/{len(results_df)}")
        print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())

    write_summary(output_dir, PREFIX, experiments=set(experiments))


if __name__ == "__main__":
    jsonargparse.CLI(run)
