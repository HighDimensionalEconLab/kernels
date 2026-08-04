from __future__ import annotations

import jax
import jsonargparse

import neoclassical_growth_kernel_recursive
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    sweep,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_kernel_recursive"

SOLVER = neoclassical_growth_kernel_recursive.neoclassical_growth_kernel_recursive

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
    "solve_time",
]


def report(name, results_df):
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())


def run(
    output_dir: str = ".results",
    n_data_seeds: int = 5,
    use_float64: bool = True,
    verbose: bool = False,
):
    meta = dict(
        n_data_seeds=n_data_seeds,
        residual_dtype="float64" if use_float64 else "float32",
    )

    # The kernel is deterministic given its data draw: the single-IC case is
    # timed over repeated solves, while the multi-IC cases sweep the data seed
    # so the sensitivity to the drawn initial conditions is measured.
    print(f"single_ic_g0 -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        1,
        options={
            "use_float64": use_float64,
            "verbose": verbose,
            "g": 0.0,
            "ic_set": {"data_seed": 0, **SINGLE_IC},
        },
    )
    save_experiment(
        "single_ic_g0",
        results_df,
        frames,
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    report("single_ic_g0", results_df)

    for name, g in (("multi_ic_g0", 0.0), ("multi_ic_g02", 0.02)):
        print(f"{name}: {n_data_seeds} data seeds -> {output_dir}/")
        # Converging solves finish in ~20 LM steps, so fail fast on the base
        # kernel geometry and retry a data seed under the longer, more
        # regularized geometry; each row records the geometry that produced it.
        results_df, frames, hyperparameters = sweep(
            SOLVER,
            "ic_set.data_seed",
            range(n_data_seeds),
            options={
                "use_float64": use_float64,
                "verbose": verbose,
                "g": g,
                "ic_set": dict(MULTI_IC),
                "stop_set": {"max_steps": 60},
            },
            retries=({"kernel_set": {"lengthscale": 0.6, "metric_epsilon": 3e-4}},),
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
        report(name, results_df)

    write_summary(
        output_dir, PREFIX, experiments={"single_ic_g0", "multi_ic_g0", "multi_ic_g02"}
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
