from __future__ import annotations

import jax
import jsonargparse

import stochastic_growth_nn_recursive
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "stochastic_growth_nn_recursive"

SOLVER = stochastic_growth_nn_recursive.stochastic_growth_nn_recursive

DIAGNOSTIC_COLUMNS = [
    "train_loss",
    "test_euler_mean_abs",
    "max_test_euler",
    "max_log_tvc_residual",
    "test_mean_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "policy_mean_abs_c_rel_error",
    "steps",
    "train_time",
]


# High-variance TFP (Aruoba et al.'s sigma = 0.035 column) where the
# precautionary correction is visible against the FD benchmark.
def run(
    output_dir: str = ".results",
    n_seeds: int = 5,
    use_float64: bool = True,
    data_seed: int = 0,
    verbose: bool = False,
):
    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds,
        options=dict(
            use_float64=use_float64,
            verbose=verbose,
            ic_set={"data_seed": data_seed},
        ),
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        {
            "n_seeds": n_seeds,
            "residual_dtype": "float64" if use_float64 else "float32",
        },
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())

    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
