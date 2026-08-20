from __future__ import annotations

import jsonargparse

from models import stochastic_growth_nn_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

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
    "solve_time",
]


def report(results_df):
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())


# High-variance TFP (Aruoba et al.'s sigma = 0.035 column) where the
# precautionary correction is visible against the FD benchmark.
def run(
    output_dir: str = ".results",
    n_seeds: int = 5,
    data_seed: int = 0,
    verbose: bool = False,
):
    configure_process()
    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds,
        options=dict(
            verbose=verbose,
            ic_set={"data_seed": data_seed},
            # Cache each accepted policy so the kernel distillation driver can
            # load the best one as its frozen reference.
            policy_cache=output_dir,
        ),
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=n_seeds, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    report(results_df)

    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
