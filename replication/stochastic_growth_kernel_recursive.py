from __future__ import annotations

import jsonargparse

from models import stochastic_growth_kernel_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "stochastic_growth_kernel_recursive"

SOLVER = stochastic_growth_kernel_recursive.stochastic_growth_kernel_recursive

DIAGNOSTIC_COLUMNS = [
    "max_train_euler",
    "test_euler_mean_abs",
    "max_test_euler",
    "max_log_tvc",
    "test_mean_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "solve_time",
]


def report(results_df):
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())


# Deterministic distillation of the cached stochastic NN reference policy;
# run replication.stochastic_growth_nn_recursive first to populate the cache.
def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    configure_process()
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        1,
        options=dict(
            verbose=verbose,
            policy_dir=output_dir,
        ),
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=1, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    report(results_df)

    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
