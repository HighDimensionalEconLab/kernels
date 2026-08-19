from __future__ import annotations

import jsonargparse

from models import stochastic_open_economy_growth_kernel_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "stochastic_open_economy_growth_kernel_recursive"

SOLVER = stochastic_open_economy_growth_kernel_recursive.stochastic_open_economy_growth_kernel_recursive

DIAGNOSTIC_COLUMNS = [
    "train_euler_mean_abs",
    "test_euler_mean_abs",
    "test_euler_max_abs",
    "test_c_euler_mean_abs",
    "test_q_euler_mean_abs",
    "solve_time",
]


# Deterministic distillation of the cached stochastic NN reference policy;
# run replication.stochastic_open_economy_growth_nn_recursive first to
# populate the cache.
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
        {"residual_dtype": "float64"},
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())

    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
