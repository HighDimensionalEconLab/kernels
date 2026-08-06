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
    "max_log_tvc_residual",
    "test_mean_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "solve_time",
]


# Deterministic distillation of the cached stochastic NN reference policy;
# run replication.stochastic_growth_nn_recursive first to populate the cache.
def run(
    output_dir: str | None = None,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        1,
        options=dict(
            use_float64=use_float64,
            verbose=verbose,
            policy_dir=output_dir,
        ),
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        {"residual_dtype": "float64" if use_float64 else "float32"},
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())

    write_summary(output_dir, PREFIX, experiments={"baseline"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
