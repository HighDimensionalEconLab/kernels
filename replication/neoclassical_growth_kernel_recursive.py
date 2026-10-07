from __future__ import annotations

import jsonargparse

from models import neoclassical_growth_kernel_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    sweep,
    write_summary,
)

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
    "max_log_tvc",
    "test_mean_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "steps",
    "solve_time",
]


def report(results_df):
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())


def run(
    output_dir: str = ".results",
    n_data_seeds: int = 1,
    experiments: list[str] | None = None,
    verbose: bool = False,
):
    configure_process()
    meta = dict(n_data_seeds=n_data_seeds, residual_dtype="float64")

    all_experiments = {
        "multi_ic_g0": (0.0, MULTI_IC),
        "multi_ic_g02": (0.02, MULTI_IC),
    }
    run_single = experiments is None or "single_ic_g0" in experiments
    if experiments is not None:
        unknown = set(experiments) - set(all_experiments) - {"single_ic_g0"}
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        all_experiments = {
            name: spec for name, spec in all_experiments.items() if name in experiments
        }

    # The kernel is deterministic given its data draw: the single-IC case runs
    # through the timing harness, while the multi-IC cases sweep the data seed.
    if run_single:
        print(f"single_ic_g0 -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            1,
            options={
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
        report(results_df)

    for name, (g, ic_set) in all_experiments.items():
        print(f"{name}: {n_data_seeds} data seeds -> {output_dir}/")
        results_df, frames, hyperparameters = sweep(
            SOLVER,
            "ic_set.data_seed",
            range(n_data_seeds),
            options={
                "verbose": verbose,
                "g": g,
                "ic_set": dict(ic_set),
                "lm_set": {"max_steps": 60},
            },
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
        report(results_df)

    write_summary(
        output_dir,
        PREFIX,
        experiments={"single_ic_g0", *all_experiments}
        if run_single
        else set(all_experiments),
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
