from __future__ import annotations

import jsonargparse

from models import neoclassical_growth_nn_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

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
    "solve_time",
]


def report(results_df):
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())


def run(
    output_dir: str = ".results",
    # Multi-start retries inside each solve keep acceptance high, so a small
    # seed ensemble suffices for the cross-seed medians.
    n_seeds: int = 5,
    test_T: float = 40.0,
    test_points: int = 41,
    data_seed: int = 0,
    experiments: list[str] | None = None,
    verbose: bool = False,
):
    configure_process()

    def options(g, ic_set):
        return dict(
            verbose=verbose,
            g=g,
            data_set={"test_T": test_T, "test_points": test_points},
            ic_set={"data_seed": data_seed, **ic_set},
        )

    all_experiments = {
        "single_ic_g0": (options(0.0, SINGLE_IC), n_seeds),
        "multi_ic_g0": (options(0.0, MULTI_IC), n_seeds),
        "multi_ic_g02": (options(0.02, MULTI_IC), n_seeds),
    }
    if experiments is not None:
        unknown = set(experiments) - set(all_experiments)
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        all_experiments = {name: all_experiments[name] for name in experiments}

    meta = dict(n_seeds=n_seeds, residual_dtype="float64")
    for name, (experiment_options, experiment_seeds) in all_experiments.items():
        print(f"{name}: {experiment_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER, experiment_seeds, options=experiment_options
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            {**meta, "n_seeds": experiment_seeds},
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )
        report(results_df)

    write_summary(output_dir, PREFIX, experiments=set(all_experiments))


if __name__ == "__main__":
    jsonargparse.CLI(run)
