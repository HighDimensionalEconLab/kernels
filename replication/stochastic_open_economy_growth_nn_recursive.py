from __future__ import annotations

import jsonargparse

from models import stochastic_open_economy_growth_nn_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "stochastic_open_economy_growth_nn_recursive"

SOLVER = stochastic_open_economy_growth_nn_recursive.stochastic_open_economy_growth_nn_recursive

DIAGNOSTIC_COLUMNS = [
    "train_loss",
    "test_euler_mean_abs",
    "test_c_euler_max_abs",
    "test_q_euler_max_abs",
    "steps",
    "solve_time",
]


def run(
    output_dir: str = ".results",
    n_seeds: int = 5,
    data_seed: int = 0,
    verbose: bool = False,
):
    configure_process()

    def go(name, experiment_n_seeds, **experiment):
        print(f"{name}: {experiment_n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=experiment_n_seeds,
            options={
                "verbose": verbose,
                "ic_set": {"data_seed": data_seed, **experiment.pop("ic_set", {})},
                # Cache accepted policies so the kernel distillation driver can
                # load the best baseline one as its frozen reference.
                "policy_cache": output_dir,
                **experiment,
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            {
                "n_seeds": experiment_n_seeds,
                "residual_dtype": "float64",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )
        accepted = int(results_df["solution_accepted"].sum())
        print(f"  accepted {accepted}/{len(results_df)}")
        print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())

    # Single initial condition (the module default): trajectories differ only
    # by their shock realization.
    go("baseline", n_seeds)
    # Initial-condition dispersion is an experiment, never a default: wide
    # NFA variation relative to log k is the point of the model.
    go(
        "dispersed_ic",
        1,
        ic_set={"train_asset_variance": 0.25, "train_log_k_variance": 0.01},
    )
    go("correlated_shocks", 1, corr_zr=-0.4)
    write_summary(
        output_dir,
        PREFIX,
        experiments={"baseline", "dispersed_ic", "correlated_shocks"},
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
