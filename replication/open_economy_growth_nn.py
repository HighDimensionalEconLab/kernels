from __future__ import annotations

import jsonargparse

from models import open_economy_growth_nn
from models.open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "open_economy_growth_nn"
SOLVER = open_economy_growth_nn.open_economy_growth_nn


def run(
    output_dir: str | None = None,
    n_seeds: int = 7,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-8)
    tol = (
        {}
        if use_float64
        else dict(train_residual=5e-3, validation_residual=2e-2, benchmark_match=1e-2)
    )

    def go(name, experiment_n_seeds, **experiment):
        print(f"{name}: {experiment_n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=experiment_n_seeds,
            options={
                "use_float64": use_float64,
                "verbose": verbose,
                "stop_set": stop,
                "tolerance_set": tol,
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
                "residual_dtype": "float64" if use_float64 else "float32",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("baseline", n_seeds)
    go("initial_debt", 1, a_0=-0.5 * output_ss)
    go("initial_assets", 1, a_0=0.5 * output_ss)

    # Explicit set: the prefix is a strict prefix of
    # open_economy_growth_nn_recursive, whose JSONs share the glob.
    write_summary(
        output_dir,
        PREFIX,
        experiments={"baseline", "initial_debt", "initial_assets"},
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
