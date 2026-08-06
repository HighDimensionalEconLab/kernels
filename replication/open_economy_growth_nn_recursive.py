from __future__ import annotations

import jsonargparse

from models import open_economy_growth_nn_recursive
from models.open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "open_economy_growth_nn_recursive"
SOLVER = open_economy_growth_nn_recursive.open_economy_growth_nn_recursive


def run(
    output_dir: str | None = None,
    n_seeds: int = 5,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    # The float32 solve resolves a 1e-12 loss and needs it to pass the
    # independent path-residual and benchmark gates (1e-10 stops
    # prematurely); the benchmark gate doubles.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-12)
    tol = {} if use_float64 else dict(benchmark_match=2e-3)

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
    write_summary(
        output_dir,
        PREFIX,
        experiments={"baseline", "initial_debt", "initial_assets"},
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
