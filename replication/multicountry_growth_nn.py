from __future__ import annotations

import jsonargparse

from models import multicountry_growth_nn
from replication.multicountry_growth_kernel import (
    LARGE_COUNTRY_COUNT,
    SENSITIVITY_TIMES,
)
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn
LARGE_COUNTRY_PARAMETERS = dict(
    gammas=(2.0,) * LARGE_COUNTRY_COUNT,
    populations=(1.0,) * LARGE_COUNTRY_COUNT,
    productivities=(1.0,) * LARGE_COUNTRY_COUNT,
)


def run(
    output_dir: str | None = None,
    n_seeds: int = 7,
    sensitivity_seeds: int = 1,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-8)
    tol = (
        {}
        if use_float64
        else dict(
            train_residual=5e-3,
            validation_residual=1e-2,
            benchmark_match=1e-2,
        )
    )

    def go(name, scenario, seeds=None, **options):
        seeds = n_seeds if seeds is None else seeds
        print(f"{name}: {seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=seeds,
            options={
                "use_float64": use_float64,
                "verbose": verbose,
                "stop_set": stop,
                "tolerance_set": tol,
                **options,
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            {
                "n_seeds": seeds,
                "scenario": scenario,
                "residual_dtype": "float64" if use_float64 else "float32",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("baseline", "two_country_drawn_initial_conditions")
    go(
        "large_countries",
        "ten_country_drawn_initial_conditions",
        parameters=LARGE_COUNTRY_PARAMETERS,
    )
    experiments = {"baseline", "large_countries"}
    # The implicit-AD experiments are separate runs on their own seed budget:
    # the tables report one solve, so training the full ensemble through the
    # reverse pass buys nothing. Float64 only -- the float32 noise floor
    # swamps the derivatives. The dates are the kernel driver's, since the
    # two solvers' tables are compared cell by cell.
    if use_float64:
        go(
            "sensitivity",
            "two_country_drawn_initial_conditions",
            seeds=sensitivity_seeds,
            sensitivity_times=SENSITIVITY_TIMES,
        )
        go(
            "sensitivity_ten_countries",
            "ten_country_drawn_initial_conditions",
            seeds=sensitivity_seeds,
            parameters=LARGE_COUNTRY_PARAMETERS,
            sensitivity_times=SENSITIVITY_TIMES,
        )
        experiments |= {"sensitivity", "sensitivity_ten_countries"}
    write_summary(output_dir, PREFIX, experiments=experiments)


if __name__ == "__main__":
    jsonargparse.CLI(run)
