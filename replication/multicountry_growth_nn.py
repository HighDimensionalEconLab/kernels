from __future__ import annotations

import jsonargparse

from models import multicountry_growth_nn
from replication.multicountry_growth_kernel import LARGE_COUNTRY_PARAMETERS
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn


# Float32 runs pass looser solver thresholds and acceptance gates; the
# float64 run uses the solver defaults. Shared with the sensitivity driver.
def solver_overrides(use_float64):
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
    return stop, tol


def run(
    output_dir: str | None = None,
    n_seeds: int = 7,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    stop, tol = solver_overrides(use_float64)

    def go(name, scenario, **options):
        print(f"{name}: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=n_seeds,
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
                "n_seeds": n_seeds,
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
    write_summary(output_dir, PREFIX, experiments={"baseline", "large_countries"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
