from __future__ import annotations

import jsonargparse

import multicountry_growth_kernel_nlls
from multicountry_growth import MulticountryParameters, multicountry_output_scale
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

enable_compilation_cache()

PREFIX = "multicountry_growth_kernel"
SOLVER = multicountry_growth_kernel_nlls.multicountry_growth_kernel_nlls


def run(output_dir: str = ".results"):
    parameters = MulticountryParameters()
    output_scale = float(multicountry_output_scale(parameters))
    meta = dict(n_seeds=1)

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options=experiment,
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

    go("baseline")
    go("initial_debt", a_0=(-0.5 * output_scale,))
    go("initial_assets", a_0=(0.5 * output_scale,))
    go("asymmetric_capital", k_0=(1.0, 2.0))
    go(
        "heterogeneous_countries",
        parameters=dict(
            populations=(1.0, 2.0),
            productivities=(1.0, 1.1),
        ),
    )
    for horizon, points in ((40.0, 41), (80.0, 61), (120.0, 81)):
        go(
            f"horizon_{round(horizon)}",
            a_0=(-0.5 * output_scale,),
            data_set=dict(
                train_T=horizon,
                train_points=points,
                sampling="uniform",
                test_T=min(50.0, horizon),
                validation_points=201,
                benchmark_T=max(160.0, horizon + 40.0),
                benchmark_points=181,
            ),
        )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
