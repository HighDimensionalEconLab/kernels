from __future__ import annotations

import jsonargparse

import multicountry_growth_nn
from multicountry_growth import MulticountryParameters, multicountry_output_scale
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn


def run(
    output_dir: str | None = None,
    n_seeds: int = 5,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = MulticountryParameters()
    output_scale = float(multicountry_output_scale(parameters))
    ten_country_parameters = MulticountryParameters(
        populations=(1.0,) * 10,
        productivities=(1.0,) * 10,
    )
    ten_country_output_scale = float(multicountry_output_scale(ten_country_parameters))
    ten_country_a_0 = (-0.5 * ten_country_output_scale,) + (
        0.5 * ten_country_output_scale / 9,
    ) * 9

    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-8)
    tol = (
        {}
        if use_float64
        else dict(
            train_residual=5e-3,
            validation_residual=1e-2,
            accounting=1e-3,
            benchmark_match=1e-2,
        )
    )

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

    go(
        "baseline",
        "two_country_initial_nfa",
        a_0=(-0.5 * output_scale, 0.5 * output_scale),
    )
    go(
        "large_countries",
        "equal_countries_balanced_initial_nfa",
        a_0=ten_country_a_0,
        parameters={
            "populations": ten_country_parameters.populations,
            "productivities": ten_country_parameters.productivities,
        },
        nn_set={"width": 32, "depth": 4},
    )
    write_summary(
        output_dir,
        PREFIX,
        experiments={"baseline", "large_countries"},
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
