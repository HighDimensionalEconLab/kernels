from __future__ import annotations

import jsonargparse

import multicountry_growth_kernel
from multicountry_growth import MulticountryParameters, multicountry_output_scale
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_kernel"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel


def run(output_dir: str | None = None, use_float64: bool = True):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = MulticountryParameters()
    output_scale = float(multicountry_output_scale(parameters))
    three_country_parameters = MulticountryParameters(
        populations=(1.0, 1.0, 1.0),
        productivities=(1.0, 1.0, 1.0),
    )
    three_country_output_scale = float(
        multicountry_output_scale(three_country_parameters)
    )
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(ridge=1e-7, ridge_init=1e-5, gtol=1e-6)
    tol = (
        {}
        if use_float64
        else dict(
            train_residual=1e-3,
            validation_residual=1e-2,
            accounting=1e-3,
            benchmark_match=1e-2,
        )
    )
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options={
                "use_float64": use_float64,
                "lm_set": lm,
                "tolerance_set": tol,
                **experiment,
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

    go("baseline")
    go("initial_debt", a_0=(-0.5 * output_scale, 0.5 * output_scale))
    go("initial_assets", a_0=(0.5 * output_scale, -0.5 * output_scale))
    go("asymmetric_capital", k_0=(1.0, 2.0))
    go(
        "three_countries",
        a_0=(
            -0.5 * three_country_output_scale,
            0.25 * three_country_output_scale,
            0.25 * three_country_output_scale,
        ),
        parameters=dict(
            populations=(1.0, 1.0, 1.0),
            productivities=(1.0, 1.0, 1.0),
        ),
    )
    go(
        "heterogeneous_countries",
        parameters=dict(
            populations=(1.0, 2.0),
            productivities=(1.0, 1.1),
        ),
    )
    write_summary(
        output_dir,
        PREFIX,
        experiments={
            "baseline",
            "initial_debt",
            "initial_assets",
            "asymmetric_capital",
            "three_countries",
            "heterogeneous_countries",
        },
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
