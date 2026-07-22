from __future__ import annotations

import jax
import jsonargparse

import multicountry_growth_nn
from multicountry_growth import MulticountryParameters, multicountry_output_scale
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn


def run(
    output_dir: str = ".results",
    n_seeds: int = 100,
    use_float64: bool = True,
    verbose: bool = False,
):
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

    def go(name, scenario, **options):
        print(f"{name}: {n_seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=n_seeds,
            options={
                "use_float64": use_float64,
                "verbose": verbose,
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
