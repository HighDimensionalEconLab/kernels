from __future__ import annotations

import jax
import jsonargparse

import multicountry_growth_kernel
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

PREFIX = "multicountry_growth_kernel"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel


def run(output_dir: str = ".results", use_float64: bool = True):
    parameters = MulticountryParameters()
    output_scale = float(multicountry_output_scale(parameters))
    three_country_parameters = MulticountryParameters(
        populations=(1.0, 1.0, 1.0),
        productivities=(1.0, 1.0, 1.0),
    )
    three_country_output_scale = float(
        multicountry_output_scale(three_country_parameters)
    )
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options={"use_float64": use_float64, **experiment},
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
