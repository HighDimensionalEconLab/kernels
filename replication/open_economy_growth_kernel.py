from __future__ import annotations

import jax
import jsonargparse

import open_economy_growth_kernel_nlls
from open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "open_economy_growth_kernel"


def run(output_dir: str = ".results", use_float64: bool = True):
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            open_economy_growth_kernel_nlls.open_economy_growth_kernel_nlls,
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
    go("initial_debt", a_0=-0.5 * output_ss)
    go("initial_assets", a_0=0.5 * output_ss)
    go(
        "horizon_200",
        data_set=dict(
            train_T=200.0,
            train_points=101,
            dense_T=40.0,
            tail_points=20,
            benchmark_T=240.0,
        ),
    )
    go(
        "uniform_321",
        data_set=dict(
            train_points=321,
            sampling="uniform",
        ),
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
