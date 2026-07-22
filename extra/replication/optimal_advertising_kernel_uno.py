from __future__ import annotations

import jax
import jsonargparse

import extra.optimal_advertising_kernel_uno as optimal_advertising_kernel_uno
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "optimal_advertising_kernel_uno"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        optimal_advertising_kernel_uno.optimal_advertising_kernel_uno,
        n_seeds=1,
        options={"verbose": verbose},
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=1),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
