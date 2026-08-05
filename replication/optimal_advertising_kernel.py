from __future__ import annotations

import jsonargparse

import optimal_advertising_kernel
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "optimal_advertising_kernel"


def run(
    output_dir: str | None = None,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(residual_atol=3e-5, ridge=1e-7, gtol=1e-5)
    tol = {} if use_float64 else dict(train_residual=1e-3)
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        optimal_advertising_kernel.optimal_advertising_kernel,
        n_seeds=1,
        options={
            "use_float64": use_float64,
            "verbose": verbose,
            "lm_set": lm,
            "tolerance_set": tol,
        },
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
