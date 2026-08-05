from __future__ import annotations

import jsonargparse

import open_economy_growth_kernel
from open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "open_economy_growth_kernel"


def run(output_dir: str | None = None, use_float64: bool = True):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = (
        {}
        if use_float64
        else dict(
            residual_atol=2e-4,
            ridge=1e-7,
            ridge_init=1e-5,
            gtol=1e-5,
            linear_solver="qr",
        )
    )
    tol = (
        {}
        if use_float64
        else dict(train_residual=5e-3, validation_residual=2e-2, benchmark_match=1e-2)
    )
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            open_economy_growth_kernel.open_economy_growth_kernel,
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
    go("r_star_005", parameters=dict(r_star=0.05))
    go("initial_debt", a_0=-0.5 * output_ss)
    go("initial_assets", a_0=0.5 * output_ss)
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
