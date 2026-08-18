from __future__ import annotations

import jsonargparse

from models import neoclassical_growth_kernel
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "neoclassical_growth_kernel"

SOLVER = neoclassical_growth_kernel.neoclassical_growth_kernel


# Float32 runs pass looser solver thresholds and acceptance gates; the
# float64 run uses the solver defaults. Shared with the robustness and
# sensitivity drivers so every experiment solves under the same regime.
def solver_overrides(use_float64):
    lm = {} if use_float64 else dict(residual_atol=5e-5, ridge=1e-7, gtol=1e-5)
    tol = (
        {}
        if use_float64
        else dict(train_residual=5e-3, validation_residual=1e-2, mpk_bound=5e-3)
    )
    return lm, tol


def run_experiment(name, output_dir, use_float64, **experiment):
    lm, tol = solver_overrides(use_float64)
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
        dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )


def run(
    output_dir: str | None = None,
    use_float64: bool = True,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"

    # QE figure experiments. The sparse grid keeps the baseline's front-loaded
    # relative spacing over the full train_T; far_steady_state trains on the
    # first ~2 e-foldings of the saddle path only.
    run_experiment("baseline", output_dir, use_float64)
    run_experiment(
        "sparse",
        output_dir,
        use_float64,
        data_set=dict(
            train_points_list=[
                0.0,
                2.5,
                7.5,
                12.5,
                25.0,
                37.5,
                50.0,
                62.5,
                75.0,
                87.5,
                95.0,
                100.0,
            ]
        ),
    )
    run_experiment(
        "far_steady_state",
        output_dir,
        use_float64,
        data_set=dict(train_T=20.0, train_points=21, test_T=30.0),
    )

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
