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


def run_experiment(name, output_dir, **experiment):
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
        dict(n_seeds=1, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )


def run(
    output_dir: str = ".results",
):
    configure_process()

    # QE figure experiments. The sparse grid keeps the baseline's front-loaded
    # relative spacing over the full train_T; far_steady_state trains on the
    # first ~2 e-foldings of the saddle path only.
    run_experiment("baseline", output_dir)
    run_experiment(
        "sparse",
        output_dir,
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
        data_set=dict(train_T=20.0, train_points=21, test_T=30.0),
    )

    run_experiment("baseline_g02", output_dir, g=0.02)
    run_experiment("baseline_g02_learned", output_dir, g=0.02, learn_g_hat=True)

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
