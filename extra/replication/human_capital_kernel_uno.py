from __future__ import annotations

import jsonargparse

from models import human_capital_kernel_uno
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

SOLVER = human_capital_kernel_uno.human_capital_kernel_uno
PREFIX = "human_capital_kernel_uno"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    configure_process()
    print(f"baseline -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds=1,
        options={"verbose": verbose},
    )
    if not bool(results_df["solution_accepted"].iloc[0]):
        # Acceptance is wall-clock gated (filtersqp time_limit); a rejected
        # baseline invalidates the QE figure, so fail loudly here.
        raise RuntimeError(
            f"human capital baseline rejected: {results_df['rejection_reason'].iloc[0]}"
        )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        dict(n_seeds=1, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
