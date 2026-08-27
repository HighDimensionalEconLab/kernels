from __future__ import annotations

import jax.numpy as jnp
import jsonargparse
import numpy as np

from models import neoclassical_growth_nn
from models.neoclassical_growth_shared import (
    KINKED_CALIBRATION,
    StructuralParameters,
    kinked_thresholds,
)
from replication.utilities import (
    classify_path_basin,
    configure_process,
    save_experiment,
    sweep,
    write_summary,
)

SOLVER = neoclassical_growth_nn.neoclassical_growth_nn
PREFIX = "neoclassical_growth_concave_convex_nn"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    configure_process()
    n_points = 70
    grid = np.linspace(0.5, 4.0, n_points)
    print(f"threshold sweep: {n_points} initial conditions -> {output_dir}/")
    results_df, frames, hyperparameters = sweep(
        SOLVER,
        "k_0",
        [float(k_0) for k_0 in grid],
        options={
            **KINKED_CALIBRATION,
            "verbose": verbose,
            # The kinked basin experiments keep their own calibrated grid:
            # paths settle by T = 40, and the file defaults carry the smooth
            # model's longer horizon.
            "data_set": dict(
                train_T=40.0, train_points=41, test_T=50.0, test_points=41
            ),
        },
    )

    # Ex-post basin classification of each finite-horizon path, identical to the
    # kernel sweep: a sequence-space solution may still be in motion at test_T,
    # so the settle-ball test the recursive policy uses cannot apply.
    k_thr, k_ss_low, k_ss_high = kinked_thresholds(
        StructuralParameters.create(dtype=jnp.float64, **hyperparameters["parameters"])
    )
    test_df = frames["test_df"]
    outcomes = []
    for _, row in results_df.iterrows():
        k_0 = row["k_0"]
        path = test_df.loc[test_df["k_0"] == k_0].sort_values("t")["k"].to_numpy()
        if not bool(row["solution_accepted"]) or len(path) < 2:
            outcomes.append(
                {"basin_correct": False, "false_positive": False, "basin_failed": True}
            )
            continue
        outcomes.append(
            classify_path_basin(path, float(k_0), k_thr, k_ss_low, k_ss_high)
        )
    for key in ("basin_correct", "false_positive", "basin_failed"):
        results_df[key] = [outcome[key] for outcome in outcomes]
    results_df["k_thr"] = k_thr
    results_df["k_ss_low"] = k_ss_low
    results_df["k_ss_high"] = k_ss_high

    n_accepted = int(results_df["solution_accepted"].sum())
    print(
        f"threshold: {n_accepted}/{n_points} accepted, "
        f"{int(results_df['basin_correct'].sum())}/{n_points} basin-correct"
    )
    save_experiment(
        "threshold",
        results_df,
        frames,
        output_dir,
        dict(
            n_seeds=1,
            n_points=n_points,
            residual_dtype="float64",
        ),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
