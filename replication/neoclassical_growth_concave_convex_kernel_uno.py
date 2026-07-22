from __future__ import annotations

import jax
import jsonargparse
import numpy as np

import neoclassical_growth_concave_convex_kernel_uno
from replication.utilities import (
    classify_path_basin,
    enable_compilation_cache,
    kinked_thresholds,
    save_experiment,
    sweep,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_concave_convex_kernel_uno"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    n_points = 70
    grid = np.linspace(0.5, 4.0, n_points)
    print(f"threshold sweep: {n_points} initial conditions -> {output_dir}/")
    results_df, frames, hyperparameters = sweep(
        neoclassical_growth_concave_convex_kernel_uno.neoclassical_growth_concave_convex_kernel_uno,
        "k_0",
        [float(k_0) for k_0 in grid],
        options={"verbose": verbose},
    )

    # Ex-post basin classification of each finite-horizon path (side of the
    # kink the trajectory ends on + direction of motion).
    parameters = hyperparameters["parameters"]
    k_thr, k_ss_low, k_ss_high = kinked_thresholds(
        a=parameters["a"],
        delta=parameters["delta"],
        rho_hat=parameters["rho_hat"],
        A=parameters["A"],
        b_1=parameters["b_1"],
        b_2=parameters["b_2"],
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
        dict(n_seeds=1, n_points=n_points),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
