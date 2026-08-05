from __future__ import annotations

import jsonargparse
import numpy as np

import neoclassical_growth_concave_convex_kernel
from replication.utilities import (
    classify_path_basin,
    configure_process,
    kinked_thresholds,
    save_experiment,
    sweep,
    write_summary,
)

PREFIX = "neoclassical_growth_concave_convex_kernel"


def run(output_dir: str | None = None, use_float64: bool = True):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(residual_atol=1e-5, ridge=1e-7, gtol=1e-6)
    tol = (
        {}
        if use_float64
        else dict(train_residual=1e-3, validation_residual=1e-2, mpk_bound=5e-3)
    )
    n_points = 70
    grid = np.linspace(0.5, 4.0, n_points)
    print(f"threshold sweep: {n_points} initial conditions -> {output_dir}/")
    results_df, frames, hyperparameters = sweep(
        neoclassical_growth_concave_convex_kernel.neoclassical_growth_concave_convex_kernel,
        "k_0",
        [float(k_0) for k_0 in grid],
        options={"use_float64": use_float64, "lm_set": lm, "tolerance_set": tol},
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
        dict(
            n_seeds=1,
            n_points=n_points,
            residual_dtype="float64" if use_float64 else "float32",
        ),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
