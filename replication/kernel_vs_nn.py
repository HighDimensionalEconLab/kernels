from __future__ import annotations

import jsonargparse

import neoclassical_growth_concave_convex_kernel
import neoclassical_growth_kernel
import neoclassical_growth_nn
import neoclassical_growth_nn_integrated
from replication.utilities import (
    BASIN_K0_GRID,
    classify_path_basin,
    enable_compilation_cache,
    kinked_thresholds,
    save_experiment,
    sweep,
    write_summary,
)

enable_compilation_cache()

PREFIX = "kernel_vs_nn"

# Shared time grid [0, 40]: error vs number of training points. The recursive
# variants are excluded here (their training set is self-simulated collocation
# states, not this grid); their density robustness is the dt sweep.
POINT_COUNTS = [4, 6, 8, 11, 16, 21, 31, 41]

BUTTERFLY = dict(A=0.5, b_1=3.0, b_2=2.5)


def classify_paths(results_df, test_df, group_keys):
    # Ex-post finite-horizon basin classification per solve (kernel and
    # sequential-NN paths end mid-transition; see classify_path_basin).
    k_thr, k_ss_low, k_ss_high = kinked_thresholds(**BUTTERFLY)
    outcomes = []
    for _, row in results_df.iterrows():
        mask = None
        for key in group_keys:
            key_mask = test_df[key] == row[key]
            mask = key_mask if mask is None else (mask & key_mask)
        path = test_df.loc[mask].sort_values("t")["k"].to_numpy()
        if not row.get("solution_accepted", False) or len(path) < 2:
            outcomes.append(
                {"basin_correct": False, "false_positive": False, "basin_failed": True}
            )
            continue
        outcomes.append(
            classify_path_basin(path, float(row["k_0"]), k_thr, k_ss_low, k_ss_high)
        )
    for key in ("basin_correct", "false_positive", "basin_failed"):
        results_df[key] = [outcome[key] for outcome in outcomes]
    results_df["k_thr"] = k_thr
    results_df["k_ss_low"] = k_ss_low
    results_df["k_ss_high"] = k_ss_high
    return results_df


def run(
    output_dir: str = ".results",
    # Exploration default; scale up (training is well under a second per
    # seed) once the experiment set is frozen for the final replication run.
    n_seeds: int = 10,
    verbose: bool = False,
):
    basin_grid = [float(k_0) for k_0 in BASIN_K0_GRID]

    def go(name, results_df, frames, hyperparameters, seeds=n_seeds):
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            dict(n_seeds=seeds),
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    # --- accuracy vs number of training points -------------------------------
    print(f"kernel_points_sweep -> {output_dir}/")
    results_df, _, hyperparameters = sweep(
        neoclassical_growth_kernel.neoclassical_growth_kernel,
        "data_set.train_points",
        POINT_COUNTS,
        options={"verbose": verbose},
    )
    go("kernel_points_sweep", results_df, {}, hyperparameters, seeds=1)

    print(f"nn_direct_points_sweep: {n_seeds} seeds -> {output_dir}/")
    results_df, _, hyperparameters = sweep(
        neoclassical_growth_nn.neoclassical_growth_nn,
        "data_set.train_points",
        POINT_COUNTS,
        n_seeds=n_seeds,
        options={"verbose": verbose},
    )
    go("nn_direct_points_sweep", results_df, {}, hyperparameters)

    print(f"nn_integrated_mlp_points_sweep: {n_seeds} seeds -> {output_dir}/")
    results_df, _, hyperparameters = sweep(
        neoclassical_growth_nn_integrated.neoclassical_growth_nn_integrated,
        "data_set.train_points",
        POINT_COUNTS,
        n_seeds=n_seeds,
        options={"verbose": verbose},
    )
    go("nn_integrated_mlp_points_sweep", results_df, {}, hyperparameters)

    print(f"nn_integrated_rff_points_sweep: {n_seeds} seeds -> {output_dir}/")
    results_df, _, hyperparameters = sweep(
        neoclassical_growth_nn_integrated.neoclassical_growth_nn_integrated,
        "data_set.train_points",
        POINT_COUNTS,
        n_seeds=n_seeds,
        options={
            "verbose": verbose,
            "nn_set": dict(
                approximation="rff",
                width=1024,
                rff_kernel="matern12",
                rff_radius=10.0,
                rff_sigma=1.0,
            ),
        },
    )
    go("nn_integrated_rff_points_sweep", results_df, {}, hyperparameters)

    # --- basin of attraction over the shared k_0 grid (secondary, path-space) --
    print(f"basin_kernel -> {output_dir}/")
    results_df, frames, hyperparameters = sweep(
        neoclassical_growth_concave_convex_kernel.neoclassical_growth_concave_convex_kernel,
        "k_0",
        basin_grid,
        options={"verbose": verbose},
        # Same rescue ladder as the threshold sweep (isolated fail-fast ipopt
        # failures solve cleanly under filtersqp).
        retries=(
            dict(
                solver_set=dict(preset="filtersqp", max_iterations=50, time_limit=2.0)
            ),
        ),
    )
    results_df = classify_paths(results_df, frames["test_df"], ["k_0"])
    go("basin_kernel", results_df, frames, hyperparameters, seeds=1)

    print(f"basin_nn_direct: {n_seeds} seeds x {len(basin_grid)} ICs")
    results_df, frames, hyperparameters = sweep(
        neoclassical_growth_nn.neoclassical_growth_nn,
        "k_0",
        basin_grid,
        n_seeds=n_seeds,
        # Fail fast per (k_0, seed): near the threshold slow solves are
        # rejections, not budget sinks. Single attempt everywhere so
        # frac_accepted measures first-try robustness.
        options={
            "verbose": verbose,
            **BUTTERFLY,
            "stop_set": dict(max_steps=500),
        },
    )
    results_df = classify_paths(results_df, frames["test_df"], ["k_0", "seed"])
    go("basin_nn_direct", results_df, {"test_df": frames["test_df"]}, hyperparameters)

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
