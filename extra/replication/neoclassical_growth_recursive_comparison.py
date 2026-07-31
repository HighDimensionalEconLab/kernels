from __future__ import annotations

import jax
import jsonargparse
import pandas as pd

import neoclassical_growth_kernel_recursive
import neoclassical_growth_nn_recursive
from replication.utilities import (
    DETERMINISTIC_TIMING_REPETITIONS,
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_recursive_comparison"


def apply_common_validation(results, train_euler_tolerance, test_error_tolerance):
    results = results.copy()
    collocation_path_ok = (
        results["collocation_path_ok"].astype(bool)
        if "collocation_path_ok" in results
        else True
    )
    benchmark_quality_ok = (
        results["test_mean_abs_k_rel_error"] <= test_error_tolerance
    ) & (results["test_mean_abs_c_rel_error"] <= test_error_tolerance)
    common_accepted = (
        results["solver_converged"].astype(bool)
        & collocation_path_ok
        & (results["max_train_euler"] <= train_euler_tolerance)
        & results["test_path_ok"].astype(bool)
        & results["transversality_ok"].astype(bool)
        & benchmark_quality_ok
    )
    results["native_solution_accepted"] = results["solution_accepted"]
    results["benchmark_quality_ok"] = benchmark_quality_ok
    results["common_validation_accepted"] = common_accepted
    # The comparison's solution_accepted has one identical meaning for both
    # methods. Native solver-specific acceptance remains in the adjacent field.
    results["solution_accepted"] = common_accepted
    return results


def method_summary(method, results):
    accepted = results[results["common_validation_accepted"]]
    sample = accepted if not accepted.empty else results
    return {
        "method": method,
        "solution_accepted": not accepted.empty,
        "num_runs": len(results),
        "frac_accepted": float(results["common_validation_accepted"].mean()),
        "optimizer_solve_time": float(sample["solve_time"].median()),
        "total_fit_time": float(sample["total_fit_time"].median()),
        "test_pooled_median_abs_k_rel_error": float(
            sample["test_pooled_median_abs_k_rel_error"].median()
        ),
        "test_pooled_median_abs_c_rel_error": float(
            sample["test_pooled_median_abs_c_rel_error"].median()
        ),
        "test_median_trajectory_mean_abs_k_rel_error": float(
            sample["test_median_trajectory_mean_abs_k_rel_error"].median()
        ),
        "test_median_trajectory_mean_abs_c_rel_error": float(
            sample["test_median_trajectory_mean_abs_c_rel_error"].median()
        ),
        "test_euler_mean_abs": float(sample["test_euler_mean_abs"].median()),
    }


def wide_comparison(method_rows, nn_first_run_fit_time, kernel_first_run_fit_time):
    nn_row, kernel_row = method_rows
    row = {
        "solution_accepted": bool(
            nn_row["solution_accepted"] and kernel_row["solution_accepted"]
        ),
        "nn_warmup_runs": 1,
        "kernel_timing_repetitions": DETERMINISTIC_TIMING_REPETITIONS,
        "nn_first_run_total_fit_time": nn_first_run_fit_time,
        "kernel_first_run_total_fit_time": kernel_first_run_fit_time,
    }
    for prefix, summary in (("nn", nn_row), ("kernel", kernel_row)):
        for key, value in summary.items():
            if key not in {"method", "solution_accepted"}:
                row[f"{prefix}_{key}"] = value
        row[f"{prefix}_solution_accepted"] = int(summary["solution_accepted"])

    nn_time = nn_row["total_fit_time"]
    row["kernel_to_nn_fit_time_ratio"] = kernel_row["total_fit_time"] / nn_time
    for variable in ("k", "c"):
        metric = f"test_median_trajectory_mean_abs_{variable}_rel_error"
        nn_error = nn_row[metric]
        row[f"kernel_to_nn_{variable}_error_ratio"] = (
            kernel_row[metric] / nn_error if nn_error > 0.0 else float("inf")
        )
    return pd.DataFrame([row])


def run(
    output_dir: str = ".results_extra",
    n_nn_seeds: int = 10,
    use_float64: bool = True,
    g: float = 0.02,
    z_0: float | list[float] = 1.0,
    k_0: float | list[float] = 1.0,
    num_test_trajectories: int = 50,
    test_log_variance: float = 0.0025,
    train_T: float = 100.0,
    train_points: int = 65,
    train_tvc_T: float = 200.0,
    tvc_steps: int = 256,
    test_T: float = 40.0,
    dt: float = 1.0,
    data_seed: int = 0,
    train_euler_tolerance: float = 2e-5,
    test_error_tolerance: float = 5e-3,
    verbose: bool = False,
):
    if n_nn_seeds < 1:
        raise ValueError("n_nn_seeds must be positive")
    if train_points < 2:
        raise ValueError("train_points must be at least two")
    data_set = {
        "num_test_trajectories": num_test_trajectories,
        "train_T": train_T,
        "train_points": train_points,
        "train_tvc_T": train_tvc_T,
        "tvc_steps": tvc_steps,
        "test_T": test_T,
        "dt": dt,
        "benchmark_T": max(200.0, train_T, test_T),
        "benchmark_points": 801,
    }
    ic_set = {
        "data_seed": data_seed,
        "test_log_variance": test_log_variance,
    }
    common = {
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "use_float64": use_float64,
        "data_set": data_set,
        "ic_set": ic_set,
        "verbose": verbose,
    }
    # ``train_points`` means the same count on both sides: the NN's rollout
    # knots and the kernel's fixed centers land on the same dates.
    meta = {
        "n_nn_seeds": n_nn_seeds,
        "data_seed": data_seed,
        "residual_dtype": "float64" if use_float64 else "float32",
        "train_euler_tolerance": train_euler_tolerance,
        "test_error_tolerance": test_error_tolerance,
    }

    # Compile and populate benchmark caches once, then time every retained NN
    # seed warm. This keeps n_nn_seeds=1 comparable to the kernel's warm median.
    nn_first_run_results, _, _ = ensemble(
        neoclassical_growth_nn_recursive.neoclassical_growth_nn_recursive,
        1,
        options=common,
    )
    nn_results, nn_frames, nn_hyperparameters = ensemble(
        neoclassical_growth_nn_recursive.neoclassical_growth_nn_recursive,
        n_nn_seeds,
        options=common,
    )
    nn_results = apply_common_validation(
        nn_results, train_euler_tolerance, test_error_tolerance
    )
    save_experiment(
        "nn",
        nn_results,
        nn_frames,
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters=nn_hyperparameters,
    )

    kernel_results, kernel_frames, kernel_hyperparameters = ensemble(
        neoclassical_growth_kernel_recursive.neoclassical_growth_kernel_recursive,
        1,
        options=common,
    )
    kernel_results = apply_common_validation(
        kernel_results, train_euler_tolerance, test_error_tolerance
    )
    save_experiment(
        "kernel",
        kernel_results,
        kernel_frames,
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters=kernel_hyperparameters,
    )

    method_rows = [
        method_summary("recursive_nn", nn_results),
        method_summary("fixed_recursive_kernel", kernel_results),
    ]
    method_df = pd.DataFrame(method_rows)
    kernel_first_run_fit_time = float(
        kernel_frames["timing_df"]["total_fit_time"].iloc[0]
    )
    comparison = wide_comparison(
        method_rows,
        float(nn_first_run_results["total_fit_time"].iloc[0]),
        kernel_first_run_fit_time,
    )
    save_experiment(
        "comparison",
        comparison,
        {"method_df": method_df},
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters={
            "data_set": data_set,
            "ic_set": ic_set,
            "parameters": {"g": g},
            "initial_conditions": {"z_0": z_0, "k_0": k_0},
            "common_validation": {
                "train_euler_tolerance": train_euler_tolerance,
                "test_error_tolerance": test_error_tolerance,
                "requires_solver_converged": True,
                "requires_positive_finite_test_path": True,
                "requires_transversality": True,
            },
        },
    )
    write_summary(output_dir, PREFIX, experiments={"nn", "kernel", "comparison"})
    print(method_df.to_string(index=False))


if __name__ == "__main__":
    jsonargparse.CLI(run)
