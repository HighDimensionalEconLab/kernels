from __future__ import annotations

import jax
import jsonargparse
import pandas as pd

import neoclassical_growth_nn_recursive
from extra.replication.neoclassical_growth_recursive_comparison import (
    apply_common_validation,
)
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_nn_recursive_tsit5_comparison"
SOLVER = neoclassical_growth_nn_recursive.neoclassical_growth_nn_recursive


def retained_tolerance_screen():
    """Machine-readable record of the bounded T=100 mesh calibration."""
    rows = []
    for rtol, states, reason in (
        (1e-6, 26, "too_few_states"),
        (3e-7, 31, "too_few_states"),
        (1e-7, 37, "too_few_states"),
        (3e-8, 45, "too_few_states"),
        (1e-8, 54, "below_point_target"),
        (3e-9, 67, "selected"),
        (1e-9, 81, "exhausts_default_capacity"),
        (3e-10, 101, "exceeds_default_capacity"),
    ):
        rows.append(
            {
                "adaptive_rtol": rtol,
                "adaptive_atol": rtol * 1e-2,
                "adaptive_dt_0": 1.0,
                "screen_attempt_budget": 128,
                "accepted_states": states,
                "train_T": 100.0,
                "collocation_path_ok": True,
                "selected": reason == "selected",
                "selection_reason": reason,
                "full_fit_timing_seconds": None,
                "timing_status": (
                    "reported_in_method_results"
                    if reason == "selected"
                    else "mesh_screen_only_not_timed"
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(method, results, first_run_total_fit_time):
    accepted = results.loc[results["common_validation_accepted"]]
    sample = accepted if not accepted.empty else results
    return {
        "method": method,
        "solution_accepted": not accepted.empty,
        "num_runs": len(results),
        "frac_accepted": float(results["common_validation_accepted"].mean()),
        "first_run_total_fit_time": first_run_total_fit_time,
        "warm_solve_time": float(sample["solve_time"].median()),
        "warm_total_fit_time": float(sample["total_fit_time"].median()),
        "num_collocation_states": float(sample["num_collocation_states"].median()),
        "collocation_states_per_trajectory": float(
            sample["collocation_states_per_trajectory_median"].median()
        ),
        "max_train_euler": float(sample["max_train_euler"].median()),
        "test_median_abs_k_rel_error": float(
            sample["test_median_abs_k_rel_error"].median()
        ),
        "test_median_abs_c_rel_error": float(
            sample["test_median_abs_c_rel_error"].median()
        ),
        "test_euler_mean_abs": float(sample["test_euler_mean_abs"].median()),
    }


def run(
    output_dir: str = ".results_extra",
    n_seeds: int = 5,
    use_float64: bool = True,
    g: float = 0.02,
    z_0: float | list[float] = 1.0,
    k_0: float | list[float] = 1.0,
    num_test_trajectories: int = 50,
    test_log_variance: float = 0.0025,
    train_T: float = 100.0,
    train_points: int = 65,
    train_steps: int = 128,
    train_tvc_T: float = 200.0,
    tvc_steps: int = 256,
    test_T: float = 40.0,
    dt: float = 1.0,
    data_seed: int = 0,
    train_euler_tolerance: float = 2e-5,
    test_error_tolerance: float = 5e-3,
    verbose: bool = False,
):
    """Compare fixed exact-knot and adaptive accepted-step collocation."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if train_points < 2:
        raise ValueError("train_points must be at least two")
    common = {
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "use_float64": use_float64,
        "ic_set": {
            "data_seed": data_seed,
            "test_log_variance": test_log_variance,
        },
        "verbose": verbose,
    }
    common_data = {
        "num_test_trajectories": num_test_trajectories,
        "train_T": train_T,
        "train_points": train_points,
        "train_steps": train_steps,
        "train_tvc_T": train_tvc_T,
        "tvc_steps": tvc_steps,
        "test_T": test_T,
        "dt": dt,
        "benchmark_T": max(200.0, train_T, test_T),
        "benchmark_points": 801,
    }
    meta = {
        "n_seeds": n_seeds,
        "data_seed": data_seed,
        "residual_dtype": "float64" if use_float64 else "float32",
        "train_euler_tolerance": train_euler_tolerance,
        "test_error_tolerance": test_error_tolerance,
        "evaluation_ode_solver": "rk4",
    }

    summaries = []
    for method, ode_solver in (
        ("fixed_rk4_exact_knots", "rk4"),
        ("tsit5_accepted_steps", "tsit5"),
    ):
        options = {
            **common,
            "data_set": {**common_data, "ode_solver": ode_solver},
        }
        first_results, _, _ = ensemble(SOLVER, 1, options=options)
        results, frames, hyperparameters = ensemble(SOLVER, n_seeds, options=options)
        results = apply_common_validation(
            results, train_euler_tolerance, test_error_tolerance
        )
        first_run_total_fit_time = float(first_results["total_fit_time"].iloc[0])
        results["first_run_total_fit_time"] = first_run_total_fit_time
        save_experiment(
            method,
            results,
            frames,
            output_dir,
            meta,
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )
        summaries.append(summarize_method(method, results, first_run_total_fit_time))

    comparison = pd.DataFrame(summaries)
    fixed = comparison.iloc[0]
    adaptive = comparison.iloc[1]
    comparison["adaptive_to_fixed_warm_fit_time_ratio"] = (
        adaptive["warm_total_fit_time"] / fixed["warm_total_fit_time"]
    )
    comparison["adaptive_to_fixed_collocation_point_ratio"] = (
        adaptive["collocation_states_per_trajectory"]
        / fixed["collocation_states_per_trajectory"]
    )
    save_experiment(
        "comparison",
        comparison,
        {"tolerance_screen_df": retained_tolerance_screen()},
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters={
            "data_set": common_data,
            "ic_set": common["ic_set"],
            "parameters": {"g": g},
            "initial_conditions": {"z_0": z_0, "k_0": k_0},
            "common_validation": {
                "train_euler_tolerance": train_euler_tolerance,
                "test_error_tolerance": test_error_tolerance,
                "requires_solver_converged": True,
                "requires_collocation_path": True,
                "requires_positive_finite_test_path": True,
                "requires_transversality": True,
            },
        },
    )
    write_summary(
        output_dir,
        PREFIX,
        experiments={"fixed_rk4_exact_knots", "tsit5_accepted_steps", "comparison"},
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    jsonargparse.CLI(run)
