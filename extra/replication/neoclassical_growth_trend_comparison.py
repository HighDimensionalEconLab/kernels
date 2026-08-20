from __future__ import annotations

import json
import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import pandas as pd

from models import (
    neoclassical_growth_kernel,
    neoclassical_growth_kernel_recursive,
    neoclassical_growth_nn,
    neoclassical_growth_nn_recursive,
)
from replication.utilities import configure_process, style

PREFIX = "neoclassical_growth_trend_comparison"


def run(
    output_dir: str = ".results_extra",
    g: float = 0.02,
    k_0: float = 1.0,
    z_0: float = 1.0,
    seed: int = 0,
    # Common held-out grid for all four solvers, inside every training range.
    test_T: float = 100.0,
    test_points: int = 101,
    verbose: bool = False,
):
    configure_process()
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Sequential solvers learn the trend rate g_hat; recursive solvers are
    # given g and carry the trend in the technology state.
    sequential_kernel_results, sequential_kernel_frames = (
        neoclassical_growth_kernel.neoclassical_growth_kernel(
            g=g,
            learn_g_hat=True,
            k_0=k_0,
            data_set=neoclassical_growth_kernel.DataSettings(
                test_T=test_T, test_points=test_points
            ),
        )
    )
    sequential_nn_results, sequential_nn_frames = (
        neoclassical_growth_nn.neoclassical_growth_nn(
            g=g,
            k_0=k_0,
            seed=seed,
            data_set=neoclassical_growth_nn.DataSettings(
                test_T=test_T, test_points=test_points
            ),
            nn_set=neoclassical_growth_nn.GrowthPolicySettings(learn_g_hat=True),
            tolerance_set=neoclassical_growth_nn.ToleranceSettings(
                train_loss_threshold=1e-14
            ),
            verbose=verbose,
        )
    )

    single_ic = dict(
        data_seed=0,
        num_train_trajectories=1,
        num_test_trajectories=1,
        train_log_z_variance=0.0,
        train_log_k_variance=0.0,
        test_log_z_variance=0.0,
        test_log_k_variance=0.0,
    )
    recursive_kernel_results, recursive_kernel_frames = (
        neoclassical_growth_kernel_recursive.neoclassical_growth_kernel_recursive(
            g=g,
            z_0=z_0,
            k_0=k_0,
            data_set=neoclassical_growth_kernel_recursive.DataSettings(
                test_T=test_T, test_points=test_points
            ),
            ic_set=neoclassical_growth_kernel_recursive.InitialConditionSettings(
                **single_ic
            ),
        )
    )
    recursive_nn_results, recursive_nn_frames = (
        neoclassical_growth_nn_recursive.neoclassical_growth_nn_recursive(
            g=g,
            z_0=z_0,
            k_0=k_0,
            seed=seed,
            data_set=neoclassical_growth_nn_recursive.DataSettings(
                test_T=test_T, test_points=test_points
            ),
            ic_set=neoclassical_growth_nn_recursive.InitialConditionSettings(
                **single_ic
            ),
            verbose=verbose,
        )
    )

    comparison = [
        {
            "method": "sequential_kernel_learned_g",
            "solution_accepted": bool(sequential_kernel_results["solution_accepted"]),
            "g_hat": sequential_kernel_results["g_hat"],
            "g_hat_error": sequential_kernel_results["g_hat_error"],
            "log_tvc_residual": sequential_kernel_results["log_tvc_residual"],
            "transversality_ok": bool(sequential_kernel_results["transversality_ok"]),
            "test_k_rel_error_mean": sequential_kernel_results["test_k_rel_error_mean"],
            "test_c_rel_error_mean": sequential_kernel_results["test_c_rel_error_mean"],
            "solve_time": sequential_kernel_results["solve_time"],
        },
        {
            "method": "sequential_nn_learned_g",
            "solution_accepted": bool(sequential_nn_results["solution_accepted"]),
            "g_hat": sequential_nn_results["g_hat"],
            "g_hat_error": sequential_nn_results["g_hat_error"],
            "log_tvc_residual": sequential_nn_results["log_tvc_residual"],
            "transversality_ok": bool(sequential_nn_results["transversality_ok"]),
            "test_k_rel_error_mean": sequential_nn_results["test_k_rel_error_mean"],
            "test_c_rel_error_mean": sequential_nn_results["test_c_rel_error_mean"],
            "solve_time": sequential_nn_results["solve_time"],
        },
        {
            "method": "recursive_kernel_given_g",
            "solution_accepted": bool(recursive_kernel_results["solution_accepted"]),
            "g_hat": None,
            "g_hat_error": None,
            "log_tvc_residual": recursive_kernel_results["max_log_tvc_residual"],
            "transversality_ok": bool(recursive_kernel_results["transversality_ok"]),
            "test_k_rel_error_mean": recursive_kernel_results[
                "test_mean_abs_k_rel_error"
            ],
            "test_c_rel_error_mean": recursive_kernel_results[
                "test_mean_abs_c_rel_error"
            ],
            "solve_time": recursive_kernel_results["solve_time"],
        },
        {
            "method": "recursive_nn_given_g",
            "solution_accepted": bool(recursive_nn_results["solution_accepted"]),
            "g_hat": None,
            "g_hat_error": None,
            "log_tvc_residual": recursive_nn_results["max_log_tvc_residual"],
            "transversality_ok": bool(recursive_nn_results["transversality_ok"]),
            "test_k_rel_error_mean": recursive_nn_results["test_mean_abs_k_rel_error"],
            "test_c_rel_error_mean": recursive_nn_results["test_mean_abs_c_rel_error"],
            "solve_time": recursive_nn_results["solve_time"],
        },
    ]
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    with open(out / f"{PREFIX}.json", "w") as f:
        json.dump(
            {"g": g, "k_0": k_0, "z_0": z_0, "seed": seed, "methods": comparison},
            f,
            indent=2,
        )

    test_paths = {
        "Sequential kernel (learned $\\hat{g}$)": sequential_kernel_frames["test_df"],
        "Sequential NN (learned $\\hat{g}$)": sequential_nn_frames["test_df"],
        "Recursive kernel (given $g$)": recursive_kernel_frames["test_df"],
        "Recursive NN (given $g$)": recursive_nn_frames["test_df"],
    }
    paths_df = pd.concat(
        [df.assign(method=name) for name, df in test_paths.items()],
        ignore_index=True,
    )
    paths_df.to_parquet(out / f"{PREFIX}_test_df.parquet")

    style(figsize=(15, 8), fontsize=17, ticksize=16)
    plt.figure(figsize=(15, 8))

    ax_k = plt.subplot(1, 2, 1)
    benchmark = sequential_kernel_frames["test_df"].sort_values("t")
    ax_k.plot(benchmark["t"], benchmark["k_benchmark"], "k--", label="Closed form")
    for name, df in test_paths.items():
        df = df.sort_values("t")
        ax_k.plot(df["t"], df["k"], label=name, alpha=0.8)
    ax_k.set_xlabel("Time")
    ax_k.set_ylabel("Capital: $k(t)$")
    ax_k.legend()

    ax_err = plt.subplot(1, 2, 2)
    for name, df in test_paths.items():
        df = df.sort_values("t")
        ax_err.semilogy(df["t"], df["k_rel_error"].abs(), label=name, alpha=0.8)
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("Capital relative error vs. closed form")
    ax_err.legend()

    plt.tight_layout()
    figure_path = out / f"{PREFIX}.pdf"
    plt.savefig(figure_path, format="pdf")
    plt.close()
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
