from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jsonargparse
import pandas as pd

import extra.trade_growth.kernel as trade_growth_kernel
import extra.trade_growth.nn as trade_growth_nn
from extra.trade_growth.model import symmetric_trade_growth_parameters
from replication.utilities import (
    compare_residual_norm_frames,
    enable_compilation_cache,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "multicountry_trade_growth_comparison"


def run(
    output_dir: str = ".results",
    use_float64: bool = True,
):
    seed = 0
    parameters = symmetric_trade_growth_parameters(3)
    kernel_results, kernel_frames = (
        trade_growth_kernel.multicountry_trade_growth_kernel(
            a_0=(-0.2, 0.1, 0.1),
            parameters=parameters,
            use_float64=use_float64,
            data_set=trade_growth_kernel.DataSettings(
                train_T=120.0,
                train_points=32,
                dense_T=40.0,
                tail_points=8,
                test_T=80.0,
                test_points=161,
                validation_points=241,
                quality_T=100.0,
            ),
            lm_set=trade_growth_kernel.SolverSettings(
                init_damping=1e-8,
                residual_atol=1e-10,
                max_steps=100,
            ),
        )
    )
    nn_results, nn_frames = trade_growth_nn.multicountry_trade_growth_nn(
        a_0=(-0.2, 0.1, 0.1),
        parameters=parameters,
        seed=seed,
        use_float64=use_float64,
        verbose=False,
    )
    quality_T = min(kernel_results["quality_T"], nn_results["quality_T"])
    comparison_results, comparison_frame = compare_residual_norm_frames(
        kernel_frames["validation_df"],
        nn_frames["validation_df"],
        quality_T,
    )
    comparison_results.update(
        {
            "seed": seed,
            "solution_accepted": (
                kernel_results["solution_accepted"] and nn_results["solution_accepted"]
            ),
            "kernel_solution_accepted": kernel_results["solution_accepted"],
            "nn_solution_accepted": nn_results["solution_accepted"],
            "kernel_solve_time": kernel_results["solve_time"],
            "nn_solve_time": nn_results["solve_time"],
            "nn_to_kernel_solve_time_ratio": (
                nn_results["solve_time"] / kernel_results["solve_time"]
            ),
        }
    )
    save_experiment(
        "three_country_initial_nfa",
        pd.DataFrame([comparison_results]),
        {"validation_df": comparison_frame},
        output_dir,
        {
            "backend": "cpu",
            "residual_dtype": "float64" if use_float64 else "float32",
        },
        prefix=PREFIX,
        hyperparameters={
            "kernel": kernel_frames["hyperparameters"],
            "nn": nn_frames["hyperparameters"],
        },
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
