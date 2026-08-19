from __future__ import annotations

import math

import jsonargparse

from models import neoclassical_growth_kernel_recursive
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    sweep,
    write_summary,
)

PREFIX = "neoclassical_growth_kernel_recursive"

SOLVER = neoclassical_growth_kernel_recursive.neoclassical_growth_kernel_recursive

SINGLE_IC = {
    "num_train_trajectories": 1,
    "num_test_trajectories": 1,
    "train_log_z_variance": 0.0,
    "train_log_k_variance": 0.0,
    "test_log_z_variance": 0.0,
    "test_log_k_variance": 0.0,
}
MULTI_IC = {
    "num_train_trajectories": 5,
    "num_test_trajectories": 10,
    "train_log_z_variance": 0.01,
    "train_log_k_variance": 0.01,
    "test_log_z_variance": 0.0025,
    "test_log_k_variance": 0.0025,
}
MULTI_IC_X4 = {**MULTI_IC, "num_train_trajectories": 20}
# Multi-IC homothetic runs fail fast on the base kernel geometry and retry a
# data seed under the longer, more regularized geometry.
RETRIES = ({"kernel_set": {"lengthscale": 1.1, "metric_epsilon": 3e-4}},)
# Non-homothetic kernels use the sweep-validated per-dimension lengthscales
# for the calibrated state geometry (k_ss ~ 3.6): at g = 0 a balanced
# (0.6, 0.65); at g = 0.02 (log z sweeps [0, 2] in training) both bandwidths
# stretch with the visited ranges.
NONHOM_G0 = {
    "homothetic_representation": False,
    "state_kernel": "ard_rbf",
    "linear_mean": True,
    "log_z_lengthscale": 0.6,
    "log_k_lengthscale": 0.65,
}
NONHOM_G02 = {**NONHOM_G0, "log_z_lengthscale": 0.8, "log_k_lengthscale": 1.3}
# Out-of-distribution test samples (g = 0 training): the ID baseline at the
# OOD sample size, a joint unequal (z, k) center shift, and a wider spread.
OOD_TESTS = {
    "id50": {},
    "shift_zk": {"test_z_0": math.exp(0.25), "test_k_0": math.exp(0.5)},
    "spread": {"test_log_z_variance": 0.04, "test_log_k_variance": 0.04},
}
OOD_BASE = {
    "num_test_trajectories": 50,
    "test_log_z_variance": 0.0025,
    "test_log_k_variance": 0.0025,
}
DIAGNOSTIC_COLUMNS = [
    "max_train_euler",
    "max_test_euler",
    "max_log_tvc_residual",
    "test_mean_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "steps",
    "solve_time",
]


def report(name, results_df):
    accepted = int(results_df["solution_accepted"].sum())
    print(f"  accepted {accepted}/{len(results_df)}")
    print(results_df[DIAGNOSTIC_COLUMNS].agg(["median", "max"]).to_string())


def run(
    output_dir: str | None = None,
    n_data_seeds: int = 1,
    use_float64: bool = True,
    experiments: list[str] | None = None,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(residual_atol=3e-4, gtol=1e-5, ridge=1e-8)
    kernel_extra = {} if use_float64 else dict(metric_epsilon=1e-4)
    meta = dict(
        n_data_seeds=n_data_seeds,
        residual_dtype="float64" if use_float64 else "float32",
    )

    # (g, ic overrides, kernel_set overrides, retry ladder) per data-seed sweep.
    all_experiments = {
        "multi_ic_g0": (0.0, MULTI_IC, None, RETRIES),
        "multi_ic_g02": (0.02, MULTI_IC, None, RETRIES),
        "multi_ic_g0_nonhom": (0.0, MULTI_IC, NONHOM_G0, ()),
        "multi_ic_g02_nonhom": (0.02, MULTI_IC, NONHOM_G02, ()),
        "multi_ic_g0_x4": (0.0, MULTI_IC_X4, None, RETRIES),
        "multi_ic_g02_x4": (0.02, MULTI_IC_X4, None, RETRIES),
        "multi_ic_g0_x4_nonhom": (0.0, MULTI_IC_X4, NONHOM_G0, ()),
        "multi_ic_g02_x4_nonhom": (0.02, MULTI_IC_X4, NONHOM_G02, ()),
    }
    for test_tag, test_ic in OOD_TESTS.items():
        all_experiments[f"ood_nonhom_n20_{test_tag}"] = (
            0.0,
            {**MULTI_IC_X4, **OOD_BASE, **test_ic},
            NONHOM_G0,
            (),
        )
    run_single = experiments is None or "single_ic_g0" in experiments
    if experiments is not None:
        unknown = set(experiments) - set(all_experiments) - {"single_ic_g0"}
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        all_experiments = {
            name: spec for name, spec in all_experiments.items() if name in experiments
        }

    # The kernel is deterministic given its data draw: the single-IC case runs
    # through the timing harness, while the multi-IC cases sweep the data seed.
    if run_single:
        print(f"single_ic_g0 -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            1,
            options={
                "use_float64": use_float64,
                "verbose": verbose,
                "g": 0.0,
                "ic_set": {"data_seed": 0, **SINGLE_IC},
                "stop_set": stop,
                "kernel_set": kernel_extra,
            },
        )
        save_experiment(
            "single_ic_g0",
            results_df,
            frames,
            output_dir,
            meta,
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )
        report("single_ic_g0", results_df)

    for name, (g, ic_set, kernel_set, retries) in all_experiments.items():
        print(f"{name}: {n_data_seeds} data seeds -> {output_dir}/")
        options = {
            "use_float64": use_float64,
            "verbose": verbose,
            "g": g,
            "ic_set": dict(ic_set),
            "stop_set": {"max_steps": 60, **stop},
            "kernel_set": dict(kernel_extra),
        }
        if kernel_set is not None:
            options["kernel_set"].update(kernel_set)
        results_df, frames, hyperparameters = sweep(
            SOLVER,
            "ic_set.data_seed",
            range(n_data_seeds),
            options=options,
            retries=retries,
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            meta,
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )
        report(name, results_df)

    write_summary(
        output_dir,
        PREFIX,
        experiments={"single_ic_g0", *all_experiments}
        if run_single
        else set(all_experiments),
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
