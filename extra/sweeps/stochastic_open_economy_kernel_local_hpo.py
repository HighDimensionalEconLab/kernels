# Local random-search HPO for the stochastic open-economy kernel-recursive
# distillation: the frozen design (trajectory count x grid density), the
# kernel geometry, and the ridge LM stop, scored on the design seeds by the
# held-out per-output Euler errors and the capital-path error against the
# reference NN. One row per (config, seed) appends to the parquet after every
# config, so a rerun with the same arguments resumes; every config recompiles
# the solver (the kernel settings are static), so bound a process with
# max_new and loop it from the shell to keep the resident executables small.
#
#   JAX_PLATFORMS=cpu uv run python -m extra.sweeps.stochastic_open_economy_kernel_local_hpo run --max_new 10
#   uv run python -m extra.sweeps.stochastic_open_economy_kernel_local_hpo report
from __future__ import annotations

import pathlib
import traceback

import jsonargparse
import numpy as np
import pandas as pd

from models.stochastic_open_economy_growth_kernel_recursive import (
    stochastic_open_economy_growth_kernel_recursive,
)
from replication.utilities import configure_process, run_solver, with_option

DISCRETE = {
    "ic_set.num_train_trajectories": (2, 4, 8, 16),
    "data_set.train_points": (11, 17, 21, 33, 41),
    "kernel_set.state_kernel": ("ard_rbf", "ard_matern52"),
    "kernel_set.free_constant": (False, True),
    "kernel_set.linear_mean": (False, True),
}
LOG_UNIFORM = {
    "kernel_set.lengthscale": (0.7, 5.0),
    "kernel_set.asset_lengthscale": (0.1, 5.0),
    "kernel_set.amplitude": (0.3, 5.0),
    "kernel_set.metric_epsilon": (1e-7, 1e-2),
    "ridge_set.ridge": (1e-13, 1e-9),
    "lm_set.atol": (1e-6, 1e-4),
}
# log_k_lengthscale is sampled as a ratio to the shared lengthscale.
LOG_K_RATIO = (0.3, 3.0)
# Designs above this many centers are skipped: the forward-mode Jacobian of
# the Ito terms scales as centers^2 x theta and exhausts memory at 16 x 41.
MAX_CENTERS = 400
# The gates only catch simulation blow-ups; the tables rank on the errors.
FIXED = {
    "lm_set.gtol": 1e-9,
    "lm_set.max_steps": 500,
    "tolerance_set.test_euler_mean_abs": 5e-2,
    "tolerance_set.test_euler_max": 1.0,
}
METRICS = (
    "solution_accepted",
    "rejection_reason",
    "test_c_euler_mean_abs",
    "test_q_euler_mean_abs",
    "test_euler_max_abs",
    "test_mean_abs_k_rel_error",
    "test_max_abs_k_rel_error",
    "test_mean_abs_c_rel_error",
    "train_euler_mean_abs",
    "steps",
    "solve_time",
    "rkhs_coefficients",
)
FIELDS = [*DISCRETE, *LOG_UNIFORM, "kernel_set.log_k_lengthscale"]


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_random(rng, n_configs):
    configs = []
    for i in range(n_configs):
        config = {"config": i}
        for key, values in DISCRETE.items():
            config[key] = values[int(rng.integers(len(values)))]
        for key, (low, high) in LOG_UNIFORM.items():
            config[key] = log_uniform(rng, low, high)
        config["kernel_set.log_k_lengthscale"] = config[
            "kernel_set.lengthscale"
        ] * log_uniform(rng, *LOG_K_RATIO)
        configs.append(config)
    return configs


# Stage 2: jitter the continuous fields of the top configurations by a
# log-normal factor (sd log 1.5), keeping their discrete fields.
def sample_around(rng, n_configs, top_configs):
    configs = []
    continuous = [*LOG_UNIFORM, "kernel_set.log_k_lengthscale"]
    for i in range(n_configs):
        base = top_configs.iloc[rng.integers(len(top_configs))]
        config = {"config": 1000 + i, **{key: base[key] for key in FIELDS}}
        for key in DISCRETE:
            config[key] = type(DISCRETE[key][0])(config[key])
        for key in continuous:
            config[key] = float(config[key] * np.exp(rng.normal(0.0, np.log(1.5))))
        configs.append(config)
    return configs


def aggregate(frame):
    rows = []
    for config, group in frame.groupby("config", sort=True):
        row = {key: group[key].iloc[0] for key in ("config", *FIELDS)}
        row["n_seeds"] = len(group)
        row["n_accepted"] = int(group["solution_accepted"].sum())
        row["rkhs_coefficients"] = group["rkhs_coefficients"].max()
        # Rejected seeds are restarts: the statistics cover accepted seeds only.
        group = group[group["solution_accepted"].astype(bool)]
        group = group.assign(
            test_euler_mean_abs=group[
                ["test_c_euler_mean_abs", "test_q_euler_mean_abs"]
            ].max(axis=1)
        )
        for metric in (
            "test_euler_mean_abs",
            "test_euler_max_abs",
            "test_mean_abs_k_rel_error",
            "test_max_abs_k_rel_error",
            "test_mean_abs_c_rel_error",
            "train_euler_mean_abs",
            "solve_time",
        ):
            row[f"worst_{metric}"] = group[metric].max()
            row[f"median_{metric}"] = group[metric].median()
        rows.append(row)
    return pd.DataFrame(rows)


OUTPUT = ".results/hpo/stochastic_open_economy_kernel_local_hpo.parquet"


def run(
    output: str = OUTPUT,
    n_configs: int = 300,
    seeds: tuple[int, ...] = (0, 1, 2),
    policy_dir: str = ".results",
    sweep_seed: int = 0,
    around: str | None = None,
    top: int = 5,
    min_accepted: int = 2,
    max_new: int | None = None,
):
    configure_process()
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sweep_seed)
    if around is None:
        configs = sample_random(rng, n_configs)
    else:
        summary = aggregate(pd.read_parquet(around))
        top_configs = summary[summary["n_accepted"] >= min_accepted].nsmallest(
            top, "worst_test_euler_mean_abs"
        )
        configs = sample_around(rng, n_configs, top_configs)
    frame = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    done = set(frame["config"]) if len(frame) else set()
    new = 0
    for config in configs:
        if config["config"] in done:
            continue
        if max_new is not None and new == max_new:
            break
        if (
            config["ic_set.num_train_trajectories"] * config["data_set.train_points"]
            > MAX_CENTERS
        ):
            continue
        new += 1
        options = {"verbose": False, "policy_dir": policy_dir}
        for key, value in {**FIXED, **config}.items():
            if key != "config":
                options = with_option(options, key, value)
        rows = []
        for seed in seeds:
            row = {**config, "data_seed": seed}
            try:
                results, _ = run_solver(
                    stochastic_open_economy_growth_kernel_recursive,
                    with_option(options, "ic_set.data_seed", seed),
                )
                row.update({key: results[key] for key in METRICS})
            except Exception as exc:
                traceback.print_exc()
                row.update(
                    solution_accepted=False,
                    rejection_reason="crashed",
                    note=f"{type(exc).__name__}: {exc}"[:200],
                )
            rows.append(row)
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        frame.to_parquet(output_path)
        worst = max(
            max(
                float(row.get("test_c_euler_mean_abs", np.nan)),
                float(row.get("test_q_euler_mean_abs", np.nan)),
            )
            for row in rows
        )
        accepted = sum(bool(row["solution_accepted"]) for row in rows)
        print(
            f"config {config['config']}: accepted {accepted}/{len(rows)} "
            f"worst held-out mean {worst:.2e} "
            f"paths {config['ic_set.num_train_trajectories']} "
            f"points {config['data_set.train_points']} "
            f"ell {config['kernel_set.lengthscale']:.2f}",
            flush=True,
        )
    return frame


def report(output: str = OUTPUT, top: int = 20, min_accepted: int = 2):
    summary = aggregate(pd.read_parquet(output))
    accepted = summary[summary["n_accepted"] >= min_accepted]
    columns = [
        "config",
        "n_accepted",
        "ic_set.num_train_trajectories",
        "data_set.train_points",
        "rkhs_coefficients",
        "kernel_set.state_kernel",
        "kernel_set.free_constant",
        "kernel_set.linear_mean",
        "kernel_set.lengthscale",
        "kernel_set.log_k_lengthscale",
        "kernel_set.asset_lengthscale",
        "kernel_set.amplitude",
        "kernel_set.metric_epsilon",
        "ridge_set.ridge",
        "lm_set.atol",
        "worst_test_euler_mean_abs",
        "median_test_euler_mean_abs",
        "worst_test_euler_max_abs",
        "worst_test_mean_abs_k_rel_error",
        "worst_test_max_abs_k_rel_error",
        "worst_solve_time",
    ]
    print(
        f"{len(summary)} configs, {len(accepted)} accepted on at least "
        f"{min_accepted} seeds, {int(summary['n_seeds'].sum())} solves"
    )
    with pd.option_context(
        "display.width", 250, "display.float_format", "{:.3g}".format
    ):
        print("\nby worst-seed held-out per-output Euler mean:")
        print(
            accepted.nsmallest(top, "worst_test_euler_mean_abs")[columns].to_string(
                index=False
            )
        )
        print("\nby worst-seed mean relative capital error:")
        print(
            accepted.nsmallest(top, "worst_test_mean_abs_k_rel_error")[
                columns
            ].to_string(index=False)
        )
    return summary


if __name__ == "__main__":
    jsonargparse.CLI([run, report])
