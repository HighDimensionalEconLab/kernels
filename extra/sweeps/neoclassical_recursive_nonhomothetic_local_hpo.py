# Local random-search HPO for the non-homothetic recursive neoclassical
# drivers on the single-IC g = 0 experiment: the NN (capacity, epoch schedule,
# collocation size, multi-start budget, LM damping) over NN seeds 0-4, or the
# kernel (collocation size, ray + ARD geometry, metric shift, ridge, stop)
# on data seed 0. Both drivers run at their production acceptance bars -- the
# NN's predicate drives multi-start acceptance and early stopping inside the
# optimizer, so loosening it would change the algorithm being tuned -- and
# `report` re-applies the same bars to the recorded metrics. One row per
# (config, seed) appends to the parquet after every config, so a rerun with the
# same arguments resumes; every config recompiles, so bound a process with
# max_new and loop it from the shell. Parallel workers take distinct
# `--output`, `--sweep_seed`, and `--config_offset`; `report` globs them.
#
#   JAX_PLATFORMS=cpu uv run python -m extra.sweeps.neoclassical_recursive_nonhomothetic_local_hpo run --driver nn --max_new 10
#   uv run python -m extra.sweeps.neoclassical_recursive_nonhomothetic_local_hpo report --driver nn
from __future__ import annotations

import glob
import pathlib
import traceback

import jsonargparse
import numpy as np
import pandas as pd

from models.neoclassical_growth_kernel_recursive_nonhomothetic import (
    neoclassical_growth_kernel_recursive_nonhomothetic,
)
from models.neoclassical_growth_nn_recursive_nonhomothetic import (
    neoclassical_growth_nn_recursive_nonhomothetic,
)
from replication.utilities import configure_process, run_solver, with_option

# The replication's single-IC g = 0 experiment.
EXPERIMENT = {
    "verbose": False,
    "g": 0.0,
    "ic_set.data_seed": 0,
    "ic_set.num_train_trajectories": 1,
    "ic_set.num_test_trajectories": 1,
    "ic_set.train_log_z_variance": 0.0,
    "ic_set.train_log_k_variance": 0.0,
    "ic_set.test_log_z_variance": 0.0,
    "ic_set.test_log_k_variance": 0.0,
}
COMMON_METRICS = (
    "solution_accepted",
    "rejection_reason",
    "test_euler_mean_abs",
    "max_test_euler",
    "test_mean_abs_k_rel_error",
    "test_max_abs_k_rel_error",
    "max_log_tvc_residual",
    "train_loss",
    "steps",
    "solve_time",
)
SPACES = {
    "nn": dict(
        solver=neoclassical_growth_nn_recursive_nonhomothetic,
        seed_option="seed",
        seeds=(0, 1, 2, 3, 4),
        discrete={
            "nn_set.mlp_width": (8, 16, 32),
            "nn_set.mlp_depth": (2, 3),
            "nn_set.hidden_activation": ("silu", "gelu", "tanh"),
            "tolerance_set.max_steps": (125, 250, 500),
            "tolerance_set.num_epochs": (1, 2, 4),
            "data_set.train_points": (21, 41, 61),
            "lm_set.num_attempts": (2, 3, 5),
        },
        log_uniform={
            "nn_set.head_init_scale": (1e-4, 1e-1),
            "lm_set.init_damping": (1e-7, 1e-2),
            "lm_set.damping_increase": (1.5, 10.0),
            "tolerance_set.train_loss_threshold": (1e-14, 1e-11),
        },
        uniform={
            "lm_set.damping_decrease": (0.1, 0.9),
            "lm_set.geodesic_acceptance_ratio": (0.5, 0.99),
        },
        fixed={"lm_set.geodesic_acceleration": True},
        metrics=(*COMMON_METRICS, "attempts_run", "epochs_run", "total_params"),
        # The production bars of the driver's ToleranceSettings.
        bars=(3e-6, 9e-5),
        size_metric="total_params",
    ),
    "kernel": dict(
        solver=neoclassical_growth_kernel_recursive_nonhomothetic,
        seed_option="ic_set.data_seed",
        # Deterministic given the data draw, which the single-IC zero variances
        # pin at the base point.
        seeds=(0,),
        discrete={"data_set.train_points": (21, 41, 61, 81)},
        log_uniform={
            "kernel_set.ray_amplitude": (0.1, 5.0),
            "kernel_set.ray_lengthscale": (0.1, 5.0),
            "kernel_set.amplitude": (0.1, 5.0),
            "kernel_set.log_z_lengthscale": (0.1, 5.0),
            "kernel_set.log_k_lengthscale": (0.1, 5.0),
            "kernel_set.metric_epsilon": (1e-7, 1e-2),
            "ridge_set.ridge": (1e-13, 1e-9),
            "lm_set.atol": (1e-6, 1e-4),
        },
        uniform={"kernel_set.ray_slope": (0.5, 3.0)},
        fixed={"lm_set.gtol": 1e-7, "lm_set.max_steps": 300},
        metrics=(*COMMON_METRICS, "rkhs_norm", "num_centers"),
        bars=(3e-5, 3e-4),
        size_metric="num_centers",
    ),
}
# Designs above this many centers are skipped (binds only for a multi-IC
# design, where num_train_trajectories * train_points can exceed it).
MAX_CENTERS = 400


def fields(space):
    return [*space["discrete"], *space["log_uniform"], *space["uniform"]]


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_random(rng, n_configs, space, offset):
    configs = []
    for i in range(n_configs):
        config = {"config": offset + i}
        for key, values in space["discrete"].items():
            config[key] = values[int(rng.integers(len(values)))]
        for key, (low, high) in space["log_uniform"].items():
            config[key] = log_uniform(rng, low, high)
        for key, (low, high) in space["uniform"].items():
            config[key] = float(rng.uniform(low, high))
        configs.append(config)
    return configs


# Stage 2: jitter the continuous fields of the top configurations, keeping
# their discrete fields: log-normal factor (sd log 1.5) on the log-uniform
# fields, normal with sd a tenth of the range on the uniform ones, clipped to
# the sampling range.
def sample_around(rng, n_configs, top_configs, space, offset):
    configs = []
    for i in range(n_configs):
        base = top_configs.iloc[rng.integers(len(top_configs))]
        config = {"config": offset + i, **{key: base[key] for key in fields(space)}}
        for key, values in space["discrete"].items():
            config[key] = type(values[0])(config[key])
        for key in space["log_uniform"]:
            config[key] = float(config[key] * np.exp(rng.normal(0.0, np.log(1.5))))
        for key, (low, high) in space["uniform"].items():
            config[key] = float(
                np.clip(config[key] + rng.normal(0.0, 0.1 * (high - low)), low, high)
            )
        configs.append(config)
    return configs


def bar_accepted(frame, bars):
    mean_bar, max_bar = bars
    return (frame["test_euler_mean_abs"] <= mean_bar) & (
        frame["max_test_euler"] <= max_bar
    )


def aggregate(frame, space):
    rows = []
    for config, group in frame.groupby("config", sort=True):
        row = {key: group[key].iloc[0] for key in ("config", *fields(space))}
        row["n_seeds"] = len(group)
        row["n_solver_accepted"] = int(group["solution_accepted"].sum())
        accepted = bar_accepted(group, space["bars"])
        row["n_accepted"] = int(accepted.sum())
        row[space["size_metric"]] = group[space["size_metric"]].max()
        # Statistics over all seeds; a rejected seed keeps its raw metrics, so
        # the worst-seed columns show how far a near miss sits from the bar.
        for metric in (
            "test_euler_mean_abs",
            "max_test_euler",
            "test_mean_abs_k_rel_error",
            "test_max_abs_k_rel_error",
            "max_log_tvc_residual",
            "solve_time",
            "steps",
        ):
            row[f"worst_{metric}"] = group[metric].max()
            row[f"median_{metric}"] = group[metric].median()
        rows.append(row)
    return pd.DataFrame(rows)


def output_path(driver):
    return (
        f".results/hpo/neoclassical_recursive_nonhomothetic_{driver}_local_hpo.parquet"
    )


def run(
    driver: str = "nn",
    output: str | None = None,
    n_configs: int = 150,
    sweep_seed: int = 0,
    config_offset: int | None = None,
    around: str | None = None,
    # pandas query on the aggregated frame restricting the jitter targets,
    # e.g. "`tolerance_set.num_epochs` == 1".
    around_query: str | None = None,
    top: int = 5,
    max_new: int | None = None,
):
    configure_process()
    space = SPACES[driver]
    path = pathlib.Path(output or output_path(driver))
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sweep_seed)
    if around is None:
        configs = sample_random(
            rng, n_configs, space, 0 if config_offset is None else config_offset
        )
    else:
        summary = aggregate(read_frames(around), space)
        summary = summary[summary["n_accepted"] == summary["n_seeds"]]
        if around_query is not None:
            summary = summary.query(around_query)
        top_configs = summary.nsmallest(top, "worst_test_euler_mean_abs")
        if len(top_configs) == 0:
            raise RuntimeError(f"no fully accepted config in {around} to jitter")
        configs = sample_around(
            rng,
            n_configs,
            top_configs,
            space,
            1000 if config_offset is None else config_offset,
        )
    frame = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(frame["config"]) if len(frame) else set()
    new = 0
    for config in configs:
        if config["config"] in done:
            continue
        if max_new is not None and new == max_new:
            break
        if (
            EXPERIMENT["ic_set.num_train_trajectories"]
            * config["data_set.train_points"]
            > MAX_CENTERS
        ):
            continue
        new += 1
        options = {}
        for key, value in {**EXPERIMENT, **space["fixed"], **config}.items():
            if key != "config":
                options = with_option(options, key, value)
        rows = []
        for seed in space["seeds"]:
            row = {**config, "seed": seed}
            try:
                results, _ = run_solver(
                    space["solver"],
                    with_option(options, space["seed_option"], seed),
                )
                row.update({key: results[key] for key in space["metrics"]})
            except Exception as exc:
                traceback.print_exc()
                row.update(
                    solution_accepted=False,
                    rejection_reason="crashed",
                    note=f"{type(exc).__name__}: {exc}"[:200],
                )
            rows.append(row)
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        frame.to_parquet(path)
        worst_mean = max(float(row.get("test_euler_mean_abs", np.nan)) for row in rows)
        worst_max = max(float(row.get("max_test_euler", np.nan)) for row in rows)
        accepted = int(bar_accepted(pd.DataFrame(rows), space["bars"]).sum())
        print(
            f"config {config['config']}: accepted {accepted}/{len(rows)} "
            f"worst held-out mean {worst_mean:.2e} max {worst_max:.2e} "
            f"points {config['data_set.train_points']} "
            f"time {max(float(row.get('solve_time', np.nan)) for row in rows):.1f}s",
            flush=True,
        )
    return frame


def read_frames(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def report(driver: str = "nn", output: str | None = None, top: int = 15):
    space = SPACES[driver]
    summary = aggregate(read_frames(output or output_path(driver)), space)
    full = summary[summary["n_accepted"] == summary["n_seeds"]]
    columns = [
        "config",
        "n_accepted",
        "n_solver_accepted",
        *fields(space),
        space["size_metric"],
        "worst_test_euler_mean_abs",
        "median_test_euler_mean_abs",
        "worst_max_test_euler",
        "worst_test_mean_abs_k_rel_error",
        "median_solve_time",
        "median_steps",
    ]
    mean_bar, max_bar = space["bars"]
    print(
        f"{len(summary)} configs, {len(full)} accepted on every seed under the "
        f"bars (mean {mean_bar:.0e}, max {max_bar:.0e}), "
        f"{int(summary['n_seeds'].sum())} solves"
    )
    with pd.option_context(
        "display.width", 400, "display.float_format", "{:.3g}".format
    ):
        print("\nby worst-seed held-out Euler mean:")
        print(
            full.nsmallest(top, "worst_test_euler_mean_abs")[columns].to_string(
                index=False
            )
        )
        print("\nby worst-seed mean relative capital error:")
        print(
            full.nsmallest(top, "worst_test_mean_abs_k_rel_error")[columns].to_string(
                index=False
            )
        )
        near = summary[
            (summary["n_accepted"] < summary["n_seeds"])
            & (summary["worst_test_euler_mean_abs"] <= 3 * mean_bar)
            & (summary["worst_max_test_euler"] <= 3 * max_bar)
        ]
        print(f"\nnear misses (within 3x of the bars on the worst seed): {len(near)}")
        print(
            near.nsmallest(top, "worst_test_euler_mean_abs")[columns].to_string(
                index=False
            )
        )
    return summary


if __name__ == "__main__":
    jsonargparse.CLI([run, report])
