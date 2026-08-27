# Local grid HPO for the multicountry growth solvers. Kernel: lengthscale x
# kernel family x grid size, each scored over the three bond configurations
# (baseline + mirror debt/asset positions) on acceptance, residuals, the
# common-gamma bond-invariance gap, and solve time. NN: width x depth x
# damping schedule over the seed ensemble, scored on accepted fraction,
# residuals, and train time. Winners are promoted to the solver dataclass
# defaults by hand.
#
#   JAX_PLATFORMS=cpu uv run python -m extra.sweeps.multicountry_growth_hpo
from __future__ import annotations

import itertools
import pathlib

import jsonargparse
import numpy as np
import pandas as pd

from models.multicountry_growth_kernel import (
    DataSettings,
    KernelSettings,
    multicountry_growth_kernel,
)
from models.multicountry_growth_nn import (
    SequencePathSettings,
    multicountry_growth_nn,
)
from models.multicountry_growth_shared import (
    MULTICOUNTRY_GROWTH_CONSTANTS,
    InitialConditionSettings,
    draw_initial_conditions,
)
from models.utilities import OptimizerSettings
from replication.utilities import configure_process

KERNEL_GRID = {
    "kernel": ("matern12", "matern32"),
    "ell": (10.0, 20.0, 40.0, 80.0),
    "train_points": (128, 192),
}
NN_GRID = {
    "width": (32, 48),
    "depth": (3, 4),
    "damping": ("oe", "shared"),
}
# Solve fields pin the multicountry NN driver's defaults; only the damping
# schedule varies.
DAMPING_SCHEDULES = {
    "oe": OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.5,
        damping_increase=2.0,
        max_steps=400,
        atol=1e-07,
        num_attempts=5,
    ),
    "shared": OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.35,
        damping_increase=8.0,
        max_steps=400,
        atol=1e-07,
        num_attempts=5,
    ),
}


def kernel_sweep(output):
    drawn_k_0, _ = draw_initial_conditions(
        InitialConditionSettings(), MULTICOUNTRY_GROWTH_CONSTANTS["populations"]
    )
    rows = []
    for kernel, ell, train_points in itertools.product(*KERNEL_GRID.values()):
        row = {"kernel": kernel, "ell": ell, "train_points": train_points}
        runs = {}
        try:
            results, extras = multicountry_growth_kernel(
                kernel_set=KernelSettings(kernel=kernel, ell=ell),
                data_set=DataSettings(train_points=train_points),
            )
            runs["baseline"] = (results, extras["validation_df"])
            # Bond configurations sized by the baseline run's own
            # steady-state output scale.
            half_output = 0.5 * results["output_scale"]
            bond_configurations = {
                "initial_debt": dict(k_0=drawn_k_0, a_0=(-half_output, half_output)),
                "initial_assets": dict(k_0=drawn_k_0, a_0=(half_output, -half_output)),
            }
            for name, overrides in bond_configurations.items():
                results, extras = multicountry_growth_kernel(
                    kernel_set=KernelSettings(kernel=kernel, ell=ell),
                    data_set=DataSettings(train_points=train_points),
                    **overrides,
                )
                runs[name] = (results, extras["validation_df"])
        except Exception as exc:
            rows.append({**row, "accepted": False, "note": f"{type(exc).__name__}"})
            print(f"{row}: {type(exc).__name__}: {exc}")
            continue
        reference = runs["baseline"][1]
        invariance = 0.0
        for name in ("initial_debt", "initial_assets"):
            frame = runs[name][1]
            for column in ("interest_rate", "k_1", "k_2", "q_1", "q_2"):
                invariance = max(
                    invariance,
                    float(np.max(np.abs(frame[column] - reference[column]))),
                )
        row.update(
            accepted=all(results["solution_accepted"] for results, _ in runs.values()),
            max_train_residual=max(
                results["max_train_residual"] for results, _ in runs.values()
            ),
            max_validation_residual=max(
                results["max_validation_residual"] for results, _ in runs.values()
            ),
            invariance_gap=invariance,
            mean_solve_time=float(
                np.mean([results["solve_time"] for results, _ in runs.values()])
            ),
        )
        rows.append(row)
        print(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "multicountry_growth_kernel_hpo.parquet")
    if "max_validation_residual" not in frame:
        print("\nno kernel configuration completed:")
        print(frame.to_string(index=False))
        return frame
    ranked = frame[frame["accepted"].fillna(False)].sort_values(
        ["max_validation_residual", "invariance_gap"]
    )
    print("\nkernel ranking (accepted only):")
    print(ranked.to_string(index=False))
    return frame


def nn_sweep(output, n_seeds):
    rows = []
    for width, depth, damping in itertools.product(*NN_GRID.values()):
        row = {"width": width, "depth": depth, "damping": damping}
        seed_results = []
        try:
            for seed in range(n_seeds):
                results, _ = multicountry_growth_nn(
                    seed=seed,
                    nn_set=SequencePathSettings(width=width, depth=depth),
                    lm_set=DAMPING_SCHEDULES[damping],
                    verbose=False,
                )
                seed_results.append(results)
        except Exception as exc:
            rows.append({**row, "accepted_fraction": 0.0, "note": type(exc).__name__})
            print(f"{row}: {type(exc).__name__}: {exc}")
            continue
        accepted = [r for r in seed_results if r["solution_accepted"]]
        row.update(
            accepted_fraction=len(accepted) / n_seeds,
            max_train_residual_median=(
                float(np.median([r["max_train_residual"] for r in accepted]))
                if accepted
                else np.nan
            ),
            train_time_median=(
                float(np.median([r["train_time"] for r in accepted]))
                if accepted
                else np.nan
            ),
        )
        rows.append(row)
        print(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "multicountry_growth_nn_hpo.parquet")
    if "max_train_residual_median" not in frame:
        print("\nno nn configuration completed:")
        print(frame.to_string(index=False))
        return frame
    ranked = frame.sort_values(
        ["accepted_fraction", "max_train_residual_median"],
        ascending=[False, True],
    )
    print("\nnn ranking:")
    print(ranked.to_string(index=False))
    return frame


def run(
    output_dir: str = ".results/hpo",
    n_seeds: int = 7,
    include_kernel: bool = True,
    include_nn: bool = True,
):
    configure_process()
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if include_kernel:
        kernel_sweep(output)
    if include_nn:
        nn_sweep(output, n_seeds)


if __name__ == "__main__":
    jsonargparse.CLI(run)
