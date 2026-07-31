from __future__ import annotations

import jax
import jsonargparse
import numpy as np

import neoclassical_growth_concave_convex_nn_recursive
from replication.utilities import (
    enable_compilation_cache,
    save_experiment,
    sweep,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_concave_convex_nn_recursive"


def run(output_dir: str = ".results", verbose: bool = False):
    n_points = 70
    grid = np.linspace(0.5, 4.0, n_points)
    print(f"threshold sweep: {n_points} initial conditions -> {output_dir}/")
    # No ex-post classification here: a recursive policy can be rolled past the
    # training horizon, so the solver module settles it to settle_T and emits
    # basin_correct / false_positive / basin_failed itself.
    results_df, frames, hyperparameters = sweep(
        neoclassical_growth_concave_convex_nn_recursive.neoclassical_growth_concave_convex_nn_recursive,
        "k_0",
        [float(k_0) for k_0 in grid],
        options={"verbose": verbose},
    )
    for key in ("basin_correct", "false_positive", "basin_failed"):
        results_df[key] = results_df[key].fillna(False).astype(bool)
    # A grid point that raised carries no columns of its own; the thresholds are
    # constant across the sweep, so fill them from the points that solved. The
    # solver reports them in the detrended state kappa = k/z, which coincides
    # with k here since the sweep runs at g = 0 and z_0 = 1.
    for key, kappa_key in (
        ("k_thr", "kappa_thr"),
        ("k_ss_low", "kappa_ss_low"),
        ("k_ss_high", "kappa_ss_high"),
    ):
        results_df[key] = results_df[kappa_key].ffill().bfill()

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
        dict(n_seeds=1, n_points=n_points),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX, experiments={"threshold"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
