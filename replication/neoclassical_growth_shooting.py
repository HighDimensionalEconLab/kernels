from __future__ import annotations

import json
import pathlib

import jsonargparse

from models import neoclassical_growth_shooting
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "neoclassical_growth_shooting"

SOLVER = neoclassical_growth_shooting.neoclassical_growth_shooting

# Newton from random guesses by relative radius around the benchmark c_0 and
# horizon; its basin of convergence shrinks like e^{-lambda_u T}.
NEWTON_RADII = [1e-8, 1e-4, 1e-1]
NEWTON_HORIZONS = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0]
NEWTON_MAX_ITER = 50
# Bisection with the boundary rule converges from any guess, so one radius:
# the widest whose guesses do not start by decumulating capital (the bound
# f(k_0) - delta k_0 is a radius of 0.34).
BISECTION_RADIUS = 0.3
BISECTION_HORIZONS = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0, 300.0]
BISECTION_MAX_ITER = 60
GRADIENT_HORIZONS = BISECTION_HORIZONS
BASIN_HORIZONS = NEWTON_HORIZONS
# Relative offsets of c_0 above the benchmark value for the elasticity table.
C_0_OFFSETS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# The derivative-seminorm table (models.neoclassical_growth_shooting.shooting_seminorms)
# is not part of the paper; its offsets at and below c_0^* are kept for reference.
# SEMINORM_OFFSETS = [0.0, -1e-6, -1e-5, -1e-4, -1e-3, -1e-2, -1e-1]


def cell_name(method, radius, T):
    return f"{method}_r{radius:g}_T{T:g}"


def run(output_dir: str = ".results", n_draws: int = 20):
    configure_process()

    def go(method, radius, T, max_iter):
        name = cell_name(method, radius, T)
        print(f"{name}: {n_draws} draws -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_draws,
            options={
                "shooting_set": {
                    "method": method,
                    "radius": radius,
                    "T": T,
                    "max_iter": max_iter,
                }
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            dict(n_seeds=n_draws, residual_dtype="float64"),
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    for radius in NEWTON_RADII:
        for T in NEWTON_HORIZONS:
            go("newton", radius, T, NEWTON_MAX_ITER)
    for T in BISECTION_HORIZONS:
        go("bisection", BISECTION_RADIUS, T, BISECTION_MAX_ITER)

    out = pathlib.Path(output_dir)
    basins, basin_meta = neoclassical_growth_shooting.newton_basins(BASIN_HORIZONS)
    basins.to_parquet(out / f"{PREFIX}_basins.parquet", index=False)
    gradients, meta = neoclassical_growth_shooting.shooting_gradients(
        C_0_OFFSETS, GRADIENT_HORIZONS
    )
    gradients.to_parquet(out / f"{PREFIX}_gradients.parquet", index=False)
    # seminorms, _ = neoclassical_growth_shooting.shooting_seminorms(
    #     SEMINORM_OFFSETS, GRADIENT_HORIZONS
    # )
    # seminorms.to_parquet(out / f"{PREFIX}_seminorms.parquet", index=False)
    (out / f"{PREFIX}_gradients.json").write_text(
        json.dumps(
            {
                **meta,
                "newton_radii": NEWTON_RADII,
                "newton_horizons": NEWTON_HORIZONS,
                "newton_max_iter": NEWTON_MAX_ITER,
                "bisection_radius": BISECTION_RADIUS,
                "bisection_horizons": BISECTION_HORIZONS,
                "bisection_max_iter": BISECTION_MAX_ITER,
                "gradient_horizons": GRADIENT_HORIZONS,
                "basin_horizons": BASIN_HORIZONS,
                "benchmark_T": neoclassical_growth_shooting.BENCHMARK_T,
                "basin_offsets": basin_meta["basin_offsets"],
                "c_0_offsets": C_0_OFFSETS,
            }
        )
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
