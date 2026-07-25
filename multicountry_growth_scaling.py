# Scaling ladder for the reduced multicountry growth kernel driver: for each
# N, run the production driver on fully heterogeneous inputs (drawn
# populations, productivities, and initial conditions) with the matrix-free
# cg path (block-eigen preconditioner) and optionally the dense cholesky
# reference, assert the standing validity bars, and report timings.
#
# Stopping on the ladder relies on the gtol calibration alone
# (residual_atol=0): the whitened floor stationarity ~ ridge*sqrt(q) ~ 2e-12
# sits below gtol=1e-11 while the last intermediate continuation level's
# ~6e-10 sits above it, so gtol can only fire at the ridge floor. The fixed
# production residual_atol=5e-8 is N=2/3-calibrated -- the O(ridge*||dual||)
# residual floor grows with N and would block the stopping certificate on a
# solve that passes every quality gate.
#
#   JAX_PLATFORMS=cpu uv run python -m multicountry_growth_scaling \
#     --country_counts '[2,3,10,25,50]'

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jsonargparse

from multicountry_growth import InitialConditionSettings, MulticountryParameters
from multicountry_growth_kernel import SolverSettings, multicountry_growth_kernel

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


# Long-run economic validity bars, applied at every rung on top of the
# driver's residual acceptance gates: by t ~ 100 consumption, capital, and
# debt are flat, capital sits at the theory steady state (MPK equalized at
# delta + rho/e), and r ~ rho; the exact one-good invariants -- constant
# consumption ratios and consumption proportional to initial total wealth
# a_0 + PV(resources) -- hold to solver accuracy. Bounds are ~3x the
# measured ladder values (flatness/theory gaps ~ 8e-3 with the a-drift
# reaching ~1.1e-2 at the deepest tail draws, ratio drift <= 7.5e-6,
# proportionality <= 1.7e-4). The wealth ranking is gated through
# wealth_rank_violation (the largest relative wealth separation among
# consumption-misordered pairs): exact rank equality is tie-fragile at
# N=50, where adjacent drawn wealths land within the proportionality
# spread of each other.
@dataclass(frozen=True)
class LongRunTolerances:
    c_flatness: float = 0.01
    k_flatness: float = 0.02
    a_flatness: float = 0.03
    k_theory_gap: float = 0.03
    mpk_equalization_gap: float = 0.02
    interest_rate_gap: float = 0.03
    consumption_ratio_drift: float = 1e-5
    wealth_proportionality_gap: float = 1e-3
    wealth_rank_violation: float = 1e-3


def run_ladder(
    country_counts: list[int] = [2, 3, 10, 25, 50],  # noqa: B006
    linear_solvers: list[str] = ["cg"],  # noqa: B006
    max_steps: int = 120,
    cg_maxiter: int = 2500,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    population_log_sigma: float = 0.15,
    productivity_log_sigma: float = 0.1,
    long_run_set: LongRunTolerances = LongRunTolerances(),
    output_path: str | None = None,
):
    # Nothing on the ladder is symmetric: populations, productivities, AND
    # the initial conditions are drawn per N (identical countries make the
    # cross-country dynamics degenerate and the ladder meaningless). All
    # seeds and dispersions are CLI-settable.
    parameter_key = jax.random.key(initial_set.seed)
    records = []
    for N in country_counts:
        population_key, productivity_key = jax.random.split(
            jax.random.fold_in(parameter_key, N)
        )
        parameters = MulticountryParameters(
            populations=tuple(
                float(value)
                for value in jnp.exp(
                    population_log_sigma * jax.random.normal(population_key, (N,))
                )
            ),
            productivities=tuple(
                float(value)
                for value in jnp.exp(
                    productivity_log_sigma * jax.random.normal(productivity_key, (N,))
                )
            ),
        )
        # The converged residual floors at O(ridge * ||dual||) and the dual
        # norm grows ~linearly in N (fully heterogeneous draws measure
        # ~1.1e-6 at N=2 up to ~1.6e-5 at N=50 at ridge 1e-12), so the floor
        # deepens with N to keep the FIXED train gate; gtol scales with the
        # floor so it still fires only at the final continuation level
        # (floor stationarity ~ ridge * sqrt(q) sits below it, the previous
        # level's above).
        if N >= 40:
            ridge, gtol = 3e-13, 5e-13
        elif N >= 25:
            ridge, gtol = 1e-12, 1.5e-12
        else:
            ridge, gtol = SolverSettings().ridge, SolverSettings().gtol
        for solver_name in linear_solvers:
            lm_set = SolverSettings(
                linear_solver=solver_name,
                residual_atol=0.0,
                max_steps=max_steps,
                cg_maxiter=cg_maxiter,
                ridge=ridge,
                gtol=gtol,
            )
            start = time.perf_counter()
            results, _ = multicountry_growth_kernel(
                parameters=parameters,
                initial_set=initial_set,
                lm_set=lm_set,
            )
            total_time = time.perf_counter() - start
            long_run_failures = [
                name
                for name, bound in asdict(long_run_set).items()
                if results[f"long_run_{name}"] > bound
            ]
            record = {
                "num_countries": N,
                "linear_solver": solver_name,
                "solve_time": results["solve_time"],
                "total_time": total_time,
                "steps": results["steps"],
                "accepted": results["solution_accepted"],
                "rejection_reason": results["rejection_reason"],
                "max_train_residual": results["max_train_residual"],
                "max_validation_residual": results["max_validation_residual"],
                "max_asset_clearing_residual": results["max_asset_clearing_residual"],
                "rkhs_norm_total": results["rkhs_norm_total"],
                "long_run_failures": long_run_failures,
                **{
                    f"long_run_{name}": results[f"long_run_{name}"]
                    for name in (
                        *asdict(long_run_set),
                        "c_rank_matches_wealth",
                        "c_rank_matches_a_0",
                        "terminal_a_rank_matches_a_0",
                    )
                },
            }
            records.append(record)
            print(
                f"N={N} {solver_name}: solve={record['solve_time']:.1f}s "
                f"total={total_time:.1f}s steps={record['steps']} "
                f"accepted={record['accepted']} "
                f"reason={record['rejection_reason']} "
                f"max_train={record['max_train_residual']:.2e} "
                f"long_run={long_run_failures or 'ok'}",
                flush=True,
            )
    failures = [
        record
        for record in records
        if not record["accepted"] or record["long_run_failures"]
    ]
    if output_path is not None:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2))
        print(f"wrote {path}")
    if failures:
        raise SystemExit(
            f"{len(failures)} ladder run(s) failed the validity bars: "
            + ", ".join(
                f"N={record['num_countries']}/{record['linear_solver']}"
                for record in failures
            )
        )
    return records


if __name__ == "__main__":
    jsonargparse.CLI(run_ladder)
