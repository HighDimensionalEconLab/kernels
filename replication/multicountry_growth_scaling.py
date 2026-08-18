from __future__ import annotations

import gc
import statistics

import jax
import jax.numpy as jnp
import jsonargparse
import pandas as pd

from models import multicountry_growth_kernel
from models.multicountry_growth import InitialConditionSettings
from replication.utilities import (
    configure_process,
    run_solver,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_scaling"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel
# The dense whitened Cholesky is O(N^3) per step; past the cap it costs
# minutes per solve and the cg column stands alone.
COUNTRY_COUNTS = [2, 3, 5, 10, 15, 25, 40, 80, 100]
DENSE_MAX_COUNTRIES = 10
# Per-rung lognormal dispersion of the drawn populations and
# productivities; curvature stays common at gamma = 2 so the rungs remain
# within the calibrated preference class.
POPULATION_LOG_SIGMA = 0.15
PRODUCTIVITY_LOG_SIGMA = 0.1
KERNEL_SET = {"kernel": "matern12", "ell": 20.0}
NARROW_RESIDUAL_ATOL = 2e-7
WIDE_RESIDUAL_ATOL = 5e-8
WIDE_STOPPING_START = 25
WARM_TIMING_REPETITIONS = 2


# Nothing on the ladder is symmetric: populations and productivities are
# drawn per rung, and the solver draws seeded k_0/a_0 when they are not
# passed (identical countries make the cross-country dynamics degenerate).
# The parameter draws share the initial-condition seed. Shared with the
# gamma-common driver so its tangent runs on exactly the published rung.
def rung_parameters(num_countries):
    parameter_key = jax.random.key(InitialConditionSettings().seed)
    population_key, productivity_key = jax.random.split(
        jax.random.fold_in(parameter_key, num_countries)
    )
    return dict(
        gammas=(2.0,) * num_countries,
        populations=tuple(
            float(value)
            for value in jnp.exp(
                POPULATION_LOG_SIGMA
                * jax.random.normal(population_key, (num_countries,))
            )
        ),
        productivities=tuple(
            float(value)
            for value in jnp.exp(
                PRODUCTIVITY_LOG_SIGMA
                * jax.random.normal(productivity_key, (num_countries,))
            )
        ),
    )


def rung_lm_overrides(num_countries):
    # The converged residual floors at O(ridge * ||dual||) and the dual
    # norm grows with the cross-section, so the fixed 1e-5 train gate
    # needs a deeper ridge floor at the wide rungs; the cg build-shift
    # floor deepens with it so the preconditioned condition number
    # ~ floor/ridge stays small.
    if num_countries >= 100:
        deep = dict(ridge=1e-13, cg_damping_floor=3e-12)
    elif num_countries >= 25:
        deep = dict(ridge=3e-13, cg_damping_floor=1e-11)
    else:
        deep = {}
    # Wide systems need the stricter scaled-residual stop to descend from
    # the 1e-11 continuation level to their configured ridge floor. The
    # narrow CG systems use the looser reachable stop; 5e-8 would keep the
    # N=2 solve running after its pointwise residual already clears the
    # uniform 1e-5 numerical gate.
    residual_atol = (
        WIDE_RESIDUAL_ATOL
        if num_countries >= WIDE_STOPPING_START
        else NARROW_RESIDUAL_ATOL
    )
    return {"residual_atol": residual_atol, **deep}


def run(
    output_dir: str | None = None,
    country_counts: list[int] = COUNTRY_COUNTS,  # noqa: B006
    dense_max_countries: int = DENSE_MAX_COUNTRIES,
    use_float64: bool = True,
):
    # The ladder certifies the float64 solver's scaling; there is no float32
    # counterpart experiment.
    if not use_float64:
        print(f"{PREFIX} is float64-only; skipping")
        return
    configure_process()
    if output_dir is None:
        output_dir = ".results"
    meta = dict(
        n_seeds=1,
        residual_dtype="float64",
        kernel="matern12",
        ell=20.0,
        warm_timing_repetitions=WARM_TIMING_REPETITIONS,
    )
    experiments = set()
    for num_countries in country_counts:
        parameters = rung_parameters(num_countries)
        solvers = ["cholesky", "cg"] if num_countries <= dense_max_countries else ["cg"]
        for solver_name in solvers:
            name = f"{'dense' if solver_name == 'cholesky' else 'cg'}_N{num_countries}"
            print(f"{name} -> {output_dir}/", flush=True)
            options = {
                "parameters": parameters,
                "kernel_set": KERNEL_SET,
                "lm_set": {
                    "linear_solver": solver_name,
                    **rung_lm_overrides(num_countries),
                },
            }
            results, extras = run_solver(SOLVER, options)
            compile_results = pd.DataFrame([results])
            frames = {
                key: value
                for key, value in extras.items()
                if isinstance(value, pd.DataFrame)
            }
            hyperparameters = extras["hyperparameters"]
            if not compile_results["solution_accepted"].all():
                raise RuntimeError(
                    f"{name} was rejected: {compile_results['rejection_reason'].iloc[0]}"
                )
            timing_columns = [
                column
                for column in compile_results.select_dtypes("number")
                if column.endswith(("_time", "_seconds"))
            ]
            timing_rows = [
                {
                    "timing_phase": "compile",
                    "timing_run": 0,
                    **{
                        column: compile_results[column].iloc[0]
                        for column in timing_columns
                    },
                }
            ]
            warm_results = []
            for timing_run in range(1, WARM_TIMING_REPETITIONS + 1):
                timed_row, timed_extras = run_solver(SOLVER, options)
                timed_results = pd.DataFrame([timed_row])
                timed_hyperparameters = timed_extras["hyperparameters"]
                if timed_hyperparameters != hyperparameters:
                    raise RuntimeError(
                        f"{name} changed hyperparameters across timing runs"
                    )
                if not timed_results["solution_accepted"].all():
                    raise RuntimeError(
                        f"{name} timing run {timing_run} was rejected: "
                        f"{timed_results['rejection_reason'].iloc[0]}"
                    )
                warm_results.append(timed_results)
                timing_rows.append(
                    {
                        "timing_phase": "warm",
                        "timing_run": timing_run,
                        **{
                            column: timed_results[column].iloc[0]
                            for column in timing_columns
                        },
                    }
                )
            results_df = compile_results.copy()
            for column in timing_columns:
                results_df.loc[0, f"compile_{column}"] = compile_results[column].iloc[0]
                results_df.loc[0, column] = statistics.median(
                    result[column].iloc[0] for result in warm_results
                )
            results_df.loc[0, "timing_repetitions"] = WARM_TIMING_REPETITIONS
            frames["timing_df"] = pd.DataFrame(timing_rows)
            save_experiment(
                name,
                results_df,
                frames,
                output_dir,
                meta,
                prefix=PREFIX,
                hyperparameters=hyperparameters,
            )
            experiments.add(name)
            print(
                f"  compile {results_df['compile_solve_time'].iloc[0]:.2f}s "
                f"warm {results_df['solve_time'].iloc[0]:.2f}s "
                f"steps {int(results_df['steps'].iloc[0])}",
                flush=True,
            )
        # Every country count has a distinct compiled shape. Releasing the
        # completed rung keeps executable storage bounded before the next,
        # wider system while the persistent cache retains reusable binaries.
        jax.clear_caches()
        gc.collect()
    write_summary(output_dir, PREFIX, experiments=experiments)


if __name__ == "__main__":
    jsonargparse.CLI(run)
