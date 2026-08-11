from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse

from models import multicountry_growth_kernel
from models.multicountry_growth import InitialConditionSettings
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_scaling"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel
# The dense whitened Cholesky is O(N^3) per step; past the cap it costs
# minutes per solve and the cg column stands alone.
COUNTRY_COUNTS = [2, 3, 5, 10, 15, 25, 40, 80, 100]
DENSE_MAX_COUNTRIES = 10
# The gamma-common tangent is published from N=40, where its value sits in
# one line with the N=2/N=10 sensitivity tables and the production
# acceptance gates hold with margin.
GAMMA_COMMON_COUNT = 40
# Per-rung lognormal dispersion of the drawn populations and
# productivities; curvature stays common at gamma = 2 so the rungs remain
# within the calibrated preference class.
POPULATION_LOG_SIGMA = 0.15
PRODUCTIVITY_LOG_SIGMA = 0.1


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
    meta = dict(n_seeds=1, residual_dtype="float64")
    experiments = set()
    # Nothing on the ladder is symmetric: populations and productivities are
    # drawn per rung, and the solver draws seeded k_0/a_0 when they are not
    # passed (identical countries make the cross-country dynamics
    # degenerate). The parameter draws share the initial-condition seed.
    parameter_key = jax.random.key(InitialConditionSettings().seed)
    for num_countries in country_counts:
        population_key, productivity_key = jax.random.split(
            jax.random.fold_in(parameter_key, num_countries)
        )
        parameters = dict(
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
        # The two widest rungs converge within a factor of two of the
        # float64 residual plateau (measured max train residual ~9e-6),
        # where run-to-run reduction ordering moves the max over ~10^5
        # residuals by tens of percent; their acceptance bar sits at 2e-5 so
        # the ladder fails on regressions, not on noise. Rungs through
        # N = 40 hold the production 1e-5 bar, and every bar is recorded in
        # the rung's saved hyperparameters.
        tolerances = {"train_residual": 2e-5} if num_countries >= 80 else {}
        solvers = ["cholesky", "cg"] if num_countries <= dense_max_countries else ["cg"]
        for solver_name in solvers:
            name = f"{'dense' if solver_name == 'cholesky' else 'cg'}_N{num_countries}"
            print(f"{name} -> {output_dir}/", flush=True)
            # The production residual_atol (5e-7) lets every rung stop one
            # continuation level above the ridge floor, where the max train
            # residual crosses the 1e-5 acceptance gate by N = 15. At 2e-7
            # the wide rungs are forced down to their (deepened) floors
            # while the narrow rungs still stop where their gate already
            # clears -- dense and cg run the same settings at every rung,
            # keeping the timing like-for-like.
            options = {
                "parameters": parameters,
                "lm_set": {
                    "linear_solver": solver_name,
                    "residual_atol": 2e-7,
                    **deep,
                },
                **({"tolerance_set": tolerances} if tolerances else {}),
            }
            # The N=40 cg rung also publishes d rate(T10)/d gamma along
            # (1, ..., 1) -- the representative-agent CRRA comparative
            # static, whose value is comparable against the N=2/N=10
            # sensitivity tables -- with the implicit-AD wall time reported
            # next to the primal solve time.
            if solver_name == "cg" and num_countries == GAMMA_COMMON_COUNT:
                options["gamma_common_sensitivity_times"] = [10.0]
            results_df, frames, hyperparameters = ensemble(
                SOLVER,
                n_seeds=1,
                options=options,
            )
            if not results_df["solution_accepted"].all():
                raise RuntimeError(
                    f"{name} was rejected: {results_df['rejection_reason'].iloc[0]}"
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
            experiments.add(name)
            print(
                f"  solve_time {results_df['solve_time'].iloc[0]:.2f}s "
                f"steps {int(results_df['steps'].iloc[0])}",
                flush=True,
            )
    write_summary(output_dir, PREFIX, experiments=experiments)


if __name__ == "__main__":
    jsonargparse.CLI(run)
