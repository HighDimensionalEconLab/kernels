from __future__ import annotations

import jsonargparse

from models import multicountry_growth_kernel
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
    for num_countries in country_counts:
        # Symmetric structural parameters: the rung isolates pure
        # cross-section width, with all heterogeneity carried by the drawn
        # initial conditions (the driver draws seeded k_0/a_0 when they are
        # not passed).
        parameters = dict(
            gammas=(2.0,) * num_countries,
            populations=(1.0,) * num_countries,
            productivities=(1.0,) * num_countries,
        )
        # The converged residual floors at O(ridge * ||dual||) and the dual
        # norm grows with the cross-section, so the fixed 1e-5 train gate
        # needs a deeper ridge floor at the wide rungs (measured max train
        # residual at ridge 3e-12: 5.1e-6 at N=15, 1.7e-5 at N=25, 2.6e-5
        # at N=40). The cg build-shift floor deepens with it so the
        # preconditioned condition number ~ floor/ridge stays small.
        if num_countries >= 100:
            deep = dict(ridge=1e-13, cg_damping_floor=3e-12)
        elif num_countries >= 40:
            deep = dict(ridge=3e-13, cg_damping_floor=1e-11)
        elif num_countries >= 25:
            deep = dict(ridge=1e-12, cg_damping_floor=1e-11)
        else:
            deep = {}
        # By N = 100 the max train residual plateaus at 1.5e-5 no matter how
        # far the ridge deepens -- the softest whitened modes sit below the
        # float64 matvec noise -- so the max-residual gate relaxes to 2e-5
        # there (the scaled RMS residual stays flat at ~1e-7). Every rung
        # through N = 80 holds the production 1e-5 bar; the relaxed bar is
        # recorded in the rung's saved hyperparameters.
        tolerances = {"train_residual": 2e-5} if num_countries >= 100 else {}
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
