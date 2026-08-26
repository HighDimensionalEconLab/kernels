from __future__ import annotations

import statistics

import jax
import jax.numpy as jnp
import jsonargparse
import pandas as pd
from nlls_gram import CG, LMStatus, RidgeLevenbergMarquardt

from models.kernels import integrated_kernel_matrices
from models.multicountry_growth_kernel import path_coordinates
from models.multicountry_growth_preconditioner import build_arrow_woodbury
from models.utilities import CGSettings, Stopwatch
from replication.multicountry_growth_scaling import (
    KERNEL_SET,
    PREFIX,
    SOLVER,
    WARM_TIMING_REPETITIONS,
    rung_lm_overrides,
    rung_parameters,
)
from replication.utilities import (
    configure_process,
    nested_results,
    run_solver,
    save_experiment,
    scalar_results,
)

# The gamma-common tangent is published from N=40, where its value sits in
# one line with the N=2/N=10 sensitivity tables and the production
# acceptance gates hold with margin. The rung configuration is the scaling
# ladder's, so the tangent is taken at exactly the published cg_N40 solve.
GAMMA_COMMON_COUNT = 40
GAMMA_COMMON_TIMES = [10.0]


# d rate(T10)/d gamma along (1, ..., 1) -- the representative-agent CRRA
# comparative static, comparable against the N=2/N=10 sensitivity tables --
# as one implicit-AD forward tangent through a re-solve of the returned
# solve handle, on the ladder's compile-then-warm timing schedule.
# Warm-started at the solved coefficients with the ridge pinned at the level
# the continuation actually reached; both the damped re-solve and the
# undamped implicit-AD system are preconditioned by the arrow-Woodbury
# inverse built at the solution, so no dense factorization enters at any
# width.
def run(output_dir: str = ".results"):
    configure_process()
    print(f"gamma_common (N={GAMMA_COMMON_COUNT}, cg) -> {output_dir}/")
    num_countries = GAMMA_COMMON_COUNT
    overrides = rung_lm_overrides(num_countries)
    results, extras = run_solver(
        SOLVER,
        {
            **rung_parameters(num_countries),
            "kernel_set": KERNEL_SET,
            **overrides,
            "ridge_set": {**overrides.get("ridge_set", {}), "linear_solver": "cg"},
        },
    )
    spec, result = extras["solve"], extras["result"]
    theta = result.x
    cg_settings = CGSettings(**overrides.get("cg_set", {}))
    t_common = jnp.asarray(GAMMA_COMMON_TIMES, dtype=jnp.float64)
    train_data = jnp.asarray(extras["train_df"]["t"].to_numpy())
    K_common, K_tilde_common = integrated_kernel_matrices(
        t_common,
        train_data,
        results["kernel_set"]["kernel"],
        results["kernel_set"]["sigma"],
        results["kernel_set"]["ell"],
    )
    primal_solver = spec["solver"]
    gamma_common_solver = RidgeLevenbergMarquardt(
        spec["residual_fn"],
        has_aux=True,
        metric=primal_solver.initial_metric,
        ridge=float(result.lm_state.ridge),
        linear_solver=CG(
            build_arrow_woodbury(
                theta,
                spec["residual_args"],
                spec["p"],
                float(result.lm_state.ridge),
                cg_settings.damping_floor,
                num_countries,
            ),
            tol=cg_settings.tol,
            maxiter=cg_settings.maxiter,
        ),
        # The undamped AD system runs once, so it gets a deeper budget than
        # the damped inner solves.
        ad_solver=CG(None, maxiter=10 * cg_settings.maxiter),
        init_damping=primal_solver.init_damping,
        damping_decrease=primal_solver.damping_decrease,
        damping_increase=primal_solver.damping_increase,
    )

    def rate_at_common_times(p_value):
        common_result = gamma_common_solver.solve(
            theta, spec["residual_args"], p=p_value, **spec["options"]
        )
        coordinates = path_coordinates(
            common_result.x, K_common, K_tilde_common, p_value
        )
        return coordinates["interest_rate"], common_result.status

    gamma_direction = jax.tree_util.tree_map(jnp.zeros_like, spec["p"]).replace(
        gammas=jnp.ones_like(spec["p"].gammas)
    )
    tangent_times = []
    for _ in range(WARM_TIMING_REPETITIONS + 1):
        tangent_timer = Stopwatch()
        primal_rate, tangent_at_p, status = jax.linearize(
            rate_at_common_times, spec["p"], has_aux=True
        )
        d_rate_common = tangent_at_p(gamma_direction)
        tangent_times.append(tangent_timer.elapsed(d_rate_common))
        if int(status) != LMStatus.CONVERGED:
            raise RuntimeError("the gamma-common sensitivity re-solve did not converge")

    fields = {}
    for index, time_value in enumerate(GAMMA_COMMON_TIMES):
        suffix = f"T{time_value:g}".replace(".", "p")
        fields[f"rate_{suffix}"] = float(primal_rate[index])
        fields[f"d_rate_{suffix}_d_gamma_common"] = float(d_rate_common[index])
    fields["compile_gamma_common_sensitivity_time"] = tangent_times[0]
    fields["gamma_common_sensitivity_time"] = statistics.median(tangent_times[1:])
    save_experiment(
        "gamma_common",
        pd.DataFrame([{**scalar_results(results), **fields}]),
        {
            key: value
            for key, value in extras.items()
            if isinstance(value, pd.DataFrame)
        },
        output_dir,
        dict(
            n_seeds=1,
            residual_dtype="float64",
            warm_timing_repetitions=WARM_TIMING_REPETITIONS,
        ),
        prefix=PREFIX,
        hyperparameters=nested_results(results),
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
