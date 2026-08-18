from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from nlls_gram import LMStatus

from models import multicountry_growth_nn
from models.multicountry_growth import (
    sensitivity_fields_frame,
    sensitivity_observable_labels,
    sensitivity_observables,
)
from models.multicountry_growth_nn import path_values
from replication.multicountry_growth_kernel import (
    LARGE_COUNTRY_COUNT,
    SENSITIVITY_TIMES,
)
from replication.utilities import (
    configure_process,
    ensemble,
    run_solver,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_nn"
SOLVER = multicountry_growth_nn.multicountry_growth_nn
LARGE_COUNTRY_PARAMETERS = dict(
    gammas=(2.0,) * LARGE_COUNTRY_COUNT,
    populations=(1.0,) * LARGE_COUNTRY_COUNT,
    productivities=(1.0,) * LARGE_COUNTRY_COUNT,
)


def run(
    output_dir: str | None = None,
    n_seeds: int = 7,
    sensitivity_seeds: int = 1,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-8)
    tol = (
        {}
        if use_float64
        else dict(
            train_residual=5e-3,
            validation_residual=1e-2,
            benchmark_match=1e-2,
        )
    )

    def go(name, scenario, seeds=None, **options):
        seeds = n_seeds if seeds is None else seeds
        print(f"{name}: {seeds} seeds -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=seeds,
            options={
                "use_float64": use_float64,
                "verbose": verbose,
                "stop_set": stop,
                "tolerance_set": tol,
                **options,
            },
        )
        save_experiment(
            name,
            results_df,
            frames,
            output_dir,
            {
                "n_seeds": seeds,
                "scenario": scenario,
                "residual_dtype": "float64" if use_float64 else "float32",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    def go_sensitivity(name, scenario, **options):
        print(f"{name}: {sensitivity_seeds} seeds -> {output_dir}/")
        t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
        rows = []
        frames: dict[str, list[pd.DataFrame]] = {}
        hyperparameters = None
        for seed in range(sensitivity_seeds):
            results, extras = run_solver(
                SOLVER,
                {
                    "use_float64": use_float64,
                    "verbose": verbose,
                    "stop_set": stop,
                    "tolerance_set": tol,
                    "seed": seed,
                    **options,
                },
            )
            spec, result = extras["solve"], extras["result"]
            num_countries = int(results["num_countries"])
            scales = spec.residual_args.scales

            # Warm-started at the trained parameters and re-solved to the
            # same residual tolerance, so the stationary point the implicit
            # tangents are taken at moves smoothly with p; the multi-start
            # draw is not re-run. The paths are the unique model solution,
            # so the observable derivatives do not depend on which network
            # parameterization the warm start selects.
            def observables_at_sensitivity_times(p_value):
                solved = spec.solver.solve(
                    result.x, spec.residual_args, p=p_value, **spec.options
                )
                policy = nnx.merge(
                    spec.residual_args.graphdef, solved.x, spec.residual_args.nondiff
                )
                values = path_values(policy, t_sens, p_value, scales)
                return sensitivity_observables(values, p_value.model), solved.status

            # The full Jacobian: every observable against every parameter,
            # at one reverse pass per output scalar.
            jacobian, status = jax.jacrev(
                observables_at_sensitivity_times, has_aux=True
            )(spec.p)
            jax.block_until_ready(jacobian)
            # A non-converged re-solve would publish exactly zero implicit
            # tangents, indistinguishable from a genuine null effect.
            if int(status) != LMStatus.CONVERGED:
                raise RuntimeError("the sensitivity re-solve did not converge")
            primal_observables = np.asarray(
                sensitivity_observables(
                    path_values(extras["model"], t_sens, spec.p, scales),
                    spec.p.model,
                )
            )
            gradient_parameters = jacobian.model
            country_blocks = {
                "gamma": np.asarray(gradient_parameters.gammas),
                "productivity": np.asarray(gradient_parameters.productivities),
                "population": np.asarray(gradient_parameters.populations),
                "k_0": np.asarray(jacobian.k_0),
                "a_0": np.asarray(jacobian.a_0),
            }
            rate_a_0_gradient = country_blocks["a_0"][:, 0, :]
            fields, _ = sensitivity_fields_frame(
                SENSITIVITY_TIMES,
                sensitivity_observable_labels(num_countries),
                primal_observables,
                {
                    "alpha": np.asarray(gradient_parameters.alpha),
                    "delta": np.asarray(gradient_parameters.delta),
                    "r": np.asarray(gradient_parameters.r),
                    "eta": np.asarray(gradient_parameters.eta),
                },
                country_blocks,
                [float(value) for value in spec.p.model.populations],
            )

            # Under identical homothetic preferences the world rate is
            # invariant to every clearing-preserving reallocation of the
            # initial bond positions, so the rate's bond derivatives are
            # zero to solver tolerance. Reported as a measured scalar: it is
            # the check on the implicit tangents themselves, not another
            # comparative static.
            for index, time_value in enumerate(SENSITIVITY_TIMES):
                suffix = f"T{time_value:g}".replace(".", "p")
                fields[f"rate_{suffix}_a_0_neutrality"] = float(
                    np.max(np.abs(rate_a_0_gradient[index]))
                )
            fields["rate_a_0_neutrality"] = float(np.max(np.abs(rate_a_0_gradient)))
            rows.append({**results, **fields, "seed": seed})
            if hyperparameters is None:
                hyperparameters = extras["hyperparameters"]
            for key, value in extras.items():
                if isinstance(value, pd.DataFrame):
                    frames.setdefault(key, []).append(value.assign(seed=seed))
        save_experiment(
            name,
            pd.DataFrame(rows),
            {key: pd.concat(dfs, ignore_index=True) for key, dfs in frames.items()},
            output_dir,
            {
                "n_seeds": sensitivity_seeds,
                "scenario": scenario,
                "residual_dtype": "float64",
            },
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("baseline", "two_country_drawn_initial_conditions")
    go(
        "large_countries",
        "ten_country_drawn_initial_conditions",
        parameters=LARGE_COUNTRY_PARAMETERS,
    )
    experiments = {"baseline", "large_countries"}
    # The implicit-AD experiments are separate runs on their own seed budget:
    # the tables report one solve, so training the full ensemble through the
    # reverse pass buys nothing. Float64 only -- the float32 noise floor
    # swamps the derivatives. The dates are the kernel driver's, since the
    # two solvers' tables are compared cell by cell.
    if use_float64:
        go_sensitivity("sensitivity", "two_country_drawn_initial_conditions")
        go_sensitivity(
            "sensitivity_ten_countries",
            "ten_country_drawn_initial_conditions",
            parameters=LARGE_COUNTRY_PARAMETERS,
        )
        experiments |= {"sensitivity", "sensitivity_ten_countries"}
    write_summary(output_dir, PREFIX, experiments=experiments)


if __name__ == "__main__":
    jsonargparse.CLI(run)
