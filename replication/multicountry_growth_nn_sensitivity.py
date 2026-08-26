from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from nlls_gram import LMStatus

from models import multicountry_growth_nn
from models.multicountry_growth_shared import (
    sensitivity_fields_frame,
    sensitivity_observable_labels,
    sensitivity_observables,
)
from replication.multicountry_growth_kernel import (
    LARGE_COUNTRY_PARAMETERS,
    SENSITIVITY_TIMES,
)
from replication.multicountry_growth_nn import PREFIX
from replication.utilities import (
    configure_process,
    nested_results,
    run_solver,
    save_experiment,
    scalar_results,
)

SOLVER = multicountry_growth_nn.multicountry_growth_nn


# Implicit-AD sensitivity experiments on their own seed budget: the tables
# report one solve, so training the full ensemble through the reverse pass
# buys nothing. The dates are the kernel driver's, since the two solvers'
# tables are compared cell by cell.
def run(
    output_dir: str = ".results",
    sensitivity_seeds: int = 1,
    verbose: bool = False,
):
    configure_process()
    t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)

    def go(name, scenario, **experiment):
        print(f"{name}: {sensitivity_seeds} seeds -> {output_dir}/")
        rows = []
        frames: dict[str, list[pd.DataFrame]] = {}
        hyperparameters = None
        for seed in range(sensitivity_seeds):
            results, extras = run_solver(
                SOLVER,
                {
                    "verbose": verbose,
                    "seed": seed,
                    **experiment,
                },
            )
            spec, result = extras["solve"], extras["result"]
            solver, residual_args = spec["solver"], spec["residual_args"]
            num_countries = len(results["parameters"]["populations"])

            # Warm-started at the trained parameters and re-solved to the
            # same residual tolerance, so the stationary point the implicit
            # tangents are taken at moves smoothly with p; the multi-start
            # draw is not re-run. The paths are the unique model solution,
            # so the observable derivatives do not depend on which network
            # parameterization the warm start selects.
            def observables_at_sensitivity_times(p_value):
                solved = solver.solve(
                    result.x, residual_args, p=p_value, **spec["options"]
                )
                policy = nnx.merge(
                    residual_args.graphdef, solved.x, residual_args.nondiff
                )
                values = jax.vmap(policy)(t_sens)
                return sensitivity_observables(values, p_value), solved.status

            # The full Jacobian: every observable against every parameter,
            # at one reverse pass per output scalar.
            jacobian, status = jax.jacrev(
                observables_at_sensitivity_times, has_aux=True
            )(spec["p"])
            jax.block_until_ready(jacobian)
            # A non-converged re-solve would publish exactly zero implicit
            # tangents, indistinguishable from a genuine null effect.
            if int(status) != LMStatus.CONVERGED:
                raise RuntimeError("the sensitivity re-solve did not converge")
            primal_observables = np.asarray(
                sensitivity_observables(
                    jax.vmap(extras["model"])(t_sens),
                    spec["p"],
                )
            )
            gradient_parameters = jacobian
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
                [float(value) for value in spec["p"].populations],
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
            rows.append({**scalar_results(results), **fields, "seed": seed})
            if hyperparameters is None:
                hyperparameters = nested_results(results)
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

    go("sensitivity", "two_country_drawn_initial_conditions")
    go(
        "sensitivity_ten_countries",
        "ten_country_drawn_initial_conditions",
        **LARGE_COUNTRY_PARAMETERS,
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
