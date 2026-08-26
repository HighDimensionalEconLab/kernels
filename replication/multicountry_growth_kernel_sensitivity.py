from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import LMStatus, RidgeLevenbergMarquardt

from models.kernels import integrated_kernel_matrices
from models.multicountry_growth_kernel import path_coordinates
from models.multicountry_growth_shared import (
    sensitivity_fields_frame,
    sensitivity_observable_labels,
    sensitivity_observables,
)
from replication.multicountry_growth_kernel import (
    LARGE_COUNTRY_PARAMETERS,
    PREFIX,
    SENSITIVITY_TIMES,
    SOLVER,
)
from replication.utilities import (
    configure_process,
    nested_results,
    run_solver,
    save_experiment,
    scalar_results,
)


# One solve with the experiment's options, then the implicit-AD pass through
# a re-solve of the returned solve handle. Warm-started at the solved
# coefficients with the ridge pinned at the level the continuation actually
# reached, so the re-solve lands on the primal's own stationary point in a
# step or two. Re-annealing from ridge_init would be free to settle at a
# different achieved ridge, and the implicit tangents are the tangents of
# whatever stationary point the solve returns.
def sensitivity_row(options, mode):
    results, extras = run_solver(SOLVER, options)
    spec, result = extras["solve"], extras["result"]
    theta = result.x
    num_countries = len(results["parameters"]["populations"])
    t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
    train_data = jnp.asarray(extras["train_df"]["t"].to_numpy())
    K_sens, K_tilde_sens = integrated_kernel_matrices(
        t_sens,
        train_data,
        results["kernel_set"]["kernel"],
        results["kernel_set"]["sigma"],
        results["kernel_set"]["ell"],
    )
    primal_solver = spec["solver"]
    sensitivity_solver = RidgeLevenbergMarquardt(
        spec["residual_fn"],
        has_aux=True,
        metric=primal_solver.initial_metric,
        ridge=float(result.lm_state.ridge),
        linear_solver=primal_solver.linear_solver,
        init_damping=primal_solver.init_damping,
        damping_decrease=primal_solver.damping_decrease,
        damping_increase=primal_solver.damping_increase,
    )

    def observables_at_sensitivity_times(p_value):
        solved = sensitivity_solver.solve(
            theta, spec["residual_args"], p=p_value, **spec["options"]
        )
        coordinates = path_coordinates(solved.x, K_sens, K_tilde_sens, p_value)
        return (
            sensitivity_observables(coordinates, p_value),
            solved.status,
        )

    if mode == "reverse":
        # The full Jacobian: every observable against every parameter, at
        # one reverse pass per output scalar. Any question about the
        # solution can be read off it afterwards.
        jacobian, status = jax.jacrev(observables_at_sensitivity_times, has_aux=True)(
            spec["p"]
        )
        jax.block_until_ready(jacobian)
        primal_observables = np.asarray(
            sensitivity_observables(
                path_coordinates(theta, K_sens, K_tilde_sens, spec["p"]),
                spec["p"],
            )
        )
        scalar_blocks = {
            "alpha": np.asarray(jacobian.alpha),
            "delta": np.asarray(jacobian.delta),
            "r": np.asarray(jacobian.r),
            "eta": np.asarray(jacobian.eta),
        }
        country_blocks = {
            "gamma": np.asarray(jacobian.gammas),
            "productivity": np.asarray(jacobian.productivities),
            "population": np.asarray(jacobian.populations),
            "k_0": np.asarray(jacobian.k_0),
            "a_0": np.asarray(jacobian.a_0),
        }
        rate_a_0_gradient = country_blocks["a_0"][:, 0, :]
    else:
        # One linearization at the solved point, then a tangent map applied
        # to each reported direction: (1, ..., 1) in gamma and one bond
        # direction per country.
        primal, tangent_at_p, status = jax.linearize(
            observables_at_sensitivity_times, spec["p"], has_aux=True
        )
        jax.block_until_ready(primal)
        primal_observables = np.asarray(primal)
        zero_tangent = jax.tree_util.tree_map(jnp.zeros_like, spec["p"])
        # (1, ..., 1) in gamma: the country count cancels, so this is the
        # representative-agent risk-aversion change at any N.
        gamma_common = np.asarray(
            tangent_at_p(zero_tangent.replace(gammas=jnp.ones_like(spec["p"].gammas)))
        )
        rate_a_0_gradient = np.column_stack(
            [
                np.asarray(
                    tangent_at_p(
                        zero_tangent.replace(a_0=zero_tangent.a_0.at[country].set(1.0))
                    )
                )[:, 0]
                for country in range(num_countries)
            ]
        )
        scalar_blocks = {"gamma_common": gamma_common}
        country_blocks = {}

    # A non-converged re-solve would publish exactly zero implicit
    # tangents, indistinguishable from a genuine null effect.
    if int(status) != LMStatus.CONVERGED:
        raise RuntimeError("the sensitivity re-solve did not converge")
    fields, _ = sensitivity_fields_frame(
        SENSITIVITY_TIMES,
        sensitivity_observable_labels(num_countries),
        primal_observables,
        scalar_blocks,
        country_blocks,
        [float(value) for value in spec["p"].populations],
    )

    # Under identical homothetic preferences the world rate is invariant
    # to every clearing-preserving reallocation of the initial bond
    # positions, so the rate's bond derivatives are zero to solver
    # tolerance. Reported as a measured scalar: it is the check on the
    # implicit tangents themselves, not another comparative static.
    for index, time_value in enumerate(SENSITIVITY_TIMES):
        suffix = f"T{time_value:g}".replace(".", "p")
        fields[f"rate_{suffix}_a_0_neutrality"] = float(
            np.max(np.abs(rate_a_0_gradient[index]))
        )
    fields["rate_a_0_neutrality"] = float(np.max(np.abs(rate_a_0_gradient)))
    return {**scalar_results(results), **fields}, nested_results(results), extras


# Implicit-AD sensitivity tables for the supplement's sensitivity appendix:
# every observable at every date in SENSITIVITY_TIMES against every
# structural parameter and initial condition. Two countries carry the
# per-country levels (reverse mode); ten carry the world aggregates, which
# is all a cross-section that wide can say (forward mode along the published
# directions).
def run(output_dir: str = ".results"):
    configure_process()

    def go(name, mode, **experiment):
        print(f"{name} -> {output_dir}/")
        row, hyperparameters, extras = sensitivity_row(experiment, mode)
        save_experiment(
            name,
            pd.DataFrame([row]),
            {
                key: value
                for key, value in extras.items()
                if isinstance(value, pd.DataFrame)
            },
            output_dir,
            dict(n_seeds=1, residual_dtype="float64"),
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    go("sensitivity", "reverse")
    go("sensitivity_ten_countries", "forward", **LARGE_COUNTRY_PARAMETERS)


if __name__ == "__main__":
    jsonargparse.CLI(run)
