from __future__ import annotations

import json
import pathlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from nlls_gram import LMStatus, RidgeLevenbergMarquardt

from models import multicountry_growth_kernel
from models.kernels import integrated_kernel_matrices
from models.multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    sensitivity_fields_frame,
    sensitivity_observable_labels,
    sensitivity_observables,
    steady_state,
)
from models.multicountry_growth_kernel import path_coordinates
from replication.utilities import (
    configure_process,
    ensemble,
    run_solver,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_kernel"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel
# The three bond configurations share the drawn capital stocks and differ
# only in the initial bond distribution; under the common gamma their
# (r*, k_i, q_i) paths must coincide exactly, so the measured maximum gap
# across configurations is the N-country analogue of the small open
# economy's comparative-statics check.
BOND_CONFIGURATIONS = ("initial_debt", "baseline", "initial_assets")
# Sensitivity dates shared by the two-country and ten-country experiments.
# The cross-check on d r*/d gamma_bar -- the same representative-agent
# perturbation solved through a 1,920-row and a 9,792-row system -- only
# compares at dates both tables report, so the two lists must stay equal.
SENSITIVITY_TIMES = [0.0, 10.0, 30.0, 80.0]
LARGE_COUNTRY_COUNT = 10


def bond_invariance(validation_frames, num_countries, test_T):
    # Reported on two horizons: the test window, where the solution is
    # benchmark-certified, and the full training horizon, whose sparse tail
    # collocation lets the configurations drift apart by orders of
    # magnitude more.
    reference = validation_frames[BOND_CONFIGURATIONS[0]]
    columns = ["interest_rate"] + [
        f"{quantity}_{country}"
        for quantity in ("k", "q")
        for country in range(1, num_countries + 1)
    ]

    def gaps(mask):
        out = {"interest_rate": 0.0, "k": 0.0, "q": 0.0}
        for name in BOND_CONFIGURATIONS[1:]:
            frame = validation_frames[name]
            if not np.array_equal(frame["t"].to_numpy(), reference["t"].to_numpy()):
                raise ValueError("bond configurations must share the validation grid")
            for column in columns:
                quantity = column.split("_")[0] if column != "interest_rate" else column
                out[quantity] = max(
                    out[quantity],
                    float(
                        np.max(np.abs(frame[column][mask] - reference[column][mask]))
                    ),
                )
        return out

    test_gaps = gaps(reference["t"] <= test_T)
    full_gaps = gaps(reference["t"] == reference["t"])
    return {
        "configurations": list(BOND_CONFIGURATIONS),
        "test_T": test_T,
        "max_interest_rate_gap": test_gaps["interest_rate"],
        "max_k_gap": test_gaps["k"],
        "max_q_gap": test_gaps["q"],
        "max_gap": max(test_gaps.values()),
        "max_interest_rate_gap_train_horizon": full_gaps["interest_rate"],
        "max_k_gap_train_horizon": full_gaps["k"],
        "max_q_gap_train_horizon": full_gaps["q"],
        "max_gap_train_horizon": max(full_gaps.values()),
    }


# One extra solve with the experiment's own options, then the implicit-AD
# pass through a re-solve of the returned SolveSpec. Warm-started at the
# solved coefficients with the ridge pinned at the level the continuation
# actually reached, so the re-solve lands on the primal's own stationary
# point in a step or two. Re-annealing from ridge_init would be free to
# settle at a different achieved ridge, and the implicit tangents are the
# tangents of whatever stationary point the solve returns.
def sensitivity_fields_for(options, mode):
    results, extras = run_solver(SOLVER, options)
    spec, result = extras["solve"], extras["result"]
    theta = result.x
    num_countries = int(results["num_countries"])
    t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
    train_data = jnp.asarray(extras["train_df"]["t"].to_numpy())
    K_sens, K_tilde_sens = integrated_kernel_matrices(
        t_sens, train_data, results["kernel"], results["sigma"], results["ell"]
    )
    sensitivity_solver = RidgeLevenbergMarquardt(
        spec.solver.residual_fn,
        metric=spec.solver.initial_metric,
        ridge=float(result.lm_state.ridge),
        linear_solver=spec.solver.linear_solver,
        init_damping=spec.solver.init_damping,
        damping_decrease=spec.solver.damping_decrease,
        damping_increase=spec.solver.damping_increase,
    )

    def observables_at_sensitivity_times(p_value):
        solved = sensitivity_solver.solve(
            theta, spec.residual_args, p=p_value, **spec.options
        )
        coordinates = path_coordinates(solved.x, K_sens, K_tilde_sens, p_value)
        return (
            sensitivity_observables(coordinates, p_value["parameters"]),
            solved.status,
        )

    if mode == "reverse":
        # The full Jacobian: every observable against every parameter, at
        # one reverse pass per output scalar. Any question about the
        # solution can be read off it afterwards.
        jacobian, status = jax.jacrev(observables_at_sensitivity_times, has_aux=True)(
            spec.p
        )
        jax.block_until_ready(jacobian)
        primal_observables = np.asarray(
            sensitivity_observables(
                path_coordinates(theta, K_sens, K_tilde_sens, spec.p),
                spec.p["parameters"],
            )
        )
        gradient_parameters = jacobian["parameters"]
        scalar_blocks = {
            "alpha": np.asarray(gradient_parameters.alpha),
            "delta": np.asarray(gradient_parameters.delta),
            "r": np.asarray(gradient_parameters.r),
            "eta": np.asarray(gradient_parameters.eta),
        }
        country_blocks = {
            "gamma": np.asarray(gradient_parameters.gammas),
            "productivity": np.asarray(gradient_parameters.productivities),
            "population": np.asarray(gradient_parameters.populations),
            "k_0": np.asarray(jacobian["k_0"]),
            "a_0": np.asarray(jacobian["a_0"]),
        }
        rate_a_0_gradient = country_blocks["a_0"][:, 0, :]
    else:
        # One linearization at the solved point, then a tangent map applied
        # to each reported direction: (1, ..., 1) in gamma and one bond
        # direction per country. The number of applications follows the
        # directions published rather than the 2N + 4 observables, and only
        # those directions exist afterwards -- there is no Jacobian left
        # over to answer a different question from. Wall time tracks
        # reverse mode either way: the re-solve inside the differentiated
        # function dominates both, so switching modes buys accuracy of
        # intent, not speed.
        primal, tangent_at_p, status = jax.linearize(
            observables_at_sensitivity_times, spec.p, has_aux=True
        )
        jax.block_until_ready(primal)
        primal_observables = np.asarray(primal)
        zero_tangent = jax.tree_util.tree_map(jnp.zeros_like, spec.p)
        # (1, ..., 1) in gamma: the country count cancels, so this is the
        # representative-agent risk-aversion change at any N.
        gamma_common = np.asarray(
            tangent_at_p(
                {
                    **zero_tangent,
                    "parameters": replace(
                        zero_tangent["parameters"],
                        gammas=jnp.ones_like(spec.p["parameters"].gammas),
                    ),
                }
            )
        )
        rate_a_0_gradient = np.column_stack(
            [
                np.asarray(
                    tangent_at_p(
                        {
                            **zero_tangent,
                            "a_0": zero_tangent["a_0"].at[country].set(1.0),
                        }
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
        [float(value) for value in spec.p["parameters"].populations],
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
    return fields


def run(output_dir: str | None = None, use_float64: bool = True):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = MulticountryParameters()
    drawn_k_0, _ = draw_initial_conditions(InitialConditionSettings(), parameters)
    output_scale = float(steady_state(parameters)["output_scale"])
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(ridge=1e-7, ridge_init=1e-5, gtol=1e-6)
    tol = (
        {}
        if use_float64
        else dict(
            train_residual=1e-3,
            validation_residual=1e-2,
            benchmark_match=1e-2,
        )
    )
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")
    validation_frames = {}

    def go(name, sensitivity_mode=None, **experiment):
        print(f"{name} -> {output_dir}/")
        options = {
            "use_float64": use_float64,
            "lm_set": lm,
            "tolerance_set": tol,
            **experiment,
        }
        results_df, frames, hyperparameters = ensemble(
            SOLVER, n_seeds=1, options=options
        )
        if sensitivity_mode is not None:
            results_df = results_df.assign(
                **sensitivity_fields_for(options, sensitivity_mode)
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
        validation_frames[name] = frames["validation_df"]

    # The baseline and the ten-country experiment publish the implicit-AD
    # sensitivity tables for the supplement's sensitivity appendix (float64
    # only): every observable at every date in SENSITIVITY_TIMES against
    # every structural parameter and initial condition. Two countries carry
    # the per-country levels; ten carry the world aggregates, which is all a
    # cross-section that wide can say.
    go("baseline", sensitivity_mode="reverse" if use_float64 else None)
    go(
        "initial_debt",
        k_0=drawn_k_0,
        a_0=(-0.5 * output_scale, 0.5 * output_scale),
    )
    go(
        "initial_assets",
        k_0=drawn_k_0,
        a_0=(0.5 * output_scale, -0.5 * output_scale),
    )
    go(
        "three_countries",
        parameters=dict(
            gammas=(2.0, 2.0, 2.0),
            populations=(1.0, 1.0, 1.0),
            productivities=(1.0, 1.0, 1.0),
        ),
    )
    go(
        "heterogeneous_countries",
        parameters=dict(
            populations=(1.0, 2.0),
            productivities=(1.0, 1.1),
        ),
    )
    # Symmetric in preferences, size, and technology: the only heterogeneity
    # is the drawn capital and the drawn bond positions, so the aggregate
    # comparative statics isolate what a wide cross-section adds over two
    # countries rather than what asymmetric calibration adds.
    # Forward mode here: this table reports one direction in gamma plus the
    # per-country bond directions the neutrality check reads, so the tangent
    # map is applied to exactly what gets published and nothing else.
    go(
        "ten_countries",
        sensitivity_mode="forward" if use_float64 else None,
        parameters=dict(
            gammas=(2.0,) * LARGE_COUNTRY_COUNT,
            populations=(1.0,) * LARGE_COUNTRY_COUNT,
            productivities=(1.0,) * LARGE_COUNTRY_COUNT,
        ),
    )
    write_summary(
        output_dir,
        PREFIX,
        experiments={
            "baseline",
            "initial_debt",
            "initial_assets",
            "three_countries",
            "heterogeneous_countries",
            "ten_countries",
        },
    )
    invariance = bond_invariance(
        validation_frames,
        parameters.num_countries,
        multicountry_growth_kernel.DataSettings().test_T,
    )
    summary_path = pathlib.Path(output_dir) / f"{PREFIX}_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["bond_invariance"] = invariance
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        "bond invariance max gap (r*, k, q): "
        f"{invariance['max_interest_rate_gap']:.2e}, "
        f"{invariance['max_k_gap']:.2e}, {invariance['max_q_gap']:.2e}"
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
