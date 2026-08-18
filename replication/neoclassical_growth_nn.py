from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import pandas as pd
from flax import nnx
from nlls_gram import LMStatus

from models import neoclassical_growth_nn
from models.neoclassical_growth_benchmark import neoclassical_steady_state_sensitivities
from models.neoclassical_growth_nn import policy_values
from replication.utilities import (
    configure_process,
    ensemble,
    run_solver,
    save_experiment,
    write_summary,
)

PREFIX = "neoclassical_growth_nn"

SOLVER = neoclassical_growth_nn.neoclassical_growth_nn

SENSITIVITY_TIMES = [120.0]


def run(
    output_dir: str | None = None,
    # Multi-start retries inside each solve keep acceptance high, so a small
    # seed ensemble suffices for the cross-seed medians.
    n_seeds: int = 7,
    use_float64: bool = True,
    verbose: bool = False,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    stop = {} if use_float64 else dict(train_loss_threshold=1e-9)
    print(f"baseline: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds,
        options={
            "use_float64": use_float64,
            "verbose": verbose,
            "k_0": 1.0,
            "stop_set": stop,
        },
    )
    save_experiment(
        "baseline",
        results_df,
        frames,
        output_dir,
        {
            "n_seeds": n_seeds,
            "residual_dtype": "float64" if use_float64 else "float32",
        },
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )

    # Implicit-AD sensitivity experiment for the supplement's sensitivity
    # appendix: train to T = 120 and differentiate k(T), c(T) through a
    # warm-started re-solve of the returned SolveSpec; the exact steady-state
    # derivatives ride along as the comparison target. Float64 only: the
    # float32 noise floor swamps that comparison.
    if use_float64:
        print(f"sensitivity: {n_seeds} seeds -> {output_dir}/")
        t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
        rows = []
        frames = {}
        hyperparameters = None
        for seed in range(n_seeds):
            results, extras = run_solver(
                SOLVER,
                {
                    "use_float64": use_float64,
                    "verbose": verbose,
                    "k_0": 1.0,
                    "stop_set": stop,
                    "data_set": dict(train_T=120.0, train_points=121),
                    "seed": seed,
                },
            )
            spec, result = extras["solve"], extras["result"]

            # Warm-started at the trained parameters and re-solved to the
            # same residual tolerance, so the stationary point the implicit
            # tangents are taken at moves smoothly with p; the multi-start
            # draw is not re-run. The paths are the unique model solution, so
            # the observable derivatives do not depend on which network
            # parameterization the warm start selects.
            def paths_at_sensitivity_times(p_value):
                solved = spec.solver.solve(
                    result.x, spec.residual_args, p=p_value, **spec.options
                )
                policy = nnx.merge(
                    spec.residual_args.graphdef, solved.x, spec.residual_args.nondiff
                )
                values = policy_values(policy, t_sens, p_value)
                return jnp.column_stack([values["k"], values["c"]]), solved.status

            jacobian, status = jax.jacrev(paths_at_sensitivity_times, has_aux=True)(
                spec.p
            )
            jax.block_until_ready(jacobian)
            # A non-converged re-solve would publish exactly zero implicit
            # tangents, indistinguishable from a genuine null effect.
            if int(status) != LMStatus.CONVERGED:
                raise RuntimeError("the sensitivity re-solve did not converge")
            primal_values = policy_values(extras["model"], t_sens, spec.p)
            fields = {}
            for index, time_value in enumerate(SENSITIVITY_TIMES):
                suffix = f"T{time_value:g}".replace(".", "p")
                fields[f"k_{suffix}"] = float(primal_values["k"][index])
                fields[f"c_{suffix}"] = float(primal_values["c"][index])
                for name in ("a", "delta", "rho_hat", "sigma_crra", "k_0"):
                    fields[f"d_k_{suffix}_d_{name}"] = float(
                        getattr(jacobian, name)[index, 0]
                    )
                    fields[f"d_c_{suffix}_d_{name}"] = float(
                        getattr(jacobian, name)[index, 1]
                    )
            # The exact steady-state derivatives the large-T columns converge
            # to; the sigma_crra and k_0 rows converge to zero.
            fields.update(
                neoclassical_steady_state_sensitivities(
                    results["a"], results["delta"], results["rho_hat"]
                )
            )
            rows.append({**results, **fields, "seed": seed})
            if hyperparameters is None:
                hyperparameters = extras["hyperparameters"]
            for key, value in extras.items():
                if isinstance(value, pd.DataFrame):
                    frames.setdefault(key, []).append(value.assign(seed=seed))
        save_experiment(
            "sensitivity",
            pd.DataFrame(rows),
            {key: pd.concat(dfs, ignore_index=True) for key, dfs in frames.items()},
            output_dir,
            {"n_seeds": n_seeds, "residual_dtype": "float64"},
            prefix=PREFIX,
            hyperparameters=hyperparameters,
        )

    # Explicit set: the prefix is a strict prefix of
    # neoclassical_growth_nn_recursive, whose JSONs share the glob.
    write_summary(output_dir, PREFIX, experiments={"baseline", "sensitivity"})


if __name__ == "__main__":
    jsonargparse.CLI(run)
