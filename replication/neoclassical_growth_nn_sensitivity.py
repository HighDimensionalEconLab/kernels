from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import pandas as pd
from flax import nnx
from nlls_gram import LMStatus

from models.neoclassical_growth_shared import neoclassical_steady_state_sensitivities
from replication.neoclassical_growth_nn import PREFIX, SOLVER, SUMMARY_EXPERIMENTS
from replication.utilities import (
    configure_process,
    nested_results,
    run_solver,
    save_experiment,
    scalar_results,
    write_summary,
)

SENSITIVITY_TIMES = [120.0]


# Implicit-AD sensitivity experiment for the supplement's sensitivity
# appendix: train to T = 120 and differentiate k(T), c(T) through a
# warm-started re-solve of the returned solve handle; the exact steady-state
# derivatives ride along as the comparison target.
def run(
    output_dir: str = ".results",
    n_seeds: int = 7,
    verbose: bool = False,
):
    configure_process()
    print(f"sensitivity: {n_seeds} seeds -> {output_dir}/")
    t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
    rows = []
    frames: dict[str, list[pd.DataFrame]] = {}
    hyperparameters = None
    for seed in range(n_seeds):
        results, extras = run_solver(
            SOLVER,
            {
                "verbose": verbose,
                "k_0": 1.0,
                "data_set": dict(train_T=120.0, train_points=121),
                "seed": seed,
            },
        )
        spec, result = extras["solve"], extras["result"]
        solver, residual_args = spec["solver"], spec["residual_args"]

        # Warm-started at the trained parameters and re-solved to the same
        # residual tolerance, so the stationary point the implicit tangents
        # are taken at moves smoothly with p; the multi-start draw is not
        # re-run. The paths are the unique model solution, so the observable
        # derivatives do not depend on which network parameterization the
        # warm start selects.
        def paths_at_sensitivity_times(p_value):
            solved = solver.solve(result.x, residual_args, p=p_value, **spec["options"])
            policy = nnx.merge(residual_args.graphdef, solved.x, residual_args.nondiff)
            # The model maps t -> [k(t), c(t)] directly.
            return jax.vmap(policy)(t_sens), solved.status

        jacobian, status = jax.jacrev(paths_at_sensitivity_times, has_aux=True)(
            spec["p"]
        )
        jax.block_until_ready(jacobian)
        # A non-converged re-solve would publish exactly zero implicit
        # tangents, indistinguishable from a genuine null effect.
        if int(status) != LMStatus.CONVERGED:
            raise RuntimeError("the sensitivity re-solve did not converge")
        primal_values = jax.vmap(extras["model"])(t_sens)
        fields = {}
        for index, time_value in enumerate(SENSITIVITY_TIMES):
            suffix = f"T{time_value:g}".replace(".", "p")
            fields[f"k_{suffix}"] = float(primal_values[index, 0])
            fields[f"c_{suffix}"] = float(primal_values[index, 1])
            for name in ("a", "delta", "rho_hat", "sigma_crra", "k_0"):
                fields[f"d_k_{suffix}_d_{name}"] = float(
                    getattr(jacobian, name)[index, 0]
                )
                fields[f"d_c_{suffix}_d_{name}"] = float(
                    getattr(jacobian, name)[index, 1]
                )
        # The exact steady-state derivatives the large-T columns converge
        # to; the sigma_crra and k_0 rows converge to zero.
        fields.update(neoclassical_steady_state_sensitivities(spec["p"]))
        rows.append({**scalar_results(results), **fields, "seed": seed})
        if hyperparameters is None:
            hyperparameters = nested_results(results)
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
    write_summary(output_dir, PREFIX, experiments=SUMMARY_EXPERIMENTS)


if __name__ == "__main__":
    jsonargparse.CLI(run)
