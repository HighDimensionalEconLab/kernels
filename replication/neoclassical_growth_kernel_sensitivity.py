from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import pandas as pd
from nlls_gram import LMStatus

from models.kernels import (
    IntegratedPath,
    integrated_kernel_matrices,
    integrated_kernel_paths,
)
from models.neoclassical_growth_shared import neoclassical_steady_state_sensitivities
from replication.neoclassical_growth_kernel import PREFIX, SOLVER
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
# appendix: train to T = 120 (the transition remnant is ~e^{-|lambda|T}
# ~ 3e-6) and differentiate k(T), c(T) through a cold re-solve of the
# returned solve handle; the exact steady-state derivatives ride along as
# the comparison target.
def run(
    output_dir: str = ".results",
):
    configure_process()
    print(f"sensitivity -> {output_dir}/")
    results, extras = run_solver(
        SOLVER,
        {"data_set": dict(train_T=120.0, train_points=121)},
    )
    spec = extras["solve"]
    solver = spec["solver"]
    t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
    train_data = jnp.asarray(extras["train_df"]["t"].to_numpy())
    K_sens, K_tilde_sens = integrated_kernel_matrices(
        t_sens,
        train_data,
        results["kernel_set"]["kernel"],
        results["kernel_set"]["sigma"],
        results["kernel_set"]["ell"],
    )

    # One reverse-mode pass through the re-solve yields d[k(t), c(t)]/dp
    # at every sensitivity time together: the implicit dependence of the
    # solved coefficients on p plus the direct dependence of the path
    # evaluation.
    def paths_at_sensitivity_times(p_value):
        solved = solver.solve(
            spec["theta_0"], spec["residual_args"], p=p_value, **spec["options"]
        )
        paths = integrated_kernel_paths(
            {
                "c": IntegratedPath(solved.x.alpha_c, solved.x.c_0),
                "k": IntegratedPath(solved.x.alpha_k, p_value.k_0),
            },
            K_sens,
            K_tilde_sens,
        )
        return jnp.column_stack([paths["k"].value, paths["c"].value]), solved.status

    jacobian, status = jax.jacrev(paths_at_sensitivity_times, has_aux=True)(spec["p"])
    jax.block_until_ready(jacobian)
    # A non-converged re-solve would publish exactly zero implicit
    # tangents, indistinguishable from a genuine null effect.
    if int(status) != LMStatus.CONVERGED:
        raise RuntimeError("the sensitivity re-solve did not converge")
    theta = extras["result"].x
    paths = integrated_kernel_paths(
        {
            "c": IntegratedPath(theta.alpha_c, theta.c_0),
            "k": IntegratedPath(theta.alpha_k, spec["p"].k_0),
        },
        K_sens,
        K_tilde_sens,
    )
    k_sens, c_sens = paths["k"].value, paths["c"].value
    fields = {}
    for index, time_value in enumerate(SENSITIVITY_TIMES):
        suffix = f"T{time_value:g}".replace(".", "p")
        fields[f"k_{suffix}"] = float(k_sens[index])
        fields[f"c_{suffix}"] = float(c_sens[index])
        for name in ("a", "delta", "rho_hat", "sigma_crra", "k_0"):
            fields[f"d_k_{suffix}_d_{name}"] = float(getattr(jacobian, name)[index, 0])
            fields[f"d_c_{suffix}_d_{name}"] = float(getattr(jacobian, name)[index, 1])
    # The exact steady-state derivatives the large-T columns converge
    # to; the sigma_crra and k_0 rows converge to zero.
    fields.update(neoclassical_steady_state_sensitivities(spec["p"]))
    save_experiment(
        "sensitivity",
        pd.DataFrame([{**scalar_results(results), **fields}]),
        {
            key: value
            for key, value in extras.items()
            if isinstance(value, pd.DataFrame)
        },
        output_dir,
        dict(n_seeds=1, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=nested_results(results),
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
