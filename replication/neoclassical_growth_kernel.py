from __future__ import annotations

import jax
import jax.numpy as jnp
import jsonargparse
import pandas as pd
from nlls_gram import LMStatus

from models import neoclassical_growth_kernel
from models.kernels import integrated_kernel_matrices
from models.neoclassical_growth_benchmark import neoclassical_steady_state_sensitivities
from models.neoclassical_growth_kernel import path_values
from replication.utilities import (
    configure_process,
    ensemble,
    run_solver,
    save_experiment,
    write_summary,
)

PREFIX = "neoclassical_growth_kernel"

SENSITIVITY_TIMES = [120.0]


def run(
    output_dir: str | None = None,
    use_float64: bool = True,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(residual_atol=5e-5, ridge=1e-7, gtol=1e-5)
    tol = (
        {}
        if use_float64
        else dict(train_residual=5e-3, validation_residual=1e-2, mpk_bound=5e-3)
    )
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            neoclassical_growth_kernel.neoclassical_growth_kernel,
            n_seeds=1,
            options={
                "use_float64": use_float64,
                "lm_set": lm,
                "tolerance_set": tol,
                **experiment,
            },
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

    # QE figure experiments. The sparse grid keeps the baseline's front-loaded
    # relative spacing over the full train_T; far_steady_state trains on the
    # first ~2 e-foldings of the saddle path only.
    go("baseline")
    go(
        "sparse",
        data_set=dict(
            train_points_list=[
                0.0,
                2.5,
                7.5,
                12.5,
                25.0,
                37.5,
                50.0,
                62.5,
                75.0,
                87.5,
                95.0,
                100.0,
            ]
        ),
    )
    go(
        "far_steady_state",
        data_set=dict(train_T=20.0, train_points=21, test_T=30.0),
    )

    # Implicit-AD sensitivity experiment for the supplement's sensitivity
    # appendix: train to T = 120 (the transition remnant is ~e^{-|lambda|T}
    # ~ 3e-6) and differentiate k(T), c(T) through a cold re-solve of the
    # returned SolveSpec; the exact steady-state derivatives ride along as
    # the comparison target. Float64 only: the float32 noise floor swamps
    # that comparison.
    if use_float64:
        print(f"sensitivity -> {output_dir}/")
        results, extras = run_solver(
            neoclassical_growth_kernel.neoclassical_growth_kernel,
            {
                "use_float64": use_float64,
                "lm_set": lm,
                "tolerance_set": tol,
                "data_set": dict(train_T=120.0, train_points=121),
            },
        )
        spec = extras["solve"]
        t_sens = jnp.asarray(SENSITIVITY_TIMES, dtype=jnp.float64)
        train_data = jnp.asarray(extras["train_df"]["t"].to_numpy())
        K_sens, K_tilde_sens = integrated_kernel_matrices(
            t_sens, train_data, results["kernel"], results["sigma"], results["ell"]
        )

        # One reverse-mode pass through the re-solve yields d[k(t), c(t)]/dp
        # at every sensitivity time together: the implicit dependence of the
        # solved coefficients on p plus the direct dependence of the path
        # evaluation.
        def paths_at_sensitivity_times(p_value):
            solved = spec.solver.solve(
                spec.theta_0, spec.residual_args, p=p_value, **spec.options
            )
            k, c, _, _, _ = path_values(solved.x, K_sens, K_tilde_sens, p_value)
            return jnp.column_stack([k, c]), solved.status

        jacobian, status = jax.jacrev(paths_at_sensitivity_times, has_aux=True)(spec.p)
        jax.block_until_ready(jacobian)
        # A non-converged re-solve would publish exactly zero implicit
        # tangents, indistinguishable from a genuine null effect.
        if int(status) != LMStatus.CONVERGED:
            raise RuntimeError("the sensitivity re-solve did not converge")
        k_sens, c_sens, _, _, _ = path_values(
            extras["result"].x, K_sens, K_tilde_sens, spec.p
        )
        fields = {}
        for index, time_value in enumerate(SENSITIVITY_TIMES):
            suffix = f"T{time_value:g}".replace(".", "p")
            fields[f"k_{suffix}"] = float(k_sens[index])
            fields[f"c_{suffix}"] = float(c_sens[index])
            for name in ("a", "delta", "rho_hat", "sigma_crra", "k_0"):
                fields[f"d_k_{suffix}_d_{name}"] = float(jacobian[name][index, 0])
                fields[f"d_c_{suffix}_d_{name}"] = float(jacobian[name][index, 1])
        # The exact steady-state derivatives the large-T columns converge
        # to; the sigma_crra and k_0 rows converge to zero.
        fields.update(
            neoclassical_steady_state_sensitivities(
                results["a"], results["delta"], results["rho_hat"]
            )
        )
        save_experiment(
            "sensitivity",
            pd.DataFrame([{**results, **fields}]),
            {
                key: value
                for key, value in extras.items()
                if isinstance(value, pd.DataFrame)
            },
            output_dir,
            meta,
            prefix=PREFIX,
            hyperparameters=extras["hyperparameters"],
        )

    # nu / ell robustness table rows. The (nu=1/2, ell=10) row aliases the
    # baseline experiment in collect_json, so it is not re-run here.
    go("nu15_ell10", kernel_set=dict(kernel="matern32", ell=10.0))
    go("nu25_ell10", kernel_set=dict(kernel="matern52", ell=10.0))
    go("nu05_ell2", kernel_set=dict(kernel="matern12", ell=2.0))
    go("nu05_ell20", kernel_set=dict(kernel="matern12", ell=20.0))
    # Gaussian/RBF (nu -> inf) referee rows: ill-conditioned Gram matrices at
    # large ell may reject -- a first-class outcome rendered as "--" in the table.
    go("rbf_ell10", kernel_set=dict(kernel="rbf", ell=10.0))
    go("rbf_ell2", kernel_set=dict(kernel="rbf", ell=2.0))
    go("rbf_ell20", kernel_set=dict(kernel="rbf", ell=20.0))

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
