from __future__ import annotations

import jsonargparse

import neoclassical_growth_kernel
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "neoclassical_growth_kernel"


def run(
    output_dir: str | None = None,
    use_float64: bool = True,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(residual_atol=1e-5, ridge=1e-7, gtol=1e-6)
    tol = (
        {}
        if use_float64
        else dict(train_residual=1e-3, validation_residual=1e-2, mpk_bound=5e-3)
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
