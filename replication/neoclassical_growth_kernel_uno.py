from __future__ import annotations

import math

import jax
import jsonargparse

import neoclassical_growth_kernel_uno
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "neoclassical_growth_kernel_uno"


def run(
    output_dir: str = ".results",
    verbose: bool = False,
):
    meta = dict(n_seeds=1)

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            neoclassical_growth_kernel_uno.neoclassical_growth_kernel_uno,
            n_seeds=1,
            options={"verbose": verbose, **experiment},
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
    go("nu15_ell10", kernel_set=dict(nu=1.5, ell=10.0))
    go("nu25_ell10", kernel_set=dict(nu=2.5, ell=10.0))
    go("nu05_ell2", kernel_set=dict(nu=0.5, ell=2.0))
    go("nu05_ell20", kernel_set=dict(nu=0.5, ell=20.0))
    # Gaussian/RBF (nu -> inf) referee rows: ill-conditioned Gram matrices at
    # large ell may reject -- a first-class outcome rendered as "--" in the table.
    go("rbf_ell10", kernel_set=dict(nu=math.inf, ell=10.0))
    go("rbf_ell2", kernel_set=dict(nu=math.inf, ell=2.0))
    go("rbf_ell20", kernel_set=dict(nu=math.inf, ell=20.0))

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
