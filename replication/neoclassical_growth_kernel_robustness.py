from __future__ import annotations

import jsonargparse

from replication.neoclassical_growth_kernel import PREFIX, run_experiment
from replication.utilities import configure_process, write_summary


# nu / ell robustness table rows. The (nu=1/2, ell=10) row aliases the
# baseline experiment in collect_json, so it is not re-run here.
def run(
    output_dir: str | None = None,
    use_float64: bool = True,
):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"

    run_experiment(
        "nu15_ell10",
        output_dir,
        use_float64,
        kernel_set=dict(kernel="matern32", ell=10.0),
    )
    run_experiment(
        "nu25_ell10",
        output_dir,
        use_float64,
        kernel_set=dict(kernel="matern52", ell=10.0),
    )
    run_experiment(
        "nu05_ell2",
        output_dir,
        use_float64,
        kernel_set=dict(kernel="matern12", ell=2.0),
    )
    run_experiment(
        "nu05_ell20",
        output_dir,
        use_float64,
        kernel_set=dict(kernel="matern12", ell=20.0),
    )
    # Gaussian/RBF (nu -> inf) referee rows: ill-conditioned Gram matrices at
    # large ell may reject -- a first-class outcome rendered as "--" in the table.
    run_experiment(
        "rbf_ell10", output_dir, use_float64, kernel_set=dict(kernel="rbf", ell=10.0)
    )
    run_experiment(
        "rbf_ell2", output_dir, use_float64, kernel_set=dict(kernel="rbf", ell=2.0)
    )
    run_experiment(
        "rbf_ell20", output_dir, use_float64, kernel_set=dict(kernel="rbf", ell=20.0)
    )

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
