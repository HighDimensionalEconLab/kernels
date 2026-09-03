from __future__ import annotations

import jsonargparse

from replication.neoclassical_growth_kernel import PREFIX, go
from replication.utilities import configure_process, write_summary


# nu / ell robustness table rows. The (nu=1/2, ell=10) row aliases the
# baseline experiment in collect_json, so it is not re-run here.
def run(
    output_dir: str = ".results",
):
    configure_process()
    # The table reports each row's validation residual without gating on it:
    # the loss of between-node precision is what the rows measure.
    validation_off = dict(gate_validation=False)

    go(
        "nu15_ell10",
        output_dir,
        tolerance_set=validation_off,
        kernel_set=dict(kernel="matern32", ell=10.0),
    )
    go(
        "nu25_ell10",
        output_dir,
        tolerance_set=validation_off,
        kernel_set=dict(kernel="matern52", ell=10.0),
    )
    go(
        "nu05_ell2",
        output_dir,
        tolerance_set=validation_off,
        kernel_set=dict(kernel="matern12", ell=2.0),
    )
    go(
        "nu05_ell20",
        output_dir,
        tolerance_set=validation_off,
        kernel_set=dict(kernel="matern12", ell=20.0),
    )
    # Gaussian/RBF (nu -> inf) referee rows: ill-conditioned Gram matrices at
    # large ell may reject -- a first-class outcome rendered as "--" in the table.
    go(
        "rbf_ell10",
        output_dir,
        kernel_set=dict(kernel="rbf", ell=10.0),
        tolerance_set=validation_off,
    )
    go(
        "rbf_ell2",
        output_dir,
        kernel_set=dict(kernel="rbf", ell=2.0),
        tolerance_set=validation_off,
    )
    go(
        "rbf_ell20",
        output_dir,
        kernel_set=dict(kernel="rbf", ell=20.0),
        tolerance_set=validation_off,
    )

    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
