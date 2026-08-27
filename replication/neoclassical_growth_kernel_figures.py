from __future__ import annotations

import json
import pathlib

import jsonargparse
import pandas as pd

from replication.utilities import plot_neoclassical_growth

PREFIX = "neoclassical_growth_kernel"


def load(out, name):
    test_df = pd.read_parquet(out / f"{PREFIX}_{name}_test_df.parquet")
    hp = json.loads((out / f"{PREFIX}_{name}_hyperparameters.json").read_text())
    return test_df, float(hp["data_set"]["train_T"])


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)

    test_df, train_T = load(out, "baseline")
    plot_neoclassical_growth(
        test_df,
        train_T,
        out / "neoclassical_growth_model_baseline.pdf",
        k_rel_error_ylim=(1e-6, 2e-2),
        c_rel_error_ylim=(1e-7, 2e-2),
        zoom=True,
        zoom_loc=(90, 99),
        fontsize=17,
        ticksize=16,
    )

    test_df, train_T = load(out, "sparse")
    plot_neoclassical_growth(
        test_df,
        train_T,
        out / "neoclassical_growth_model_sparse.pdf",
        c_rel_error_ylim=(1e-7, 2e-2),
        zoom=True,
        zoom_loc=(10, 20),
    )

    test_df, train_T = load(out, "far_steady_state")
    plot_neoclassical_growth(
        test_df,
        train_T,
        out / "neoclassical_growth_model_far_steady_state.pdf",
        k_rel_error_ylim=(1e-4, 1e-1),
        c_rel_error_ylim=(1e-4, 1e-1),
        zoom=False,
    )
    print(f"wrote 3 growth figures -> {output_dir}/")


if __name__ == "__main__":
    jsonargparse.CLI(run)
