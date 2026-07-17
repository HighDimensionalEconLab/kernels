from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jsonargparse

import multicountry_trade_growth_nn
from replication.multicountry_trade_growth_kernel import THREE_COUNTRY_PARAMETERS
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

enable_compilation_cache()

PREFIX = "multicountry_trade_growth_nn"
SOLVER = multicountry_trade_growth_nn.multicountry_trade_growth_nn


def run(
    output_dir: str = ".results",
    n_seeds: int = 3,
    verbose: bool = False,
):
    meta = dict(n_seeds=n_seeds, backend="cpu")
    print(f"three_country_initial_nfa: {n_seeds} seeds -> {output_dir}/")
    results_df, frames, hyperparameters = ensemble(
        SOLVER,
        n_seeds=n_seeds,
        options={
            "a_0": (-0.2, 0.1, 0.1),
            "parameters": THREE_COUNTRY_PARAMETERS,
            "verbose": verbose,
        },
    )
    save_experiment(
        "three_country_initial_nfa",
        results_df,
        frames,
        output_dir,
        meta,
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
