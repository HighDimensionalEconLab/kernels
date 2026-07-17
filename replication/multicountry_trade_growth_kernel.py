from __future__ import annotations

import jsonargparse

import multicountry_trade_growth_kernel_nlls
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

enable_compilation_cache()

PREFIX = "multicountry_trade_growth_kernel"
SOLVER = multicountry_trade_growth_kernel_nlls.multicountry_trade_growth_kernel_nlls


def run(output_dir: str = ".results"):
    meta = dict(n_seeds=1)

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options=experiment,
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

    go("two_country")
    go(
        "three_country_initial_nfa",
        a_0=(-0.2, 0.1, 0.1),
        parameters=dict(
            populations=(1.0, 1.0, 1.0),
            intermediate_productivities=(1.0, 1.0, 1.0),
            consumption_productivities=(1.0, 1.0, 1.0),
            investment_productivities=(1.0, 1.0, 1.0),
            consumption_value_added_shares=(0.75, 0.75, 0.75),
            intermediate_value_added_shares=(0.5, 0.5, 0.5),
            investment_value_added_shares=(0.5, 0.5, 0.5),
            trade_costs=(
                (1.0, 1.5, 1.5),
                (1.5, 1.0, 1.5),
                (1.5, 1.5, 1.0),
            ),
        ),
    )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
