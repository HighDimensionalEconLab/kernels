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
THREE_COUNTRY_PARAMETERS = dict(
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
)


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
        parameters=THREE_COUNTRY_PARAMETERS,
    )
    horizon_settings = (
        (120.0, 32, 40.0, 8),
        (160.0, 40, 50.0, 10),
        (240.0, 72, 80.0, 20),
    )
    for horizon, points, dense_horizon, tail_points in horizon_settings:
        go(
            f"three_country_horizon_{round(horizon)}",
            a_0=(-0.2, 0.1, 0.1),
            parameters=THREE_COUNTRY_PARAMETERS,
            data_set=dict(
                train_T=horizon,
                train_points=points,
                dense_T=dense_horizon,
                tail_points=tail_points,
                test_T=40.0,
                test_points=81,
                validation_points=321,
            ),
            lm_set=dict(
                init_damping=1e-8,
                residual_atol=1e-10,
                max_steps=100,
            ),
        )
    long_data = dict(
        train_T=240.0,
        train_points=72,
        dense_T=80.0,
        tail_points=20,
        test_T=40.0,
        test_points=81,
        validation_points=321,
    )
    long_lm = dict(init_damping=1e-8, residual_atol=1e-10, max_steps=100)
    for name, initial_nfa in (
        ("zero_nfa", (0.0, 0.0, 0.0)),
        ("reversed_nfa", (0.2, -0.1, -0.1)),
        ("large_debt", (-0.4, 0.2, 0.2)),
    ):
        go(
            f"three_country_{name}",
            a_0=initial_nfa,
            parameters=THREE_COUNTRY_PARAMETERS,
            data_set=long_data,
            lm_set=long_lm,
        )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
