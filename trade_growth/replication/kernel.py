from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jsonargparse

import trade_growth.kernel as trade_growth_kernel
from replication.utilities import (
    enable_compilation_cache,
    ensemble,
    save_experiment,
    write_summary,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()

PREFIX = "multicountry_trade_growth_kernel"
SOLVER = trade_growth_kernel.multicountry_trade_growth_kernel
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


def run(output_dir: str = ".results", use_float64: bool = True):
    meta = dict(
        n_seeds=1,
        backend="cpu",
        residual_dtype="float64" if use_float64 else "float32",
    )

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options={"use_float64": use_float64, **experiment},
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
    transition_data = dict(
        train_T=120.0,
        train_points=32,
        dense_T=40.0,
        tail_points=8,
        test_T=80.0,
        test_points=161,
        validation_points=241,
        quality_T=100.0,
    )
    transition_lm = dict(init_damping=1e-8, residual_atol=1e-10, max_steps=100)
    for name, initial_nfa in (
        ("initial_nfa", (-0.2, 0.1, 0.1)),
        ("zero_nfa", (0.0, 0.0, 0.0)),
        ("reversed_nfa", (0.2, -0.1, -0.1)),
        ("large_debt", (-0.4, 0.2, 0.2)),
    ):
        go(
            f"three_country_{name}",
            a_0=initial_nfa,
            parameters=THREE_COUNTRY_PARAMETERS,
            data_set=transition_data,
            lm_set=transition_lm,
        )
    write_summary(output_dir, PREFIX)


if __name__ == "__main__":
    jsonargparse.CLI(run)
