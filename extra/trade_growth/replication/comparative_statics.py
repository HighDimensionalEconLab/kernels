from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import time
import timeit

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from jax.flatten_util import ravel_pytree

from extra.trade_growth.comparative_statics import (
    build_multicountry_trade_comparative_statics,
    full_structural_jacobian,
    trade_growth_statistics,
)
from extra.trade_growth.kernel import DataSettings, SolverSettings
from extra.trade_growth.model import TradeGrowthParameters
from replication.utilities import enable_compilation_cache

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

enable_compilation_cache()


def heterogeneous_four_country_parameters():
    return TradeGrowthParameters(
        populations=(1.0, 1.4, 0.8, 1.2),
        intermediate_productivities=(1.0, 1.08, 0.94, 1.03),
        consumption_productivities=(1.0, 0.97, 1.06, 1.02),
        investment_productivities=(1.0, 1.04, 0.96, 1.08),
        consumption_value_added_shares=(0.75, 0.70, 0.80, 0.73),
        intermediate_value_added_shares=(0.50, 0.55, 0.45, 0.52),
        investment_value_added_shares=(0.50, 0.47, 0.54, 0.44),
        trade_costs=(
            (1.0, 1.30, 1.60, 1.40),
            (1.40, 1.0, 1.35, 1.55),
            (1.50, 1.45, 1.0, 1.25),
            (1.35, 1.60, 1.30, 1.0),
        ),
    )


def selected_shock_directions(problem):
    p = problem.structural_parameters
    parameters = p["parameters"]
    zero = jax.tree.map(jnp.zeros_like, p)
    zero_parameters = zero["parameters"]

    def parameter_direction(**updates):
        return {
            **zero,
            "parameters": dataclasses.replace(zero_parameters, **updates),
        }

    def scalar_proportional(name):
        return parameter_direction(**{name: 0.01 * getattr(parameters, name)})

    def country_proportional(name, country):
        values = getattr(parameters, name)
        return parameter_direction(
            **{name: jnp.zeros_like(values).at[country].set(0.01 * values[country])}
        )

    off_diagonal = ~jnp.eye(parameters.num_countries, dtype=bool)
    all_trade_costs = jnp.where(
        off_diagonal,
        0.01 * parameters.trade_costs,
        0.0,
    )

    def trade_cost_direction(pairs):
        direction = jnp.zeros_like(parameters.trade_costs)
        for importer, exporter in pairs:
            direction = direction.at[importer, exporter].set(
                0.01 * parameters.trade_costs[importer, exporter]
            )
        return parameter_direction(trade_costs=direction)

    capital = jnp.zeros_like(p["k_0"]).at[0].set(0.01 * p["k_0"][0])
    transfer_total = 0.01 * problem.aggregate_income_scale
    assets = (
        jnp.zeros_like(p["a_0"])
        .at[0]
        .set(transfer_total / parameters.populations[0])
        .at[-1]
        .set(-transfer_total / parameters.populations[-1])
    )
    return {
        "rho_up_1pct": scalar_proportional("rho"),
        "delta_up_1pct": scalar_proportional("delta"),
        "gamma_up_1pct": scalar_proportional("gamma"),
        "installation_elasticity_up_1pp": parameter_direction(
            installation_elasticity=jnp.asarray(
                0.01,
                dtype=parameters.rho.dtype,
            )
        ),
        "trade_elasticity_up_1pct": scalar_proportional("trade_elasticity"),
        "all_trade_costs_up_1pct": parameter_direction(trade_costs=all_trade_costs),
        "trade_cost_1_from_2_up_1pct": trade_cost_direction(((0, 1),)),
        "trade_cost_2_from_1_up_1pct": trade_cost_direction(((1, 0),)),
        "trade_costs_1_2_both_directions_up_1pct": trade_cost_direction(
            ((0, 1), (1, 0))
        ),
        "country_1_inward_trade_costs_up_1pct": trade_cost_direction(
            tuple((0, exporter) for exporter in range(1, parameters.num_countries))
        ),
        "country_1_intermediate_productivity_up_1pct": country_proportional(
            "intermediate_productivities",
            0,
        ),
        "country_1_consumption_productivity_up_1pct": country_proportional(
            "consumption_productivities",
            0,
        ),
        "country_1_investment_productivity_up_1pct": country_proportional(
            "investment_productivities",
            0,
        ),
        "country_1_initial_capital_up_1pct": {**zero, "k_0": capital},
        "initial_nfa_transfer_1pct_world_income_country_4_to_1": {
            **zero,
            "a_0": assets,
        },
    }


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem = build_multicountry_trade_comparative_statics(
        k_0=(1.0, 0.85, 1.15, 0.95),
        a_0=(-0.20, -0.05, 0.12, 0.145),
        parameters=heterogeneous_four_country_parameters(),
        data_set=DataSettings(
            train_T=120.0,
            train_points=32,
            dense_T=40.0,
            tail_points=8,
            test_T=80.0,
            test_points=161,
            validation_points=241,
            quality_T=100.0,
        ),
        lm_set=SolverSettings(max_steps=100),
        statistic_times=(0.0, 40.0, 100.0),
    )
    p = problem.structural_parameters

    def timed_statistics():
        values = problem.statistics()
        jax.block_until_ready(values)
        return values

    timed_statistics()
    solve_times = timeit.repeat(timed_statistics, repeat=3, number=1)

    jacobian_start = time.perf_counter()
    statistics, jacobian = full_structural_jacobian(problem)
    jax.block_until_ready(jacobian)
    jacobian_seconds = time.perf_counter() - jacobian_start
    interpretable_jacobian = jacobian * problem.parameter_coordinate_scales[None, :]

    statistics_frame = pd.DataFrame(
        {
            "statistic": problem.statistic_names,
            "value": np.asarray(statistics),
        }
    )
    jacobian_frame = pd.DataFrame(
        np.asarray(jacobian),
        index=problem.statistic_names,
        columns=problem.parameter_names,
    )
    jacobian_frame.index.name = "statistic"
    interpretable_frame = pd.DataFrame(
        np.asarray(interpretable_jacobian),
        index=problem.statistic_names,
        columns=problem.parameter_coordinate_names,
    )
    interpretable_frame.index.name = "statistic"

    flat_p, _ = ravel_pytree(p)
    direction_columns = {}
    for name, direction in selected_shock_directions(problem).items():
        flat_direction, _ = ravel_pytree(direction)
        if flat_direction.shape != flat_p.shape:
            raise RuntimeError(f"shock direction {name} has the wrong shape")
        direction_columns[name] = np.asarray(jacobian @ flat_direction)
    shocks_frame = pd.DataFrame(
        direction_columns,
        index=problem.statistic_names,
    )
    shocks_frame.index.name = "statistic"

    solution = problem.baseline_solution
    residual = problem.residual_fn(solution.x, None, p)
    baseline_values = problem.evaluate_parametric(
        problem.statistic_times,
        solution.x,
        p,
    )
    baseline_statistics = trade_growth_statistics(baseline_values, p)
    np.testing.assert_allclose(statistics, baseline_statistics, rtol=2e-8, atol=2e-9)
    metadata = {
        "status": int(solution.status),
        "steps": int(solution.steps),
        "final_damping": float(solution.lm_state.damping),
        "final_ridge": float(solution.lm_state.ridge),
        "max_scaled_dae_residual": float(jnp.max(jnp.abs(residual))),
        "median_warm_solve_seconds": float(np.median(solve_times)),
        "solve_timing_repetitions": len(solve_times),
        "jacobian_seconds": jacobian_seconds,
        "num_statistics": len(problem.statistic_names),
        "num_parameters": len(problem.parameter_names),
        "jacobian_method": "ridge_lm_gauss_newton_implicit_vjp",
        "parameterization": "full_p_with_initial_conditions_and_trade_cost_matrix",
    }

    prefix = out / "multicountry_trade_growth_comparative_statics"
    statistics_frame.to_csv(
        prefix.with_name(f"{prefix.name}_statistics.csv"),
        index=False,
    )
    jacobian_frame.to_csv(prefix.with_name(f"{prefix.name}_jacobian.csv"))
    interpretable_frame.to_csv(
        prefix.with_name(f"{prefix.name}_interpretable_jacobian.csv")
    )
    shocks_frame.to_csv(prefix.with_name(f"{prefix.name}_selected_shocks.csv"))
    prefix.with_name(f"{prefix.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    jsonargparse.CLI(run)
