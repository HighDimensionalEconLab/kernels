from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import json
import pathlib
import time

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from jax.flatten_util import ravel_pytree

from replication.utilities import enable_compilation_cache
from trade_growth.comparative_statics import (
    build_multicountry_trade_comparative_statics,
    full_structural_jacobian,
)
from trade_growth.model import TradeGrowthParameters

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


def _selected_shock_directions(problem):
    p = problem.structural_parameters
    zero = jax.tree.map(jnp.zeros_like, p)

    def scalar_proportional(name):
        return {**zero, name: 0.01 * p[name]}

    def country_proportional(name, country):
        direction = zero[name].at[country].set(0.01 * p[name][country])
        return {**zero, name: direction}

    num_countries = p["populations"].shape[0]
    off_diagonal_pairs = tuple(
        (importer, exporter)
        for importer in range(num_countries)
        for exporter in range(num_countries)
        if importer != exporter
    )

    def trade_cost_direction(selected_pairs):
        direction = zero["trade_costs_off_diagonal"]
        for pair in selected_pairs:
            index = off_diagonal_pairs.index(pair)
            direction = direction.at[index].set(
                0.01 * p["trade_costs_off_diagonal"][index]
            )
        return {**zero, "trade_costs_off_diagonal": direction}

    initial_capital = zero["initial_log_capital"].at[0].set(0.01)
    initial_nfa = (
        zero["initial_aggregate_assets_free"]
        .at[0]
        .set(0.01 * problem.aggregate_income_scale)
    )
    trade_costs = 0.01 * p["trade_costs_off_diagonal"]
    return {
        "rho_up_1pct": scalar_proportional("rho"),
        "delta_up_1pct": scalar_proportional("delta"),
        "gamma_up_1pct": scalar_proportional("gamma"),
        "installation_elasticity_up_1pp": {
            **zero,
            "installation_elasticity": jnp.asarray(0.01, dtype=p["rho"].dtype),
        },
        "trade_elasticity_up_1pct": scalar_proportional("trade_elasticity"),
        "all_trade_costs_up_1pct": {
            **zero,
            "trade_costs_off_diagonal": trade_costs,
        },
        "trade_cost_1_from_2_up_1pct": trade_cost_direction(((0, 1),)),
        "trade_cost_2_from_1_up_1pct": trade_cost_direction(((1, 0),)),
        "trade_costs_1_2_both_directions_up_1pct": trade_cost_direction(
            ((0, 1), (1, 0))
        ),
        "country_1_inward_trade_costs_up_1pct": trade_cost_direction(
            tuple((0, exporter) for exporter in range(1, num_countries))
        ),
        "country_1_intermediate_productivity_up_1pct": country_proportional(
            "intermediate_productivities", 0
        ),
        "country_1_consumption_productivity_up_1pct": country_proportional(
            "consumption_productivities", 0
        ),
        "country_1_investment_productivity_up_1pct": country_proportional(
            "investment_productivities", 0
        ),
        "country_1_initial_capital_up_1pct": {
            **zero,
            "initial_log_capital": initial_capital,
        },
        "initial_nfa_transfer_1pct_world_income_country_4_to_1": {
            **zero,
            "initial_aggregate_assets_free": initial_nfa,
        },
    }


def _continuation_trade_cost_shock(
    problem,
    baseline_solution,
    selected_pairs,
    log_change,
    continuation_steps=20,
):
    p = problem.structural_parameters
    num_countries = p["populations"].shape[0]
    off_diagonal_pairs = tuple(
        (importer, exporter)
        for importer in range(num_countries)
        for exporter in range(num_countries)
        if importer != exporter
    )
    selected_indices = jnp.asarray(
        tuple(off_diagonal_pairs.index(pair) for pair in selected_pairs)
    )
    baseline_trade_costs = p["trade_costs_off_diagonal"]
    theta = baseline_solution.x
    result = baseline_solution
    for step in range(1, continuation_steps + 1):
        fraction = step / continuation_steps
        trade_costs = baseline_trade_costs.at[selected_indices].set(
            baseline_trade_costs[selected_indices] * jnp.exp(fraction * log_change)
        )
        p_step = {**p, "trade_costs_off_diagonal": trade_costs}
        result = problem.solver.solve(
            theta,
            p=p_step,
            max_steps=problem.max_steps,
            atol=problem.residual_atol,
            max_steps_is_success=False,
        )
        theta = result.x
    return result


def run(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem = build_multicountry_trade_comparative_statics(
        k_0=(1.0, 0.85, 1.15, 0.95),
        a_0=(-0.20, -0.05, 0.12, 0.145),
        parameters=heterogeneous_four_country_parameters(),
    )

    solve_times = []
    jacobian_times = []
    for _ in range(5):
        solve_start = time.perf_counter()
        solution = problem.solve()
        jax.block_until_ready(solution.aux)
        solve_times.append(time.perf_counter() - solve_start)

        jacobian_start = time.perf_counter()
        statistics, jacobian = full_structural_jacobian(problem)
        jax.block_until_ready(jacobian)
        jacobian_times.append(time.perf_counter() - jacobian_start)
    solve_seconds = float(np.median(solve_times))
    jacobian_seconds = float(np.median(jacobian_times))
    interpretable_jacobian = jacobian * problem.parameter_coordinate_scales[None, :]

    statistics_array = np.asarray(statistics)
    jacobian_array = np.asarray(jacobian)
    statistics_frame = pd.DataFrame(
        {
            "statistic": problem.statistic_names,
            "value": statistics_array,
        }
    )
    jacobian_frame = pd.DataFrame(
        jacobian_array,
        index=problem.statistic_names,
        columns=problem.parameter_names,
    )
    jacobian_frame.index.name = "statistic"
    interpretable_jacobian_frame = pd.DataFrame(
        np.asarray(interpretable_jacobian),
        index=problem.statistic_names,
        columns=problem.parameter_coordinate_names,
    )
    interpretable_jacobian_frame.index.name = "statistic"

    flat_p, _ = ravel_pytree(problem.structural_parameters)
    direction_columns = {}
    for name, direction in _selected_shock_directions(problem).items():
        flat_direction, _ = ravel_pytree(direction)
        if flat_direction.shape != flat_p.shape:
            raise RuntimeError(f"shock direction {name} has the wrong shape")
        direction_columns[name] = jacobian @ flat_direction
    shocks_frame = pd.DataFrame(
        direction_columns,
        index=problem.statistic_names,
    )
    shocks_frame.index.name = "statistic"
    trade_shock_names = (
        "all_trade_costs_up_1pct",
        "trade_cost_1_from_2_up_1pct",
        "trade_cost_2_from_1_up_1pct",
        "trade_costs_1_2_both_directions_up_1pct",
        "country_1_inward_trade_costs_up_1pct",
    )
    trade_statistic_names = tuple(
        name
        for name in problem.statistic_names
        if name.startswith("log_population_weighted_home_trade_share")
        or name.startswith("log_population_weighted_import_share")
        or name.startswith("log_trade_share_")
    )
    trade_elasticities_frame = (
        shocks_frame.loc[
            trade_statistic_names,
            trade_shock_names,
        ]
        / 0.01
    )
    baseline_trade_frame = statistics_frame.set_index("statistic").loc[
        list(trade_statistic_names)
    ]
    baseline_trade_frame = baseline_trade_frame.rename(columns={"value": "log_share"})
    baseline_trade_frame["share"] = np.exp(baseline_trade_frame["log_share"])

    num_countries = problem.structural_parameters["populations"].shape[0]
    all_trade_pairs = tuple(
        (importer, exporter)
        for importer in range(num_countries)
        for exporter in range(num_countries)
        if importer != exporter
    )
    discrete_shocks = {
        "all_trade_costs_down_10pct": (all_trade_pairs, np.log(0.9)),
        "trade_costs_1_2_both_directions_up_10pct": (
            ((0, 1), (1, 0)),
            np.log(1.1),
        ),
    }
    discrete_response_columns = {}
    discrete_secant_columns = {}
    discrete_diagnostics = {}
    for name, (selected_pairs, log_change) in discrete_shocks.items():
        shocked_solution = _continuation_trade_cost_shock(
            problem,
            solution,
            selected_pairs,
            log_change,
        )
        jax.block_until_ready(shocked_solution.aux)
        response = shocked_solution.aux - statistics
        discrete_response_columns[name] = response
        discrete_secant_columns[name] = response / log_change
        shocked_residual = problem.solver.residual_fn(
            shocked_solution.x,
            None,
            shocked_solution.p,
        )[0]
        discrete_diagnostics[name] = {
            "final_continuation_steps": int(shocked_solution.steps),
            "max_dae_residual": float(jnp.max(jnp.abs(shocked_residual))),
            "log_trade_cost_change": float(log_change),
        }
    discrete_response_frame = pd.DataFrame(
        discrete_response_columns,
        index=problem.statistic_names,
    )
    discrete_response_frame.index.name = "statistic"
    discrete_secant_frame = pd.DataFrame(
        discrete_secant_columns,
        index=problem.statistic_names,
    )
    discrete_secant_frame.index.name = "statistic"

    residual = problem.solver.residual_fn(
        solution.x,
        None,
        problem.structural_parameters,
    )[0]
    metadata = {
        "status": int(solution.status),
        "steps": int(solution.steps),
        "max_dae_residual": float(jnp.max(jnp.abs(residual))),
        "solve_seconds": solve_seconds,
        "jacobian_seconds": jacobian_seconds,
        "timing_repetitions": 5,
        "num_statistics": len(problem.statistic_names),
        "num_parameters": len(problem.parameter_names),
        "jacobian_method": "nlls_gram_ad_solver_auto_vjp",
        "terminal_conditions_in_solve": False,
        "discrete_trade_shocks": discrete_diagnostics,
    }

    prefix = out / "multicountry_trade_growth_comparative_statics"
    statistics_frame.to_csv(
        prefix.with_name(f"{prefix.name}_statistics.csv"), index=False
    )
    jacobian_frame.to_csv(prefix.with_name(f"{prefix.name}_jacobian.csv"))
    interpretable_jacobian_frame.to_csv(
        prefix.with_name(f"{prefix.name}_interpretable_jacobian.csv")
    )
    shocks_frame.to_csv(prefix.with_name(f"{prefix.name}_selected_shocks.csv"))
    trade_elasticities_frame.to_csv(
        prefix.with_name(f"{prefix.name}_trade_elasticities.csv")
    )
    baseline_trade_frame.to_csv(
        prefix.with_name(f"{prefix.name}_baseline_trade_shares.csv")
    )
    discrete_response_frame.to_csv(
        prefix.with_name(f"{prefix.name}_discrete_trade_shock_responses.csv")
    )
    discrete_secant_frame.to_csv(
        prefix.with_name(f"{prefix.name}_discrete_trade_shock_secants.csv")
    )
    prefix.with_name(f"{prefix.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    report_rows = (
        *(f"log_population_weighted_import_share_t{time}" for time in (0, 40, 100)),
        *(
            f"log_trade_share_1_from_{exporter}_t{time}"
            for time in (0, 40, 100)
            for exporter in range(1, 5)
        ),
        *(
            f"log_trade_share_2_from_{exporter}_t{time}"
            for time in (0, 40, 100)
            for exporter in range(1, 5)
        ),
    )
    print("\nSelected trade elasticities:")
    print(
        trade_elasticities_frame.loc[list(report_rows)].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
    )
    return metadata


if __name__ == "__main__":
    jsonargparse.CLI(run)
