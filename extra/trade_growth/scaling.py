from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import asdict

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np

from extra.trade_growth.kernel import (
    DataSettings,
    SolverSettings,
    multicountry_trade_growth_kernel,
)
from extra.trade_growth.model import TradeGrowthParameters
from multicountry_growth import InitialConditionSettings, draw_initial_conditions

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def draw_trade_growth_parameters(
    num_countries,
    seed=0,
    trade_cost=1.5,
    population_log_sigma=0.1,
    productivity_log_sigma=0.05,
    value_added_share_sigma=0.02,
    trade_cost_log_sigma=0.05,
):
    keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), num_countries), 8)

    def lognormal(key, sigma):
        return tuple(
            float(value)
            for value in jnp.exp(
                sigma * jax.random.normal(key, (num_countries,), dtype=jnp.float64)
            )
        )

    def shares(key, center):
        values = center + value_added_share_sigma * jax.random.normal(
            key, (num_countries,), dtype=jnp.float64
        )
        return tuple(float(value) for value in jnp.clip(values, 0.1, 0.9))

    trade_noise = jax.random.normal(
        keys[7], (num_countries, num_countries), dtype=jnp.float64
    )
    trade_costs = 1.0 + (trade_cost - 1.0) * jnp.exp(trade_cost_log_sigma * trade_noise)
    trade_costs = trade_costs.at[jnp.diag_indices(num_countries)].set(1.0)
    return TradeGrowthParameters(
        populations=lognormal(keys[0], population_log_sigma),
        intermediate_productivities=lognormal(keys[1], productivity_log_sigma),
        consumption_productivities=lognormal(keys[2], productivity_log_sigma),
        investment_productivities=lognormal(keys[3], productivity_log_sigma),
        consumption_value_added_shares=shares(keys[4], 0.75),
        intermediate_value_added_shares=shares(keys[5], 0.5),
        investment_value_added_shares=shares(keys[6], 0.5),
        trade_costs=tuple(tuple(float(value) for value in row) for row in trade_costs),
    )


def run_ladder(
    country_counts: list[int] = [2, 3],  # noqa: B006
    linear_solvers: list[str] = ["auto", "cg"],  # noqa: B006
    dense_max_countries: int = 5,
    seed: int = 0,
    trade_cost: float = 1.5,
    heterogeneous_parameters: bool = True,
    train_T: float = 100.0,
    train_points: int = 48,
    train_points_by_country: dict[int, int] | None = None,
    dense_T: float = 40.0,
    tail_points: int = 8,
    test_T: float = 50.0,
    test_points: int = 201,
    validation_points: int = 401,
    max_steps: int = 100,
    optimizer: str = "ridge_lm",
    metric_eps: float = 1e-7,
    init_damping: float = 1e-8,
    damping_decrease: float = 0.1,
    residual_atol: float = 0.0,
    geodesic_acceleration: bool = True,
    preconditioner_refresh_steps: int = 0,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 2500,
    output_path: str | None = None,
):
    initial_set = InitialConditionSettings(seed=seed)
    records = []
    for num_countries in country_counts:
        points = (
            train_points
            if train_points_by_country is None
            else train_points_by_country.get(num_countries, train_points)
        )
        data_set = DataSettings(
            train_T=train_T,
            train_points=points,
            sampling="front_loaded",
            dense_T=dense_T,
            tail_points=tail_points,
            test_T=test_T,
            test_points=test_points,
            validation_points=validation_points,
            quality_T=train_T,
        )
        if heterogeneous_parameters:
            parameters = draw_trade_growth_parameters(
                num_countries,
                seed=seed,
                trade_cost=trade_cost,
            )
        else:
            parameters = TradeGrowthParameters(
                populations=(1.0,) * num_countries,
                intermediate_productivities=(1.0,) * num_countries,
                consumption_productivities=(1.0,) * num_countries,
                investment_productivities=(1.0,) * num_countries,
                consumption_value_added_shares=(0.75,) * num_countries,
                intermediate_value_added_shares=(0.5,) * num_countries,
                investment_value_added_shares=(0.5,) * num_countries,
                trade_costs=tuple(
                    tuple(
                        1.0 if importer == exporter else trade_cost
                        for exporter in range(num_countries)
                    )
                    for importer in range(num_countries)
                ),
            )
        k_0, a_0 = draw_initial_conditions(initial_set, parameters, dtype=jnp.float64)
        dense_extra = None
        for linear_solver in linear_solvers:
            if linear_solver in ("auto", "cholesky") and (
                num_countries > dense_max_countries
            ):
                continue
            lm_set = SolverSettings(
                optimizer=optimizer,
                metric_eps=metric_eps,
                linear_solver=linear_solver,
                preconditioner=(
                    "hybrid_blocks" if linear_solver == "gram_cg" else "balanced_modes"
                ),
                init_damping=init_damping,
                damping_decrease=damping_decrease,
                residual_atol=residual_atol,
                geodesic_acceleration=geodesic_acceleration,
                preconditioner_refresh_steps=preconditioner_refresh_steps,
                max_steps=max_steps,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
            )
            start = time.perf_counter()
            results, extra = multicountry_trade_growth_kernel(
                k_0=k_0,
                a_0=a_0,
                parameters=parameters,
                data_set=data_set,
                lm_set=lm_set,
            )
            total_time = time.perf_counter() - start
            dense_path_error = None
            if linear_solver in ("auto", "cholesky"):
                dense_extra = extra
            elif dense_extra is not None:
                path_errors = []
                for name in ("k", "q", "marginal_wealth", "consumption"):
                    for country in range(1, num_countries + 1):
                        candidate = np.asarray(
                            extra["validation_df"][f"{name}_{country}"]
                        )
                        dense_values = np.asarray(
                            dense_extra["validation_df"][f"{name}_{country}"]
                        )
                        path_errors.append(
                            np.max(np.abs(candidate - dense_values))
                            / max(np.max(np.abs(dense_values)), 1e-12)
                        )
                for country in range(1, num_countries + 1):
                    candidate = np.asarray(extra["validation_df"][f"a_{country}"])
                    dense_values = np.asarray(
                        dense_extra["validation_df"][f"a_{country}"]
                    )
                    path_errors.append(np.max(np.abs(candidate - dense_values)))
                dense_path_error = float(max(path_errors))
            record = {
                "num_countries": num_countries,
                "optimizer": optimizer,
                "metric_eps": metric_eps,
                "preconditioner": lm_set.preconditioner,
                "linear_solver": linear_solver,
                "solve_time": results["solve_time"],
                "total_time": total_time,
                "steps": results["steps"],
                "asset_homotopy_used": results["asset_homotopy_used"],
                "asset_homotopy_solve_steps": results["asset_homotopy_solve_steps"],
                "accepted": results["solution_accepted"],
                "rejection_reason": results["rejection_reason"],
                "max_train_residual": results["max_train_residual"],
                "max_validation_residual": results["max_validation_residual"],
                "max_asset_clearing_residual": results["max_asset_clearing_residual"],
                "max_rate_consistency_residual": results[
                    "max_rate_consistency_residual"
                ],
                "max_world_trade_balance": results["max_world_trade_balance"],
                "dense_path_error": dense_path_error,
                "tail_interest_rate_gap": results["tail_interest_rate_gap"],
                "tail_max_current_account": results["tail_max_current_account"],
                "data_set": asdict(data_set),
            }
            records.append(record)
            print(
                f"N={num_countries:2d} {linear_solver:8s} "
                f"solve={record['solve_time']:.2f}s total={total_time:.2f}s "
                f"steps={record['steps']:2d} accepted={record['accepted']} "
                f"train={record['max_train_residual']:.2e} "
                f"validation={record['max_validation_residual']:.2e} "
                f"asset={record['max_asset_clearing_residual']:.2e} "
                f"dense_error={dense_path_error}",
                flush=True,
            )
    failures = [record for record in records if not record["accepted"]]
    parity_failures = [
        record
        for record in records
        if record["dense_path_error"] is not None and record["dense_path_error"] > 1e-6
    ]
    if output_path is not None:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2))
        print(f"wrote {path}", flush=True)
    if failures or parity_failures:
        raise SystemExit(
            f"{len(failures)} rejected solve(s), "
            f"{len(parity_failures)} dense-parity failure(s)"
        )
    return records


if __name__ == "__main__":
    jsonargparse.CLI(run_ladder)
