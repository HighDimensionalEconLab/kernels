# Scaling ladder for the two-phase multicountry trade-growth solve
# (HighDimensionalEconLab/kernels#8): for each N, run the production driver
# with linear_solver="cg" + two_phase_endgame on symmetric N-country inputs,
# assert the standing validity bars (asset_clearing <= 5e-5, train residual
# <= 1e-5), and report the per-phase timing breakdown. Reference result
# (Mac M-series CPU, float64, steady-state): N=4 2.5s / N=10 3.7s / N=25 8.0s
# / N=50 18.5s with phase1_steps=18-20.
#
#   JAX_PLATFORMS=cpu uv run python -m experiments.multicountry_trade_growth_scaling \
#     --country_counts '[4,10,25,50]' --phase1_steps 20

from __future__ import annotations

import json
import os
import pathlib
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jsonargparse

from multicountry_trade_growth import TradeGrowthParameters
from multicountry_trade_growth_kernel_nlls import (
    NLLSGramSettings,
    multicountry_trade_growth_kernel_nlls,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

ASSET_BAR = 5e-5
TRAIN_BAR = 1e-5


def symmetric_parameters(num_countries, trade_cost=1.5):
    ones = (1.0,) * num_countries
    return TradeGrowthParameters(
        populations=ones,
        intermediate_productivities=ones,
        consumption_productivities=ones,
        investment_productivities=ones,
        consumption_value_added_shares=(0.75,) * num_countries,
        intermediate_value_added_shares=(0.5,) * num_countries,
        investment_value_added_shares=(0.5,) * num_countries,
        trade_costs=tuple(
            tuple(1.0 if i == j else trade_cost for j in range(num_countries))
            for i in range(num_countries)
        ),
    )


def run_case(num_countries, phase1_steps, phase2_mode, trade_cost):
    lm_set = NLLSGramSettings(
        linear_solver="cg",
        two_phase_endgame=True,
        phase2_mode=phase2_mode,
        max_steps=phase1_steps,
    )
    outer_start = time.perf_counter()
    results, _ = multicountry_trade_growth_kernel_nlls(
        parameters=symmetric_parameters(num_countries, trade_cost), lm_set=lm_set
    )
    outer_wall = time.perf_counter() - outer_start
    asset = results["max_asset_clearing_residual"]
    train = results["max_train_residual"]
    passed = asset <= ASSET_BAR and train <= TRAIN_BAR
    row = {
        "num_countries": num_countries,
        "phase1_steps": results["steps"],
        "phase2_steps": results["phase2_steps"],
        "total_inner_iterations": results["total_inner_iterations"],
        "asset_clearing": asset,
        "train_residual": train,
        "train_loss": results["train_loss"],
        "phase1_solve_time": results["phase1_solve_time"],
        "phase2_build_time": results["phase2_build_time"],
        "phase2_solve_time": results["phase2_solve_time"],
        "wall_time": results["wall_time"],
        "compile_time": results["compile_time"],
        "outer_wall": outer_wall,
        "passed": passed,
    }
    print(
        f"[N={num_countries:2d} {phase2_mode}] p1={row['phase1_steps']:2d} "
        f"p2={row['phase2_steps']} inner={row['total_inner_iterations']:5d} "
        f"| asset={asset:.2e} train={train:.2e} loss={row['train_loss']:.2e} "
        f"| p1={row['phase1_solve_time']:.2f}s p2_build={row['phase2_build_time']:.2f}s "
        f"p2_solve={row['phase2_solve_time']:.2f}s wall={row['wall_time']:.2f}s "
        f"-> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return row


def run(
    country_counts: list[int] = [4, 10, 25, 50],
    phase1_steps: int = 20,
    phase2_mode: str = "dense",
    trade_cost: float = 1.5,
    output_path: str = "",
):
    rows = [run_case(n, phase1_steps, phase2_mode, trade_cost) for n in country_counts]
    all_passed = all(row["passed"] for row in rows)
    print(f"\nLADDER {'ALL PASS' if all_passed else 'HAS FAILURES'}", flush=True)
    if output_path:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        print(f"wrote {path}", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(jsonargparse.CLI(run))
