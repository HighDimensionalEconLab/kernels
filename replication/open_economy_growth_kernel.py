from __future__ import annotations

import json
import pathlib

import jsonargparse
import numpy as np

from models import open_economy_growth_kernel
from models.open_economy_growth import ModelParameters, steady_state
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "open_economy_growth_kernel"
ASSET_CONFIGURATIONS = ("initial_debt", "baseline", "initial_assets")


def run(output_dir: str = ".results"):
    configure_process()
    parameters = ModelParameters()
    output_ss = float(steady_state(parameters)["output"])
    meta = dict(n_seeds=1, residual_dtype="float64")
    test_frames = {}
    initial_assets = {}
    resolved_rates = {}

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            open_economy_growth_kernel.open_economy_growth_kernel,
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
        test_frames[name] = frames["test_df"]
        initial_assets[name] = float(results_df["a_0"].iloc[0])
        resolved_rates[name] = float(results_df["r_star_resolved"].iloc[0])

    go("baseline")
    go("r_star_005", parameters=dict(r_star=0.05))
    go("initial_debt", a_0=-0.5 * output_ss)
    go("initial_assets", a_0=0.5 * output_ss)
    write_summary(output_dir, PREFIX)
    reference = test_frames["baseline"]
    comparative_statics = {
        "configurations": list(ASSET_CONFIGURATIONS),
        "test_T": float(reference["t"].iloc[-1]),
        "max_k_gap": 0.0,
        "max_q_gap": 0.0,
        "max_investment_gap": 0.0,
        "max_current_account_gap": 0.0,
        "max_consumption_shift_error": 0.0,
        "max_asset_shift_error": 0.0,
    }
    for name in ("initial_debt", "initial_assets"):
        frame = test_frames[name]
        if not np.array_equal(frame["t"].to_numpy(), reference["t"].to_numpy()):
            raise ValueError(
                "open-economy asset configurations must share the test grid"
            )
        if resolved_rates[name] != resolved_rates["baseline"]:
            raise ValueError(
                "open-economy asset configurations must share the world rate"
            )
        for column, field in (
            ("k", "max_k_gap"),
            ("q", "max_q_gap"),
            ("x", "max_investment_gap"),
            ("current_account", "max_current_account_gap"),
        ):
            comparative_statics[field] = max(
                comparative_statics[field],
                float(np.max(np.abs(frame[column] - reference[column]))),
            )
        asset_shift = initial_assets[name] - initial_assets["baseline"]
        consumption_shift = resolved_rates["baseline"] * asset_shift
        comparative_statics["max_consumption_shift_error"] = max(
            comparative_statics["max_consumption_shift_error"],
            float(np.max(np.abs((frame["c"] - reference["c"]) - consumption_shift))),
        )
        comparative_statics["max_asset_shift_error"] = max(
            comparative_statics["max_asset_shift_error"],
            float(np.max(np.abs((frame["a"] - reference["a"]) - asset_shift))),
        )
    summary_path = pathlib.Path(output_dir) / f"{PREFIX}_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["comparative_statics"] = comparative_statics
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        "comparative statics max gap (k, q, x, ca): "
        f"{comparative_statics['max_k_gap']:.2e}, "
        f"{comparative_statics['max_q_gap']:.2e}, "
        f"{comparative_statics['max_investment_gap']:.2e}, "
        f"{comparative_statics['max_current_account_gap']:.2e}"
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
