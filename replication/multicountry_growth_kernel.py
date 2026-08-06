from __future__ import annotations

import json
import pathlib

import jsonargparse
import numpy as np

from models import multicountry_growth_kernel
from models.multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    steady_state,
)
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

PREFIX = "multicountry_growth_kernel"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel
# The three bond configurations share the drawn capital stocks and differ
# only in the initial bond distribution; under the common gamma their
# (r*, k_i, q_i) paths must coincide exactly, so the measured maximum gap
# across configurations is the N-country analogue of the small open
# economy's comparative-statics check.
BOND_CONFIGURATIONS = ("initial_debt", "baseline", "initial_assets")


def bond_invariance(validation_frames, num_countries, test_T):
    # Reported on two horizons: the test window, where the solution is
    # benchmark-certified, and the full training horizon, whose sparse tail
    # collocation lets the configurations drift apart by orders of
    # magnitude more.
    reference = validation_frames[BOND_CONFIGURATIONS[0]]
    columns = ["interest_rate"] + [
        f"{quantity}_{country}"
        for quantity in ("k", "q")
        for country in range(1, num_countries + 1)
    ]

    def gaps(mask):
        out = {"interest_rate": 0.0, "k": 0.0, "q": 0.0}
        for name in BOND_CONFIGURATIONS[1:]:
            frame = validation_frames[name]
            if not np.array_equal(frame["t"].to_numpy(), reference["t"].to_numpy()):
                raise ValueError("bond configurations must share the validation grid")
            for column in columns:
                quantity = column.split("_")[0] if column != "interest_rate" else column
                out[quantity] = max(
                    out[quantity],
                    float(
                        np.max(np.abs(frame[column][mask] - reference[column][mask]))
                    ),
                )
        return out

    test_gaps = gaps(reference["t"] <= test_T)
    full_gaps = gaps(reference["t"] == reference["t"])
    return {
        "configurations": list(BOND_CONFIGURATIONS),
        "test_T": test_T,
        "max_interest_rate_gap": test_gaps["interest_rate"],
        "max_k_gap": test_gaps["k"],
        "max_q_gap": test_gaps["q"],
        "max_gap": max(test_gaps.values()),
        "max_interest_rate_gap_train_horizon": full_gaps["interest_rate"],
        "max_k_gap_train_horizon": full_gaps["k"],
        "max_q_gap_train_horizon": full_gaps["q"],
        "max_gap_train_horizon": max(full_gaps.values()),
    }


def run(output_dir: str | None = None, use_float64: bool = True):
    configure_process()
    if output_dir is None:
        output_dir = ".results" if use_float64 else ".results_float32"
    parameters = MulticountryParameters()
    drawn_k_0, _ = draw_initial_conditions(InitialConditionSettings(), parameters)
    output_scale = float(steady_state(parameters)["output_scale"])
    # Float32 runs pass looser solver thresholds and acceptance gates; the
    # float64 run uses the solver defaults.
    lm = {} if use_float64 else dict(ridge=1e-7, ridge_init=1e-5, gtol=1e-6)
    tol = (
        {}
        if use_float64
        else dict(
            train_residual=1e-3,
            validation_residual=1e-2,
            benchmark_match=1e-2,
        )
    )
    meta = dict(n_seeds=1, residual_dtype="float64" if use_float64 else "float32")
    validation_frames = {}

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options={
                "use_float64": use_float64,
                "lm_set": lm,
                "tolerance_set": tol,
                **experiment,
            },
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
        validation_frames[name] = frames["validation_df"]

    go("baseline")
    go(
        "initial_debt",
        k_0=drawn_k_0,
        a_0=(-0.5 * output_scale, 0.5 * output_scale),
    )
    go(
        "initial_assets",
        k_0=drawn_k_0,
        a_0=(0.5 * output_scale, -0.5 * output_scale),
    )
    go(
        "three_countries",
        parameters=dict(
            gammas=(2.0, 2.0, 2.0),
            populations=(1.0, 1.0, 1.0),
            productivities=(1.0, 1.0, 1.0),
        ),
    )
    go(
        "heterogeneous_countries",
        parameters=dict(
            populations=(1.0, 2.0),
            productivities=(1.0, 1.1),
        ),
    )
    write_summary(
        output_dir,
        PREFIX,
        experiments={
            "baseline",
            "initial_debt",
            "initial_assets",
            "three_countries",
            "heterogeneous_countries",
        },
    )
    invariance = bond_invariance(
        validation_frames,
        parameters.num_countries,
        multicountry_growth_kernel.DataSettings().test_T,
    )
    summary_path = pathlib.Path(output_dir) / f"{PREFIX}_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["bond_invariance"] = invariance
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        "bond invariance max gap (r*, k, q): "
        f"{invariance['max_interest_rate_gap']:.2e}, "
        f"{invariance['max_k_gap']:.2e}, {invariance['max_q_gap']:.2e}"
    )


if __name__ == "__main__":
    jsonargparse.CLI(run)
