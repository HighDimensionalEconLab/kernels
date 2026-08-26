from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import jsonargparse
import numpy as np

from models import multicountry_growth_kernel
from models.multicountry_growth_shared import (
    MULTICOUNTRY_GROWTH_CONSTANTS,
    InitialConditionSettings,
    StructuralParameters,
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
# Sensitivity dates shared by the two-country and ten-country experiments
# (multicountry_growth_kernel_sensitivity.py and the NN sensitivity driver).
# The cross-check on d r*/d gamma_bar -- the same representative-agent
# perturbation solved through a 1,920-row and a 9,792-row system -- only
# compares at dates both tables report, so the two lists must stay equal.
SENSITIVITY_TIMES = [0.0, 10.0, 30.0, 80.0]
LARGE_COUNTRY_COUNT = 10
LARGE_COUNTRY_PARAMETERS = dict(
    gammas=(2.0,) * LARGE_COUNTRY_COUNT,
    populations=(1.0,) * LARGE_COUNTRY_COUNT,
    productivities=(1.0,) * LARGE_COUNTRY_COUNT,
)


def bond_invariance(validation_frames, num_countries, test_T):
    # Reported on two horizons: the densely collocated test window, and the
    # full training horizon, whose sparse tail collocation lets the
    # configurations drift apart by orders of magnitude more.
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


def run(output_dir: str = ".results"):
    configure_process()
    drawn_k_0, _ = draw_initial_conditions(
        InitialConditionSettings(), MULTICOUNTRY_GROWTH_CONSTANTS["populations"]
    )
    output_scale = float(
        steady_state(
            StructuralParameters.create(
                dtype=jnp.float64, **MULTICOUNTRY_GROWTH_CONSTANTS
            )
        )["output_scale"]
    )
    meta = dict(n_seeds=1, residual_dtype="float64")
    validation_frames = {}

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
        gammas=(2.0, 2.0, 2.0),
        populations=(1.0, 1.0, 1.0),
        productivities=(1.0, 1.0, 1.0),
    )
    go(
        "heterogeneous_countries",
        populations=(1.0, 2.0),
        productivities=(1.0, 1.1),
    )
    # Symmetric in preferences, size, and technology: the only heterogeneity
    # is the drawn capital and the drawn bond positions, so the aggregate
    # comparative statics isolate what a wide cross-section adds over two
    # countries rather than what asymmetric calibration adds.
    go("ten_countries", **LARGE_COUNTRY_PARAMETERS)
    write_summary(
        output_dir,
        PREFIX,
        experiments={
            "baseline",
            "initial_debt",
            "initial_assets",
            "three_countries",
            "heterogeneous_countries",
            "ten_countries",
        },
    )
    invariance = bond_invariance(
        validation_frames,
        len(MULTICOUNTRY_GROWTH_CONSTANTS["populations"]),
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
