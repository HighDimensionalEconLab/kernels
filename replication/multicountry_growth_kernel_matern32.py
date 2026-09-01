from __future__ import annotations

import json
import pathlib

import jsonargparse

from models import multicountry_growth_kernel
from models.multicountry_growth_shared import (
    MULTICOUNTRY_GROWTH_CONSTANTS,
    InitialConditionSettings,
    draw_initial_conditions,
)
from replication.multicountry_growth_kernel import BOND_CONFIGURATIONS, bond_invariance
from replication.utilities import (
    configure_process,
    ensemble,
    save_experiment,
    write_summary,
)

# The published bond-configuration matrix rerun under matern32 (nu = 3/2):
# identical settings except the kernel family, so the supplement's
# configuration-comparison table isolates what the smoother kernel changes.
PREFIX = "multicountry_growth_kernel_matern32"
SOLVER = multicountry_growth_kernel.multicountry_growth_kernel
KERNEL_SET = {"kernel": "matern32", "ell": 20.0}


def run(output_dir: str = ".results"):
    configure_process()
    drawn_k_0, _ = draw_initial_conditions(
        InitialConditionSettings(), MULTICOUNTRY_GROWTH_CONSTANTS["populations"]
    )
    meta = dict(n_seeds=1, residual_dtype="float64")
    validation_frames = {}

    def go(name, **experiment):
        print(f"{name} -> {output_dir}/")
        results_df, frames, hyperparameters = ensemble(
            SOLVER,
            n_seeds=1,
            options={"kernel_set": KERNEL_SET, **experiment},
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
        return results_df

    baseline = go("baseline")
    # Bond configurations sized by the baseline run's own steady-state
    # output scale.
    output_scale = float(baseline["output_scale"].iloc[0])
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
    write_summary(output_dir, PREFIX, experiments=set(BOND_CONFIGURATIONS))
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
