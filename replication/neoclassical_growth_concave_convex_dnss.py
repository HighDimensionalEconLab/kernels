from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import jsonargparse
import numpy as np

from models.neoclassical_growth_benchmark import (
    BenchmarkParameters,
    butterfly_candidate_value,
)
from models.neoclassical_growth_shared import (
    BUTTERFLY_CALIBRATION,
    StructuralParameters,
    butterfly_thresholds,
)
from replication.utilities import configure_process

PREFIX = "neoclassical_growth_concave_convex_dnss"

# The DNSS indifference point k* is the root of the value gap
# V_high(k_0) - V_low(k_0) between the benchmark's two saddle candidates;
# outside their overlap region exactly one candidate solves and the gap is
# +/- inf, so the sign scan below still brackets the root. The g > 0
# scenarios probe how trend growth moves k*; their horizons keep the
# detrended branch switch k_thr e^{-g t} above the low steady state over
# [0, T], where the two-saddle candidate structure remains meaningful.
SCENARIOS = (
    {"g": 0.0, "T": 200.0},
    {"g": 0.002, "T": 200.0},
    {"g": 0.005, "T": 180.0},
)
VALUE_POINTS = 40001


def value_gap(k_0, g, bench_set, k_thr):
    p = StructuralParameters.create(
        dtype=jnp.float64, **BUTTERFLY_CALIBRATION, g=g, z_0=1.0, k_0=k_0, k_thr=k_thr
    )
    v_low, _ = butterfly_candidate_value(p, bench_set, 1.0, VALUE_POINTS)
    v_high, _ = butterfly_candidate_value(
        p, bench_set, BUTTERFLY_CALIBRATION["b_1"], VALUE_POINTS
    )
    if not (np.isfinite(v_low) or np.isfinite(v_high)):
        raise RuntimeError(f"no candidate solved at k_0={k_0}, g={g}")
    return v_high - v_low, v_low, v_high


def run(output_dir: str = ".results"):
    configure_process()
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal = BUTTERFLY_CALIBRATION
    k_thr_cal = (cal["b_2"] / (cal["b_1"] - 1.0)) ** (1.0 / cal["a"])
    k_thr, k_ss_low, k_ss_high = butterfly_thresholds(
        StructuralParameters.create(
            dtype=jnp.float64, **cal, g=0.0, z_0=1.0, k_0=1.0, k_thr=k_thr_cal
        )
    )

    scenarios = []
    for scenario in SCENARIOS:
        g, T = scenario["g"], scenario["T"]
        bench_set = BenchmarkParameters(T=T, tol=1e-8)
        scan = np.linspace(1.1 * k_ss_low, 0.9 * k_ss_high, 41)
        gaps = [value_gap(float(k_0), g, bench_set, k_thr)[0] for k_0 in scan]
        brackets = [
            (float(scan[i]), float(scan[i + 1]))
            for i in range(len(scan) - 1)
            if gaps[i] < 0.0 <= gaps[i + 1]
        ]
        if len(brackets) != 1:
            raise RuntimeError(f"expected one sign change at g={g}, found {brackets}")
        lo, hi = brackets[0]
        while hi - lo > 1e-8 * hi:
            mid = 0.5 * (lo + hi)
            if value_gap(mid, g, bench_set, k_thr)[0] > 0.0:
                hi = mid
            else:
                lo = mid
        k_dnss = 0.5 * (lo + hi)
        gap, v_low, v_high = value_gap(k_dnss, g, bench_set, k_thr)
        scenarios.append(
            {
                "g": g,
                "T": T,
                "k_dnss": k_dnss,
                "k_dnss_minus_k_thr": k_dnss - k_thr,
                "bisection_width": hi - lo,
                "V_low": v_low,
                "V_high": v_high,
            }
        )
        print(
            f"g={g}: k* = {k_dnss:.8f} (k_thr = {k_thr:.8f}, "
            f"k* - k_thr = {k_dnss - k_thr:+.8f}), V_low = {v_low:.8f}, "
            f"V_high = {v_high:.8f}"
        )

    results = {
        "experiment": "dnss",
        "parameters": {key: cal[key] for key in ("A", "b_1", "b_2")},
        "k_thr": k_thr,
        "k_ss_low": k_ss_low,
        "k_ss_high": k_ss_high,
        "value_points": VALUE_POINTS,
        "scenarios": scenarios,
    }
    (out / f"{PREFIX}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    jsonargparse.CLI(run)
