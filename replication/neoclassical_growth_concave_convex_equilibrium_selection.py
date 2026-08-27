from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import jsonargparse
import numpy as np

from models.kernels import integrated_kernel_matrices, rkhs_norm_squared
from models.neoclassical_growth_benchmark import (
    BenchmarkParameters,
    butterfly_candidate_value,
)
from models.neoclassical_growth_kernel import Theta, neoclassical_growth_kernel
from models.neoclassical_growth_shared import (
    BUTTERFLY_CALIBRATION,
    StructuralParameters,
    butterfly_thresholds,
)
from models.utilities import KernelResidualArgs
from replication.utilities import configure_process, run_solver

PREFIX = "neoclassical_growth_concave_convex_equilibrium_selection"

# Equilibrium-selection diagnostics for the butterfly production model.
# Inside (k_thr, k*) two extremals satisfy the Euler equation, the law of
# motion, and transversality: nonconvex production breaks the welfare
# theorems' sufficiency, so the FOC system's solution set is strictly larger
# than the planner's optimum there. The min-RKHS-norm criterion selects the
# crossing-free k_thr-side extremal (the benchmark) while discounted utility
# selects the crossing one; the scenarios locate the DNSS indifference point
# k* and the selection table quantifies both criteria on the sweep grid
# points in the wedge.

# The DNSS indifference point k* is the root of the value gap
# V_high(k_0) - V_low(k_0) between the two saddle candidates, bisected
# inside the overlap region where both candidates solve (finite gap). The
# g = 0.02 scenario matches the paper's trend calibration; its horizon must
# keep the moving switch k_thr e^{-g t} above the low steady state over
# [0, T] (the switch decays past kappa_ss_low at t ~ 51, so the low basin is
# transient and k* is a truncated-horizon indifference point).
SCENARIOS = (
    {"g": 0.0, "T": 200.0},
    {"g": 0.02, "T": 45.0},
)
VALUE_POINTS = 40001

# The concave-convex sweeps' k_0 grid and data grid; the selection table
# covers the grid points strictly inside (k_thr, k*).
SWEEP_K0_GRID = np.linspace(0.5, 4.0, 70)
SWEEP_DATA_SET = dict(train_T=40.0, train_points=41, test_T=50.0, test_points=41)
WELFARE_T = np.linspace(0.0, 200.0, 4001)


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


# One selection-table row at g = 0: the high solution is the sweep solver's
# own stationary point; the low one re-solves the same NLLS problem from a
# least-squares kernel fit of the benchmark's crossing path on the training
# grid. RKHS norms use the training Gram per alpha block; welfare is the
# discounted-log-utility trapezoid on the kernel consumption path.
def selection_row(k_0, k_ss_low, k_ss_high):
    results, extras = run_solver(
        neoclassical_growth_kernel,
        {**BUTTERFLY_CALIBRATION, "k_0": k_0, "data_set": SWEEP_DATA_SET},
    )
    if not results["solution_accepted"]:
        raise RuntimeError(f"sweep solve rejected at k_0={k_0}")
    solve = extras["solve"]
    solver, residual_fn = solve["solver"], solve["residual_fn"]
    args, p, options = solve["residual_args"], solve["p"], solve["options"]
    theta_high = extras["result"].x
    kernel_set = results["kernel_set"]

    bench_set = BenchmarkParameters(tol=1e-8)
    v_low, path_low = butterfly_candidate_value(p, bench_set, 1.0, VALUE_POINTS)
    v_high, _ = butterfly_candidate_value(p, bench_set, float(p.b_1), VALUE_POINTS)
    if path_low is None:
        raise RuntimeError(f"low crossing candidate failed at k_0={k_0}")

    # g = 0: detrended = level paths. The kernel representation is
    # x(t) = x_0 + K_tilde alpha, so each block's alpha is the least-squares
    # fit of the benchmark path's deviation from its initial condition.
    t_train = args.batch["t"]
    kappa_b, c_tilde_b = path_low(t_train)
    k_target, c_target = np.asarray(kappa_b), np.asarray(c_tilde_b)
    sigma_crra = float(p.sigma_crra)
    c_0 = float(c_target[0])
    mu_0 = c_0**-sigma_crra
    K_tilde = np.asarray(args.K_tilde)

    def fit(target, init):
        return jnp.asarray(np.linalg.lstsq(K_tilde, target - init, rcond=None)[0])

    theta_low_init = Theta(
        alpha_mu=fit(c_target**-sigma_crra, mu_0),
        alpha_c=fit(c_target, c_0),
        alpha_k=fit(k_target, float(p.k_0)),
        c_0=jnp.asarray(c_0),
        mu_0=jnp.asarray(mu_0),
    )
    result_low = solver.solve(theta_low_init, args, p=p, **options)
    theta_low = result_low.x
    values_low = residual_fn(theta_low, args, p)[1]
    max_residual_low = max(
        float(np.abs(values_low["resource"]).max()),
        float(np.abs(values_low["euler"]).max()),
        float(np.abs(values_low["shadow_price"]).max()),
    )
    k_end = float(values_low["k"][-1])
    if abs(k_end - k_ss_high) < abs(k_end - k_ss_low):
        raise RuntimeError(f"low re-solve at k_0={k_0} landed in the high basin")
    if max_residual_low > results["tolerance_set"]["train_residual"]:
        raise RuntimeError(
            f"low re-solve at k_0={k_0}: max residual {max_residual_low:.3e}"
        )

    def welfare(theta):
        points = jnp.asarray(WELFARE_T, dtype=t_train.dtype)
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            t_train,
            kernel_set["kernel"],
            kernel_set["sigma"],
            kernel_set["ell"],
        )
        eval_args = KernelResidualArgs(
            batch={
                "t": points,
                "z_trend": jnp.exp((1.0 - float(p.a)) * float(p.g) * points),
                "g_hat": jnp.asarray(float(p.g), dtype=points.dtype),
            },
            K=K_eval,
            K_tilde=K_tilde_eval,
        )
        c = np.asarray(residual_fn(theta, eval_args, p)[1]["c"])
        return float(
            np.trapezoid(np.exp(-float(p.rho_hat) * WELFARE_T) * np.log(c), WELFARE_T)
        )

    def rkhs_blocks(theta):
        return {
            "rkhs2_k": float(rkhs_norm_squared(theta.alpha_k, args.K)),
            "rkhs2_c": float(rkhs_norm_squared(theta.alpha_c, args.K)),
            "rkhs2_mu": float(rkhs_norm_squared(theta.alpha_mu, args.K)),
        }

    return {
        "k_0": float(k_0),
        "high": {**rkhs_blocks(theta_high), "welfare": welfare(theta_high)},
        "low": {**rkhs_blocks(theta_low), "welfare": welfare(theta_low)},
        "benchmark": {"V_low": v_low, "V_high": v_high},
        "low_solve": {
            "max_residual": max_residual_low,
            "steps": int(result_low.steps),
        },
    }


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
        # Bracket only finite sign changes. With g > 0 the descending switch
        # also admits wait-and-cross extremals (the path drifts near the low
        # branch until the switch falls to it, then crosses); the crossing
        # solver captures that branch only at isolated scan points, so a sign
        # change against a +/- inf neighbor is a branch switch of the solver,
        # not a root of the prompt-crossing value gap.
        brackets = [
            (float(scan[i]), float(scan[i + 1]))
            for i in range(len(scan) - 1)
            if np.isfinite(gaps[i])
            and np.isfinite(gaps[i + 1])
            and gaps[i] < 0.0 <= gaps[i + 1]
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

    k_star = scenarios[0]["k_dnss"]
    selection_table = []
    for k_0 in SWEEP_K0_GRID:
        if not k_thr < float(k_0) < k_star:
            continue
        row = selection_row(float(k_0), k_ss_low, k_ss_high)
        selection_table.append(row)
        print(
            f"k_0={row['k_0']:.4f}: "
            f"rkhs2(high) = {sum(v for k, v in row['high'].items() if k != 'welfare'):.4f}, "
            f"rkhs2(low) = {sum(v for k, v in row['low'].items() if k != 'welfare'):.4f}, "
            f"welfare(high) = {row['high']['welfare']:.6f}, "
            f"welfare(low) = {row['low']['welfare']:.6f}"
        )

    results = {
        "experiment": "equilibrium_selection",
        "parameters": {key: cal[key] for key in ("A", "b_1", "b_2")},
        "k_thr": k_thr,
        "k_ss_low": k_ss_low,
        "k_ss_high": k_ss_high,
        "value_points": VALUE_POINTS,
        "scenarios": scenarios,
        "selection_table": selection_table,
    }
    (out / f"{PREFIX}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    jsonargparse.CLI(run)
