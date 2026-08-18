from __future__ import annotations

import datetime
import json
import math
import os
import pathlib
import platform
import subprocess

import jsonargparse
import pandas as pd

from replication.utilities import DETERMINISTIC_TIMING_REPETITIONS, QUANTILES

# Collect every number embedded in the paper into a single authoritative JSON,
# .results/all_results.json. Selection and rounding to copy/paste form happen
# HERE and only here: the paper must never post-process these values, and a
# number missing from this file is added here rather than computed downstream.
# Keep it selective -- only values the paper text actually cites (parameters,
# architecture/config, and headline error statistics); cross-seed statistics
# are medians (key suffix _median). Runs as the final step of
# generate_all_results.sh and fails loudly on any missing input. Also writes
# the nu/ell robustness table (.results/neoclassical_growth_model_nu_rho.tex).


# Round to n significant digits; None/NaN/inf become None and are pruned.
def sig(x, n=3):
    if x is None or not math.isfinite(x):
        return None
    if x == 0.0:
        return 0
    return round(float(x), n - 1 - math.floor(math.log10(abs(x))))


# Error magnitudes below 0.001 as a LaTeX-ready string, e.g. "1.11 \times 10^{-4}"
# for math mode; larger values stay plain rounded numbers. This is the single
# place that decides how the paper renders scientific notation.
def sci(x, n=3):
    x = sig(x, n)
    if x is None or x == 0 or abs(x) >= 0.001:
        return x
    exp = math.floor(math.log10(abs(x)))
    mantissa = round(x / 10**exp, n - 1)
    if mantissa == int(mantissa):
        mantissa = int(mantissa)
    return f"{mantissa} \\times 10^{{{exp}}}"


# The median of an ensemble stats dict; a column that save_experiment collapsed
# to a constant scalar is already the value.
def med(block):
    return block["median"] if isinstance(block, dict) else block


def final_damping(j):
    return {"final_damping": sig(med(j["final_damping"]), 4)}


def maybe_final_damping(j):
    return final_damping(j) if "final_damping" in j else {}


def final_damping_summary(frame):
    return {
        "final_damping_mean": sig(float(frame["final_damping"].mean()), 4),
        "final_damping_median": sig(float(frame["final_damping"].median()), 4),
    }


def load(out, name):
    # Plain json.load: the ensemble JSONs may contain bare NaN/Infinity.
    return json.loads((out / f"{name}.json").read_text())


# Recursive output cleanup: drop None values (unavailable numbers) and print
# integral floats as integers (steps 20 and k_0 = 1, not 20.0 and 1.0).
def clean(x):
    if isinstance(x, dict):
        d = {k: clean(v) for k, v in x.items()}
        return {k: v for k, v in d.items() if v is not None}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def machine_info():
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if brand:
            info["processor"] = brand
    elif platform.processor():
        info["processor"] = platform.processor()
    try:
        info["ram_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        )
    except (AttributeError, ValueError, OSError):
        pass
    summary = info.get("processor", f"{info['system']} {info['machine']}")
    summary += f" with {info['cpu_count']} CPUs"
    if "ram_gb" in info:
        summary += f" and {info['ram_gb']} GB of RAM"
    info["summary"] = summary
    return info


# --- neoclassical growth (kernel) ------------------------------------------------

# nu / ell robustness table rows: (experiment name, LaTeX nu label, ell).
# The (nu = 1/2, ell = 10) row aliases the baseline experiment.
NU_RHO_ROWS = [
    ("baseline", r"$1/2$", 10),
    ("nu15_ell10", r"$3/2$", 10),
    ("nu25_ell10", r"$5/2$", 10),
    ("nu05_ell2", r"$1/2$", 2),
    ("nu05_ell20", r"$1/2$", 20),
    ("rbf_ell10", r"$\infty$", 10),
    ("rbf_ell2", r"$\infty$", 2),
    ("rbf_ell20", r"$\infty$", 20),
]

NU_RHO_COLUMNS = [
    "test_k_rel_error_max",
    "test_c_rel_error_max",
    "test_k_rel_error_min",
    "test_c_rel_error_min",
]

KERNEL_NU = {
    "matern12": 0.5,
    "matern32": 1.5,
    "matern52": 2.5,
    "rbf": "inf",
}


# One cell of a comparison table: "%.1e" for error magnitudes, "%.2g" for
# seconds, "--" when the metric does not exist for that solver.
def comparison_cell(value, kind="error"):
    if value is None:
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if kind == "time":
        return f"{value:.2g}"
    return f"{value:.1e}"


def accepted_cell(j):
    # Deterministic sweeps label their count n_data_seeds; frac_accepted always
    # accompanies n_accepted, so the total is recoverable when neither is given.
    total = j.get("n_seeds", j.get("n_data_seeds"))
    if total is None:
        total = round(j["n_accepted"] / j["frac_accepted"]) if j["n_accepted"] else 0
    return f"{int(j['n_accepted'])}/{int(total)}"


# Sequential kernel-vs-NN comparison across models. Error columns use each
# model's headline benchmark metric (p for asset pricing; k for growth; the
# max over k/q/c/a-scaled for the open economy); the concave-convex rows
# report the correct-basin fraction over the 70 initial conditions instead,
# since the kinked model has no closed-form benchmark path.
def write_sequential_comparison_table(out):
    def basin_fraction(name):
        frame = pd.read_parquet(out / f"{name}_threshold_results.parquet")
        return f"{int(frame['basin_correct'].sum())}/{len(frame)}"

    # The concave-convex experiment is a sweep over initial conditions, so the
    # acceptance denominator is the sweep size rather than the n_seeds meta.
    def sweep_accepted(name):
        frame = pd.read_parquet(out / f"{name}_threshold_results.parquet")
        return f"{int(frame['solution_accepted'].sum())}/{len(frame)}"

    def oe_max_error(j):
        return max(
            med(j["k_relative_error_max"]),
            med(j["q_relative_error_max"]),
            med(j["c_relative_error_max"]),
            med(j["a_scaled_error_max"]),
        )

    ap_kernel = load(out, "asset_pricing_kernel_baseline")
    ap_nn = load(out, "asset_pricing_nn_baseline")
    ngm_kernel = load(out, "neoclassical_growth_kernel_baseline")
    ngm_nn = load(out, "neoclassical_growth_nn_baseline")
    cc_kernel = load(out, "neoclassical_growth_concave_convex_kernel_threshold")
    cc_nn = load(out, "neoclassical_growth_concave_convex_nn_threshold")
    oe_kernel = load(out, "open_economy_growth_kernel_baseline")
    oe_nn = load(out, "open_economy_growth_nn_baseline")

    rows = [
        (
            "Asset pricing",
            "Kernel",
            med(ap_kernel["p_rel_error_mean"]),
            med(ap_kernel["p_rel_error_max"]),
            None,
            med(ap_kernel["solve_time"]),
            accepted_cell(ap_kernel),
        ),
        (
            "",
            "NN",
            med(ap_nn["p_rel_error_mean"]),
            med(ap_nn["p_rel_error_max"]),
            None,
            med(ap_nn["solve_time"]),
            accepted_cell(ap_nn),
        ),
        (
            "Neoclassical growth",
            "Kernel",
            med(ngm_kernel["test_k_rel_error_mean"]),
            med(ngm_kernel["test_k_rel_error_max"]),
            None,
            med(ngm_kernel["solve_time"]),
            accepted_cell(ngm_kernel),
        ),
        (
            "",
            "NN",
            med(ngm_nn["test_k_rel_error_mean"]),
            med(ngm_nn["test_k_rel_error_max"]),
            None,
            med(ngm_nn["solve_time_success"]),
            accepted_cell(ngm_nn),
        ),
        (
            "Concave-convex",
            "Kernel",
            None,
            None,
            basin_fraction("neoclassical_growth_concave_convex_kernel"),
            med(cc_kernel["solve_time"]),
            sweep_accepted("neoclassical_growth_concave_convex_kernel"),
        ),
        (
            "",
            "NN",
            None,
            None,
            basin_fraction("neoclassical_growth_concave_convex_nn"),
            med(cc_nn["solve_time"]),
            sweep_accepted("neoclassical_growth_concave_convex_nn"),
        ),
        (
            "Open economy",
            "Kernel",
            None,
            oe_max_error(oe_kernel),
            None,
            med(oe_kernel["solve_time"]),
            accepted_cell(oe_kernel),
        ),
        (
            "",
            "NN",
            None,
            oe_max_error(oe_nn),
            None,
            med(oe_nn["solve_time_success"]),
            accepted_cell(oe_nn),
        ),
    ]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Solver & Mean rel.\ error & Max rel.\ error & "
        r"Correct basin & Time (s) & Accepted \\",
        r"\midrule",
    ]
    for model, solver, mean_err, max_err, basin, seconds, accepted in rows:
        lines.append(
            f"{model} & {solver} & {comparison_cell(mean_err)} & "
            f"{comparison_cell(max_err)} & {basin if basin is not None else '--'} & "
            f"{comparison_cell(seconds, 'time')} & {accepted} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = out / "sequential_solver_comparison.tex"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# Neoclassical growth across all four solvers: the sequential and recursive
# formulations share the calibration exactly (AFVRR constants, g = 0,
# z_0 = k_0 = 1), so accuracy against the shared BVP benchmark and fit time
# are directly comparable.
def write_neoclassical_growth_comparison_table(out):
    kernel = load(out, "neoclassical_growth_kernel_baseline")
    nn = load(out, "neoclassical_growth_nn_baseline")
    kernel_recursive = load(out, "neoclassical_growth_kernel_recursive_single_ic_g0")
    nn_recursive = load(out, "neoclassical_growth_nn_recursive_single_ic_g0")

    rows = [
        (
            "Sequential kernel",
            med(kernel["test_k_rel_error_mean"]),
            med(kernel["test_k_rel_error_max"]),
            med(kernel["test_c_rel_error_mean"]),
            med(kernel["solve_time"]),
            accepted_cell(kernel),
        ),
        (
            "Sequential NN",
            med(nn["test_k_rel_error_mean"]),
            med(nn["test_k_rel_error_max"]),
            med(nn["test_c_rel_error_mean"]),
            med(nn["solve_time_success"]),
            accepted_cell(nn),
        ),
        (
            "Recursive kernel",
            med(kernel_recursive["test_mean_abs_k_rel_error"]),
            med(kernel_recursive["test_max_abs_k_rel_error"]),
            med(kernel_recursive["test_mean_abs_c_rel_error"]),
            med(kernel_recursive["solve_time"]),
            accepted_cell(kernel_recursive),
        ),
        (
            "Recursive NN",
            med(nn_recursive["test_mean_abs_k_rel_error"]),
            med(nn_recursive["test_max_abs_k_rel_error"]),
            med(nn_recursive["test_mean_abs_c_rel_error"]),
            med(nn_recursive["solve_time_success"]),
            accepted_cell(nn_recursive),
        ),
    ]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Solver & Mean rel.\ error: $k$ & Max rel.\ error: $k$ & "
        r"Mean rel.\ error: $c$ & Time (s) & Accepted \\",
        r"\midrule",
    ]
    for solver, mean_k, max_k, mean_c, seconds, accepted in rows:
        lines.append(
            f"{solver} & {comparison_cell(mean_k)} & {comparison_cell(max_k)} & "
            f"{comparison_cell(mean_c)} & {comparison_cell(seconds, 'time')} & "
            f"{accepted} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = out / "neoclassical_growth_solver_comparison.tex"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def write_nu_rho_table(out):
    # Same layout as the original pandas to_latex output (booktabs, %.1e cells);
    # a rejected solve (ill-conditioned RBF Gram) renders as "--".
    header = (
        r"$\nu$ & $\ell$ & Max of Rel. Error: $\hat{x}(t)$ & "
        r"Max of Rel. Error: $\hat{y}(t)$ & Min of Rel. Error: $\hat{x}(t)$ & "
        r"Min of Rel. Error: $\hat{y}(t)$ \\"
    )
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", header, r"\midrule"]
    for name, nu_label, ell in NU_RHO_ROWS:
        j = load(out, f"neoclassical_growth_kernel_{name}")
        accepted = j["n_accepted"] == 1
        if accepted:
            cells = [f"{med(j[column]):.1e}" for column in NU_RHO_COLUMNS]
        else:
            cells = ["--"] * len(NU_RHO_COLUMNS)
        lines.append(f"{nu_label} & {ell} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = out / "neoclassical_growth_model_nu_rho.tex"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# --- sensitivity appendix ---------------------------------------------------

# (results-field name, LaTeX row label in the paper's notation: capital x,
# consumption y, discount rate r, CRRA gamma, initial condition x_0);
# sigma_crra and k_0 have exactly zero steady-state derivatives, so their
# rows certify the invariance rather than a magnitude.
NEOCLASSICAL_SENSITIVITY_ROWS = (
    ("a", r"$a$"),
    ("delta", r"$\delta$"),
    ("rho_hat", r"$r$"),
    ("sigma_crra", r"$\gamma$"),
    ("k_0", r"$x_0$"),
)
MULTICOUNTRY_SENSITIVITY_TIMES = (0, 10, 30, 80)
MULTICOUNTRY_SENSITIVITY_TABLE_TIMES = (30, 80)
MULTICOUNTRY_SENSITIVITY_OBSERVABLES = ("rate", "c_1", "c_2", "k_1", "k_2")
# The ten-country table: no cross-section that wide has a readable
# per-country column, so its observables are world statistics and the one
# reported direction is (1, ..., 1) in gamma -- the representative-agent
# risk-aversion perturbation, which the two-country table reports at the same
# dates. Agreement between the two is the cross-check the pair exists for.
MULTICOUNTRY_AGGREGATE_ROWS = (
    ("rate", r"World interest rate $r^{\star}$"),
    ("saving_rate", r"World saving rate $X/Y$"),
    ("nfa_dispersion", r"NFA dispersion"),
    ("output_gini", r"Output Gini"),
)
# Labels in the supplement's multicountry notation (k_{i,0} and b_{i,0}
# initial conditions, capital share a); a_0_transfer is the
# clearing-preserving bond transfer tau, and z_i scales country i's
# technology (one in the baseline).
MULTICOUNTRY_SENSITIVITY_ROWS = (
    ("k_0_1", r"$k_{1,0}$"),
    ("k_0_2", r"$k_{2,0}$"),
    ("a_0_transfer", r"$\tau$"),
    ("gamma_1", r"$\gamma_1$"),
    ("gamma_2", r"$\gamma_2$"),
    ("alpha", r"$a$"),
    ("delta", r"$\delta$"),
    ("r", r"$r$"),
    ("eta", r"$\eta$"),
    ("productivity_1", r"$z_1$"),
    ("productivity_2", r"$z_2$"),
)


# Field accessors shared by the sensitivity tables and payloads: a collapsed
# ensemble JSON for deterministic runs, accepted-seed parquet medians for the
# NN ensembles (matching each family's existing convention).
def json_values(j):
    return lambda field: med(j[field])


def accepted_values(out, stem):
    frame = pd.read_parquet(out / f"{stem}_results.parquet")
    accepted = frame[frame["solution_accepted"]]
    if accepted.empty:
        raise RuntimeError(f"{stem} has no accepted seeds")
    return lambda field: float(accepted[field].median())


def neoclassical_sensitivity_payload(value, reference=None):
    derivatives = {}
    closed_form_gaps = []
    invariant_magnitudes = []
    cross_solver_gaps = []
    for name, _ in NEOCLASSICAL_SENSITIVITY_ROWS:
        d_k = value(f"d_k_T120_d_{name}")
        d_c = value(f"d_c_T120_d_{name}")
        d_k_ss = value(f"d_k_ss_d_{name}")
        d_c_ss = value(f"d_c_ss_d_{name}")
        derivatives[name] = {
            "d_k_T": sig(d_k, 6),
            "d_c_T": sig(d_c, 6),
            "d_k_ss": sig(d_k_ss, 6),
            "d_c_ss": sig(d_c_ss, 6),
        }
        if name in ("sigma_crra", "k_0"):
            invariant_magnitudes += [abs(d_k), abs(d_c)]
        else:
            closed_form_gaps += [abs(d_k - d_k_ss), abs(d_c - d_c_ss)]
        if reference is not None:
            cross_solver_gaps += [
                abs(d_k - reference(f"d_k_T120_d_{name}")),
                abs(d_c - reference(f"d_c_T120_d_{name}")),
            ]
    payload = {
        "sensitivity_time": 120,
        "train_T": value("train_T"),
        "num_train_points": int(value("num_train_points")),
        "k_T": sig(value("k_T120"), 6),
        "c_T": sig(value("c_T120"), 6),
        "k_ss": sig(value("k_ss"), 6),
        "c_ss": sig(value("c_ss"), 6),
        "derivatives": derivatives,
        # (a, delta, rho_hat) rows against the exact steady-state
        # derivatives; the residual is the finite-T transition remnant plus
        # solver error.
        "max_abs_gap_vs_closed_form": sci(max(closed_form_gaps)),
        # (sigma_crra, k_0) rows are exactly zero in the closed form.
        "max_abs_invariant_derivative": sci(max(invariant_magnitudes)),
    }
    if reference is not None:
        payload["max_abs_gap_vs_kernel"] = sci(max(cross_solver_gaps))
    return payload


def multicountry_sensitivity_payload(value, reference=None):
    levels = {}
    derivatives = {}
    cross_solver_gaps = []
    for time in MULTICOUNTRY_SENSITIVITY_TIMES:
        for observable in MULTICOUNTRY_SENSITIVITY_OBSERVABLES:
            levels[f"{observable}_T{time}"] = sig(value(f"{observable}_T{time}"), 6)
        derivatives[f"T{time}"] = {}
        for suffix, _ in MULTICOUNTRY_SENSITIVITY_ROWS:
            derivatives[f"T{time}"][suffix] = {}
            for observable in MULTICOUNTRY_SENSITIVITY_OBSERVABLES:
                field = f"d_{observable}_T{time}_d_{suffix}"
                derivative = value(field)
                derivatives[f"T{time}"][suffix][observable] = sig(derivative, 6)
                if reference is not None:
                    cross_solver_gaps.append(abs(derivative - reference(field)))
    payload = {
        "times": list(MULTICOUNTRY_SENSITIVITY_TIMES),
        "levels": levels,
        "derivatives": derivatives,
    }
    if reference is not None:
        payload["max_abs_gap_vs_kernel"] = sig(max(cross_solver_gaps), 3)
    return payload


# The ten-country aggregate comparative statics, with the two numbers the
# table exists to certify: the redistribution-neutrality residual, and the
# gap against the two-country table on the one perturbation both report.
# Moving every gamma_i together is the same representative-agent change at
# any N, so the two must agree despite solving systems of different size.
def multicountry_aggregate_payload(value, reference=None):
    def percent_per_gamma(accessor, name, time):
        return (
            100.0
            * accessor(f"d_{name}_T{time}_d_gamma_common")
            / accessor(f"{name}_T{time}")
        )

    statistics = {}
    neutrality = {}
    rate_gaps = []
    for time in MULTICOUNTRY_SENSITIVITY_TIMES:
        statistics[f"T{time}"] = {
            name: {
                "level": sig(value(f"{name}_T{time}"), 6),
                "d_d_gamma_common": sig(value(f"d_{name}_T{time}_d_gamma_common"), 6),
                "percent_per_gamma": sig(percent_per_gamma(value, name, time), 3),
            }
            for name, _ in MULTICOUNTRY_AGGREGATE_ROWS
        }
        neutrality[f"T{time}"] = sci(value(f"rate_T{time}_a_0_neutrality"))
        if reference is not None:
            rate_gaps.append(
                abs(
                    percent_per_gamma(value, "rate", time)
                    - percent_per_gamma(reference, "rate", time)
                )
            )
    payload = {
        "times": list(MULTICOUNTRY_SENSITIVITY_TIMES),
        "statistics": statistics,
        "rate_redistribution_neutrality": neutrality,
    }
    if reference is not None:
        payload["max_rate_percent_gap_vs_two_countries"] = sig(max(rate_gaps), 3)
    return payload


def write_neoclassical_sensitivity_table(out):
    kernel_value = json_values(load(out, "neoclassical_growth_kernel_sensitivity"))
    nn_value = accepted_values(out, "neoclassical_growth_nn_sensitivity")

    def cell(value):
        return f"${sci(value, 4)}$"

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{$\partial x(T)/\partial \theta$} & "
        r"\multicolumn{3}{c}{$\partial y(T)/\partial \theta$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"$\theta$ & Kernel & NN & Closed Form & Kernel & NN & Closed Form \\",
        r"\midrule",
    ]
    for name, label in NEOCLASSICAL_SENSITIVITY_ROWS:
        cells = [
            cell(kernel_value(f"d_k_T120_d_{name}")),
            cell(nn_value(f"d_k_T120_d_{name}")),
            cell(kernel_value(f"d_k_ss_d_{name}")),
            cell(kernel_value(f"d_c_T120_d_{name}")),
            cell(nn_value(f"d_c_T120_d_{name}")),
            cell(kernel_value(f"d_c_ss_d_{name}")),
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = out / "neoclassical_growth_sensitivity.tex"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def write_multicountry_sensitivity_table(out, source, name):
    value = json_values(load(out, source))

    def cell(x):
        return f"${sci(x, 3)}$"

    observable_header = (
        r"$\theta$ & $r^{\star}(T)$ & $y_{1,1}(T)$ & $y_{1,2}(T)$ & "
        r"$x_{1,1}(T)$ & $x_{1,2}(T)$ \\"
    )
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule"]
    for time in MULTICOUNTRY_SENSITIVITY_TABLE_TIMES:
        lines += [
            rf"\multicolumn{{6}}{{c}}{{$T = {time}$}} \\",
            r"\midrule",
            observable_header,
            r"\midrule",
        ]
        for suffix, label in MULTICOUNTRY_SENSITIVITY_ROWS:
            cells = [
                cell(value(f"d_{observable}_T{time}_d_{suffix}"))
                for observable in MULTICOUNTRY_SENSITIVITY_OBSERVABLES
            ]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if time != MULTICOUNTRY_SENSITIVITY_TABLE_TIMES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = out / name
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def write_multicountry_aggregate_table(out, source, name):
    value = json_values(load(out, source))
    times = MULTICOUNTRY_SENSITIVITY_TIMES
    span = len(times) + 1

    # Percent change per unit of gamma. NFA dispersion and the output Gini
    # are functions of the predetermined k_0 and b_0 at T = 0, so no
    # parameter can move them there; the derivative is zero to rounding and
    # the cell says so rather than reporting a rounded-down small number.
    def semi_elasticity(name, time):
        response = (
            100.0 * value(f"d_{name}_T{time}_d_gamma_common") / value(f"{name}_T{time}")
        )
        return "$0$" if abs(response) < 1e-6 else f"${response:.2f}$"

    def block(title, cells_for):
        rows = [rf"\multicolumn{{{span}}}{{c}}{{{title}}} \\", r"\midrule"]
        for name, label in MULTICOUNTRY_AGGREGATE_ROWS:
            rows.append(
                f"{label} & " + " & ".join(cells_for(name, t) for t in times) + r" \\"
            )
        return rows

    lines = [
        rf"\begin{{tabular}}{{l{'r' * len(times)}}}",
        r"\toprule",
        " & " + " & ".join(rf"$T = {time}$" for time in times) + r" \\",
        r"\midrule",
        *block("Level", lambda name, t: f"${sci(value(f'{name}_T{t}'), 3)}$"),
        r"\midrule",
        *block(
            r"Common risk aversion: $100\,\partial \ln(\cdot)/\partial\gamma$ "
            r"(\% per unit $\gamma$)",
            semi_elasticity,
        ),
        r"\midrule",
        rf"\multicolumn{{{span}}}{{c}}{{Redistribution neutrality}} \\",
        r"\midrule",
        r"$\max_i |\partial r^{\star}/\partial b_{i,0}|$ & "
        + " & ".join(
            f"${sci(value(f'rate_T{time}_a_0_neutrality'), 2)}$" for time in times
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    path = out / name
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def neoclassical_growth_kernel(out):
    prefix = "neoclassical_growth_kernel"
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    baseline = load(out, f"{prefix}_baseline")

    def errors(j):
        return {
            "solution_accepted": bool(j["n_accepted"] == 1),
            "test_k_rel_error_mean": sci(med(j["test_k_rel_error_mean"])),
            "test_c_rel_error_mean": sci(med(j["test_c_rel_error_mean"])),
            "test_k_rel_error_max": sci(med(j["test_k_rel_error_max"])),
            "test_c_rel_error_max": sci(med(j["test_c_rel_error_max"])),
            "test_k_rel_error_min": sci(med(j["test_k_rel_error_min"])),
            "test_c_rel_error_min": sci(med(j["test_c_rel_error_min"])),
            "max_train_residual": sci(med(j["max_train_residual"]), 2),
            "solve_time": sig(med(j["solve_time"]), 2),
            "timing_repetitions": int(med(j["timing_repetitions"])),
            **final_damping(j),
        }

    sparse = load(out, f"{prefix}_sparse")
    far = load(out, f"{prefix}_far_steady_state")
    experiments = {
        "baseline": {
            "parameters": {
                "a": sig(med(baseline["a"]), 6),
                "delta": med(baseline["delta"]),
                "rho_hat": med(baseline["rho_hat"]),
                "k_0": med(baseline["k_0"]),
            },
            "data": {
                "num_train_points": int(med(baseline["num_train_points"])),
                "train_T": med(baseline["train_T"]),
                "test_T": med(baseline["test_T"]),
                "test_points": int(med(baseline["test_points"])),
            },
            **errors(baseline),
        },
        "sparse": {
            "train_points_list": hp_sparse_points(out, prefix),
            "num_train_points": int(med(sparse["num_train_points"])),
            **errors(sparse),
        },
        "far_steady_state": {
            "train_T": med(far["train_T"]),
            "num_train_points": int(med(far["num_train_points"])),
            "test_T": med(far["test_T"]),
            **errors(far),
        },
    }

    # Robustness table (the nu = 1/2, ell = 10 row aliases baseline).
    robustness_table = {}
    for name, _, ell in NU_RHO_ROWS:
        j = load(out, f"{prefix}_{name}")
        experiment_hp = load(out, f"{prefix}_{name}_hyperparameters")
        kernel_name = experiment_hp["kernel_set"]["kernel"]
        robustness_table[name if name != "baseline" else "nu05_ell10"] = {
            "kernel": kernel_name,
            "nu": KERNEL_NU[kernel_name],
            "ell": ell,
            **errors(j),
        }

    baseline_kernel = hp["kernel_set"]["kernel"]
    payload = {
        "kernel": {
            "name": baseline_kernel,
            "nu": KERNEL_NU[baseline_kernel],
            "ell": hp["kernel_set"]["ell"],
            "sigma": hp["kernel_set"]["sigma"],
        },
        "experiments": experiments,
        "robustness_table": robustness_table,
    }
    # The sensitivity experiment exists only in the float64 pass.
    if (out / f"{prefix}_sensitivity.json").exists():
        sensitivity = load(out, f"{prefix}_sensitivity")
        if sensitivity["frac_accepted"] != 1.0:
            raise RuntimeError("the neoclassical kernel sensitivity run was rejected")
        payload["sensitivity"] = neoclassical_sensitivity_payload(
            json_values(sensitivity)
        )
    return payload


def neoclassical_growth_kernel_uno(out):
    prefix = "neoclassical_growth_kernel_uno"
    hp = load(out, f"{prefix}_baseline_hyperparameters")

    def errors(j):
        return {
            "solution_accepted": bool(j["n_accepted"] == 1),
            "test_k_rel_error_mean": sci(med(j["test_k_rel_error_mean"])),
            "test_c_rel_error_mean": sci(med(j["test_c_rel_error_mean"])),
            "test_k_rel_error_max": sci(med(j["test_k_rel_error_max"])),
            "test_c_rel_error_max": sci(med(j["test_c_rel_error_max"])),
            "max_train_residual": sci(med(j["max_train_residual"]), 2),
            "solve_time": sig(med(j["solve_time"]), 2),
            "timing_repetitions": int(med(j["timing_repetitions"])),
        }

    experiments = {}
    for name in ("baseline", "sparse", "far_steady_state"):
        result = load(out, f"{prefix}_{name}")
        experiment = errors(result)
        if name == "sparse":
            experiment["train_points_list"] = hp_sparse_points(out, prefix)
        experiments[name] = experiment

    robustness_table = {}
    for name, _, ell in NU_RHO_ROWS:
        result = load(out, f"{prefix}_{name}")
        experiment_hp = load(out, f"{prefix}_{name}_hyperparameters")
        kernel_name = experiment_hp["kernel_set"]["kernel"]
        robustness_table[name if name != "baseline" else "nu05_ell10"] = {
            "kernel": kernel_name,
            "nu": KERNEL_NU[kernel_name],
            "ell": ell,
            **errors(result),
        }

    baseline_kernel = hp["kernel_set"]["kernel"]
    return {
        "kernel": {
            "name": baseline_kernel,
            "nu": KERNEL_NU[baseline_kernel],
            "ell": hp["kernel_set"]["ell"],
            "sigma": hp["kernel_set"]["sigma"],
        },
        "experiments": experiments,
        "robustness_table": robustness_table,
    }


def hp_sparse_points(out, prefix):
    hp = load(out, f"{prefix}_sparse_hyperparameters")
    return hp["data_set"]["train_points_list"]


def neoclassical_growth_concave_convex_kernel(
    out,
    prefix="neoclassical_growth_concave_convex_kernel",
    require_all=True,
):
    j = load(out, f"{prefix}_threshold")
    hp = load(out, f"{prefix}_threshold_hyperparameters")
    df = pd.read_parquet(out / f"{prefix}_threshold_results.parquet")
    n_total = int(len(df))
    n_accepted = int(df["solution_accepted"].sum())
    if require_all and n_accepted != n_total:
        # The QE caption states every initial condition converges.
        rejected = df.loc[~df["solution_accepted"], "k_0"].tolist()
        raise RuntimeError(f"rejected threshold solves at k_0={rejected}")
    # Empirical basin threshold from the side each path ACTUALLY settles on
    # (basin_correct = analytic side, false_positive = opposite side), so a
    # wrong-basin path moves the bracket instead of being silently dropped.
    above_thr = df["k_0"] > df["k_thr"]
    ends_high = (df["basin_correct"] & above_thr) | (df["false_positive"] & ~above_thr)
    goes_low = (df["basin_correct"] & ~above_thr) | (df["false_positive"] & above_thr)
    resolved = ends_high | goes_low
    if require_all and not resolved.all():
        unresolved = df.loc[~(ends_high | goes_low), "k_0"].tolist()
        raise RuntimeError(f"unclassified threshold paths at k_0={unresolved}")
    k_0_low_max = float(df.loc[goes_low & resolved, "k_0"].max())
    k_0_high_min = float(df.loc[ends_high & resolved, "k_0"].min())
    threshold = {
        "k_0_grid": {
            "min": sig(df["k_0"].min()),
            "max": sig(df["k_0"].max()),
            "n": n_total,
        },
        "n_accepted": n_accepted,
        "n_basin_correct": int(df["basin_correct"].sum()),
        "n_unresolved": int((~resolved).sum()),
        "k_thr": sig(float(df["k_thr"].iloc[0]), 6),
        "k_ss_low": sig(float(df["k_ss_low"].iloc[0]), 4),
        "k_ss_high": sig(float(df["k_ss_high"].iloc[0]), 4),
        # Bracketing grid points around the empirical basin split.
        "k_0_threshold_bracket": [sig(k_0_low_max, 4), sig(k_0_high_min, 4)],
        "k_0_threshold": sig(0.5 * (k_0_low_max + k_0_high_min), 4),
    }
    if "final_damping" in df:
        threshold.update(final_damping_summary(df.loc[df["solution_accepted"]]))
    return {
        "parameters": {
            "A": hp["parameters"]["A"],
            "b_1": hp["parameters"]["b_1"],
            "b_2": hp["parameters"]["b_2"],
        },
        "threshold": threshold,
        "n_seeds": int(j["n_seeds"]),
    }


def asset_pricing_kernel(out, prefix="asset_pricing_kernel"):
    j = load(out, f"{prefix}_baseline")
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    r, c, g = (hp["parameters"][key] for key in ("r", "c", "g"))
    x_0 = hp["initial_conditions"]["x_0"]
    return {
        "experiments": {
            "baseline": {
                "parameters": {"r": r, "c": c, "g": g, "x_0": x_0},
                "data": {
                    "num_train_points": int(med(j["num_train_points"])),
                    "train_T": med(j["train_T"]),
                    "test_T": med(j["test_T"]),
                },
                "solution_accepted": bool(j["n_accepted"] == 1),
                "p_0": sig(med(j["p_0"]), 4),
                "p_rel_error_mean": sci(med(j["p_rel_error_mean"])),
                "p_rel_error_max": sci(med(j["p_rel_error_max"])),
                "p_rel_error_min": sci(med(j["p_rel_error_min"])),
                "max_train_residual": sci(med(j["max_train_residual"]), 2),
                "solve_time": sig(med(j["solve_time"]), 2),
                "timing_repetitions": int(med(j["timing_repetitions"])),
                **maybe_final_damping(j),
            }
        },
        # Corrected no-bubble fundamental (QE eq. 19/20 replacement):
        #   mu_f(t) = (x_0 + c/g) e^{g t} / (r - g) - c / (g r),
        # general solution mu = mu_f + zeta e^{r t}, zeta = 0 by no-bubble.
        "benchmark": {
            "mu_f_0": sig((x_0 + c / g) / (r - g) - c / (g * r), 4),
            "r_minus_g": sig(r - g, 4),
        },
    }


def asset_pricing_nn(out):
    prefix = "asset_pricing_nn"
    j = load(out, f"{prefix}_baseline")
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    frame = pd.read_parquet(out / f"{prefix}_baseline_results.parquet")
    accepted = frame[frame["solution_accepted"]]
    if accepted.empty:
        raise RuntimeError("asset-pricing NN has no accepted seeds")
    return {
        "parameters": hp["parameters"],
        "data": hp["data_set"],
        "architecture": {
            "width": hp["nn_set"]["width"],
            "depth": hp["nn_set"]["depth"],
            "hidden_activation": hp["nn_set"]["hidden_activation"],
            "total_params": int(accepted["total_params"].median()),
        },
        "n_seeds": int(j["n_seeds"]),
        "frac_accepted": sig(j["frac_accepted"]),
        "selection": "minimum_identity_metric_parameter_step",
        "training_information": {
            "terminal_residual_count": 0,
            "tvc_residual_count": 0,
            "benchmark_residual_count": 0,
        },
        "experiments": {
            "baseline": {
                "p_0_median": sig(float(accepted["p_0"].median()), 6),
                "p_rel_error_mean_median": sci(
                    float(accepted["p_rel_error_mean"].median())
                ),
                "p_rel_error_max_median": sci(
                    float(accepted["p_rel_error_max"].median())
                ),
                "max_abs_bubble_coefficient_median": sci(
                    float(accepted["max_abs_bubble_coefficient"].median())
                ),
                "max_train_residual_median": sci(
                    float(accepted["max_train_residual"].median())
                ),
                "solve_time_median": sig(float(accepted["solve_time"].median())),
                **final_damping_summary(accepted),
            }
        },
    }


def optimal_advertising_kernel(out):
    prefix = "optimal_advertising_kernel"
    j = load(out, f"{prefix}_baseline")
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    kernel_name = hp["kernel_set"]["kernel"]
    return {
        "parameters": {
            "rho_hat": hp["parameters"]["rho_hat"],
            "c": hp["parameters"]["c"],
            "beta": hp["parameters"]["beta"],
            "kappa": hp["parameters"]["kappa"],
            "gamma": sig(hp["parameters"]["gamma"], 4),
            "x_0": hp["initial_conditions"]["x_0"],
        },
        # ell differs from the headline ell = 10 used elsewhere.
        "kernel": {
            "name": kernel_name,
            "nu": KERNEL_NU[kernel_name],
            "ell": hp["kernel_set"]["ell"],
            "sigma": hp["kernel_set"]["sigma"],
        },
        "solution_accepted": bool(j["n_accepted"] == 1),
        "mu_0": sig(med(j["mu_0"]), 4),
        "u_0": sig(med(j["u_0"]), 4),
        "max_train_residual": sci(med(j["max_train_residual"]), 2),
        "solve_time": sig(med(j["solve_time"]), 2),
        "timing_repetitions": int(med(j["timing_repetitions"])),
        **final_damping(j),
    }


def human_capital_kernel(out):
    prefix = "human_capital_kernel"
    j = load(out, f"{prefix}_baseline")
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    if j["n_accepted"] != 1:
        raise RuntimeError("human capital baseline rejected")
    return {
        "parameters": {
            "a_k": sig(hp["parameters"]["a_k"], 6),
            "a_h": hp["parameters"]["a_h"],
            "delta_k": hp["parameters"]["delta_k"],
            "delta_h": hp["parameters"]["delta_h"],
            "rho_hat": hp["parameters"]["rho_hat"],
            "k_0": hp["initial_conditions"]["k_0"],
        },
        "data": {
            "num_train_points": int(med(j["num_train_points"])),
            "train_T": med(j["train_T"]),
            "test_T": med(j["test_T"]),
        },
        "solution_accepted": True,
        # QE prose cites x_h0; solved from the initial-condition system.
        "h_0": sig(med(j["h_0"]), 6),
        "c_0": sig(med(j["c_0"]), 4),
        "max_train_residual": sci(med(j["max_train_residual"]), 2),
        "max_validation_residual": sci(med(j["max_validation_residual"]), 2),
        "max_test_feasibility": sci(med(j["max_test_feasibility"]), 2),
        "solve_time": sig(med(j["solve_time"]), 2),
        "timing_repetitions": int(med(j["timing_repetitions"])),
        **final_damping(j),
    }


def neoclassical_growth_nn(out):
    prefix = "neoclassical_growth_nn"

    # Medians over ACCEPTED seeds only, matching the recursive NN contract.
    baseline_json = load(out, f"{prefix}_baseline")
    baseline_df = pd.read_parquet(out / f"{prefix}_baseline_results.parquet")
    accepted = baseline_df[baseline_df["solution_accepted"]]
    if accepted.empty:
        raise RuntimeError("sequence-space growth NN has no accepted seeds")
    hp = load(out, f"{prefix}_baseline_hyperparameters")

    payload = {
        "parameters": hp["parameters"],
        "initial_conditions": hp["initial_conditions"],
        "data": hp["data_set"],
        "architecture": {
            "width": hp["nn_set"]["width"],
            "depth": hp["nn_set"]["depth"],
            "hidden_activation": hp["nn_set"]["hidden_activation"],
            "total_params": int(med(baseline_json["total_params"])),
        },
        "stopping": {
            "train_loss_threshold": hp["stop_set"]["train_loss_threshold"],
            "max_steps": hp["stop_set"]["max_steps"],
            "num_attempts": hp["stop_set"]["num_attempts"],
            "log_tvc_threshold": hp["stop_set"]["log_tvc_threshold"],
        },
        "n_seeds": int(baseline_json["n_seeds"]),
        "experiments": {
            "baseline": {
                "frac_accepted": sig(baseline_json["frac_accepted"]),
                "test_k_rel_error_mean_median": sci(
                    float(accepted["test_k_rel_error_mean"].median())
                ),
                "test_c_rel_error_mean_median": sci(
                    float(accepted["test_c_rel_error_mean"].median())
                ),
                "test_resource_mean_abs_median": sci(
                    float(accepted["test_resource_mean_abs"].median())
                ),
                "test_euler_mean_abs_median": sci(
                    float(accepted["test_euler_mean_abs"].median())
                ),
                "max_train_residual_median": sci(
                    float(accepted["max_train_residual"].median())
                ),
                # Post-hoc saddle check at test_tvc_T: negative and O(1)-scaled
                # away from zero is the no-bubble side.
                "log_tvc_residual_median": sig(
                    float(accepted["log_tvc_residual"].median()), 4
                ),
                "frac_transversality_ok": sig(
                    float(accepted["transversality_ok"].mean())
                ),
                "solve_time_median": sig(float(accepted["solve_time"].median())),
                "steps_median": sig(float(accepted["steps"].median())),
                **final_damping_summary(accepted),
                "num_train_points": int(med(baseline_json["num_train_points"])),
            }
        },
    }
    # The sensitivity experiment exists only in the float64 pass; the kernel
    # sensitivity is its cross-solver reference.
    if (out / f"{prefix}_sensitivity.json").exists():
        payload["sensitivity"] = neoclassical_sensitivity_payload(
            accepted_values(out, f"{prefix}_sensitivity"),
            reference=json_values(load(out, "neoclassical_growth_kernel_sensitivity")),
        )
    return payload


RECURSIVE_EXPERIMENTS = ("single_ic_g0", "multi_ic_g0", "multi_ic_g02")


def neoclassical_growth_nn_recursive(out):
    prefix = "neoclassical_growth_nn_recursive"
    hp = load(out, f"{prefix}_single_ic_g0_hyperparameters")
    single_json = load(out, f"{prefix}_single_ic_g0")

    # Medians over ACCEPTED seeds only (a rejected attempt burns the full
    # training budget, so all-seed medians would mix in outliers). Acceptance
    # is solver convergence; Euler, path, transversality, and benchmark errors
    # are reported diagnostics, never gates.
    def experiment(name):
        j = load(out, f"{prefix}_{name}")
        df = pd.read_parquet(out / f"{prefix}_{name}_results.parquet")
        accepted = df[df["solution_accepted"]]
        sample = accepted if not accepted.empty else df
        return {
            "frac_accepted": sig(j["frac_accepted"]),
            "test_mean_abs_k_rel_error_median": sci(
                float(sample["test_mean_abs_k_rel_error"].median())
            ),
            "test_mean_abs_c_rel_error_median": sci(
                float(sample["test_mean_abs_c_rel_error"].median())
            ),
            "test_euler_mean_abs_median": sci(
                float(sample["test_euler_mean_abs"].median())
            ),
            "max_train_euler_median": sci(float(sample["max_train_euler"].median())),
            "max_test_euler_median": sci(float(sample["max_test_euler"].median())),
            "max_log_tvc_residual_median": sig(
                float(sample["max_log_tvc_residual"].median()), 4
            ),
            "solve_time_median": sig(float(sample["solve_time"].median())),
            "steps_median": sig(float(sample["steps"].median())),
            **final_damping_summary(sample),
            "num_collocation_states": int(med(j["num_collocation_states"])),
            "ode_solver": str(sample["ode_solver"].iloc[0]),
            "num_train_trajectories": int(med(j["num_train_trajectories"])),
            "num_test_trajectories": int(med(j["num_test_trajectories"])),
        }

    return {
        "parameters": hp["parameters"],
        "data": hp["data_set"],
        "architecture": {
            "width": hp["hc_set"]["mlp_width"],
            "depth": hp["hc_set"]["mlp_depth"],
            "hidden_activation": hp["hc_set"]["hidden_activation"],
            "total_params": int(med(single_json["total_params"])),
        },
        "stopping": {
            "train_loss_threshold": hp["stop_set"]["train_loss_threshold"],
            "max_steps": hp["stop_set"]["max_steps"],
            "num_attempts": hp["stop_set"]["num_attempts"],
            "log_tvc_threshold": hp["stop_set"]["log_tvc_threshold"],
        },
        "n_seeds": int(single_json["n_seeds"]),
        "experiments": {name: experiment(name) for name in RECURSIVE_EXPERIMENTS},
    }


def stochastic_growth_nn_recursive(out):
    prefix = "stochastic_growth_nn_recursive"
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    j = load(out, f"{prefix}_baseline")
    df = pd.read_parquet(out / f"{prefix}_baseline_results.parquet")
    accepted = df[df["solution_accepted"]]
    if accepted.empty:
        raise RuntimeError("stochastic growth recursive NN has no accepted seeds")

    return {
        "parameters": hp["parameters"],
        "initial_conditions": hp["initial_conditions"],
        "data": hp["data_set"],
        "architecture": {
            "width": hp["hc_set"]["mlp_width"],
            "depth": hp["hc_set"]["mlp_depth"],
            "hidden_activation": hp["hc_set"]["hidden_activation"],
            "total_params": int(med(j["total_params"])),
        },
        "stopping": hp["stop_set"],
        "n_seeds": int(j["n_seeds"]),
        "experiments": {
            "baseline": {
                "frac_accepted": sig(j["frac_accepted"]),
                "test_mean_abs_k_rel_error_median": sci(
                    float(accepted["test_mean_abs_k_rel_error"].median())
                ),
                "test_mean_abs_c_rel_error_median": sci(
                    float(accepted["test_mean_abs_c_rel_error"].median())
                ),
                "policy_mean_abs_c_rel_error_median": sci(
                    float(accepted["policy_mean_abs_c_rel_error"].median())
                ),
                "test_euler_mean_abs_median": sci(
                    float(accepted["test_euler_mean_abs"].median())
                ),
                "max_test_euler_median": sci(
                    float(accepted["max_test_euler"].median())
                ),
                # FD-HJBE benchmark residual floor: NN-vs-FD relative errors are
                # only interpretable down to this scale.
                "fd_test_euler_mean_abs": sci(med(j["fd_test_euler_mean_abs"])),
                "max_log_tvc_residual_median": sig(
                    float(accepted["max_log_tvc_residual"].median()), 4
                ),
                "solve_time_median": sig(float(accepted["solve_time"].median())),
                "steps_median": sig(float(accepted["steps"].median())),
                **final_damping_summary(accepted),
                "sde_solver": str(accepted["sde_solver"].iloc[0]),
                "num_collocation_states": int(med(j["num_collocation_states"])),
                "num_train_trajectories": int(med(j["num_train_trajectories"])),
                "num_test_trajectories": int(med(j["num_test_trajectories"])),
            }
        },
    }


def neoclassical_growth_kernel_recursive(out):
    prefix = "neoclassical_growth_kernel_recursive"
    hp = load(out, f"{prefix}_single_ic_g0_hyperparameters")

    def experiment(name):
        j = load(out, f"{prefix}_{name}")
        return {
            "frac_accepted": sig(j["frac_accepted"]),
            "test_mean_abs_k_rel_error": sci(med(j["test_mean_abs_k_rel_error"])),
            "test_mean_abs_c_rel_error": sci(med(j["test_mean_abs_c_rel_error"])),
            "test_euler_mean_abs": sci(med(j["test_euler_mean_abs"])),
            "max_train_euler": sci(med(j["max_train_euler"])),
            "max_test_euler": sci(med(j["max_test_euler"])),
            "max_log_tvc_residual": sig(med(j["max_log_tvc_residual"]), 4),
            "solve_time": sig(med(j["solve_time"])),
            "steps": int(med(j["steps"])),
            "num_collocation_states": int(med(j["num_collocation_states"])),
            "penalized_parameter_count": int(med(j["penalized_parameter_count"])),
            "rkhs_norm": sig(med(j["rkhs_norm"])),
            "num_train_trajectories": int(med(j["num_train_trajectories"])),
            "num_test_trajectories": int(med(j["num_test_trajectories"])),
        }

    return {
        "parameters": hp["parameters"],
        "data": hp["data_set"],
        "kernel": hp["kernel_set"],
        "solver": {"ridge": hp["stop_set"]["ridge"]},
        "experiments": {name: experiment(name) for name in RECURSIVE_EXPERIMENTS},
    }


def open_economy_growth_kernel(
    out,
    prefix="open_economy_growth_kernel",
    selection="minimum_shifted_rkhs_metric",
):
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    baseline = load(out, f"{prefix}_baseline")
    summary = load(out, f"{prefix}_summary")
    if prefix == "open_economy_growth_kernel" and "comparative_statics" not in summary:
        raise RuntimeError(
            f"{prefix} summary lacks comparative_statics; rerun the replication driver"
        )
    comparative_statics = summary.get("comparative_statics")
    if comparative_statics is not None:
        comparative_statics = {
            "configurations": comparative_statics["configurations"],
            "test_T": sig(comparative_statics["test_T"]),
            **{
                field: sci(comparative_statics[field])
                for field in (
                    "max_k_gap",
                    "max_q_gap",
                    "max_investment_gap",
                    "max_current_account_gap",
                    "max_consumption_shift_error",
                    "max_asset_shift_error",
                )
            },
        }

    def experiment(name):
        j = load(out, f"{prefix}_{name}")
        results = pd.read_parquet(out / f"{prefix}_{name}_results.parquet")
        return {
            "solution_accepted": bool(j["frac_accepted"] == 1.0),
            **(
                {"validation_pass": bool(results["validation_pass"].all())}
                if "validation_pass" in results
                else {}
            ),
            "r_star": sig(med(j["r_star_resolved"]), 5),
            "k_0": sig(med(j["k_0"]), 5),
            "a_0": sig(med(j["a_0"]), 5),
            "num_train_points": int(med(j["num_train_points"])),
            "train_T": sig(med(j["train_T"]), 5),
            "max_train_residual": sci(med(j["max_train_residual"])),
            "max_validation_residual": sci(med(j["max_validation_residual"])),
            "max_benchmark_error": sci(med(j["max_benchmark_error"])),
            "k_relative_error_max": sci(med(j["k_relative_error_max"])),
            "q_relative_error_max": sci(med(j["q_relative_error_max"])),
            "c_relative_error_max": sci(med(j["c_relative_error_max"])),
            "a_scaled_error_max": sci(med(j["a_scaled_error_max"])),
            "benchmark_a_0_gap": sci(med(j["benchmark_a_0_gap"])),
            "max_consumption_variation": sci(med(j["max_consumption_variation"])),
            "rkhs_norm_c": sci(med(j["rkhs_norm_c"])),
            "ex_post_present_value_gap": sci(med(j["ex_post_present_value_gap"])),
            "ex_post_asset_tvc": sci(med(j["ex_post_asset_tvc"])),
            "ex_post_capital_tvc": sci(med(j["ex_post_capital_tvc"])),
            "solve_time_median": sig(med(j["solve_time"])),
            "timing_repetitions": int(med(j["timing_repetitions"])),
            **maybe_final_damping(j),
        }

    return {
        "parameters": hp["parameters"],
        "kernel": hp["kernel_set"],
        "selection": selection,
        **(
            {"comparative_statics": comparative_statics}
            if comparative_statics is not None
            else {}
        ),
        "training_information": {
            "terminal_residual_count": 0,
            "differential_residual_count": int(
                med(baseline["differential_residual_count"])
            ),
            "algebraic_residual_count": int(med(baseline["algebraic_residual_count"])),
            "dae_residual_count": int(med(baseline["dae_residual_count"])),
            "unknown_count": int(med(baseline["unknown_count"])),
        },
        "experiments": {
            name: experiment(name)
            for name in (
                "baseline",
                "r_star_005",
                "initial_debt",
                "initial_assets",
            )
        },
    }


def open_economy_growth_nn_recursive(out, prefix="open_economy_growth_nn_recursive"):
    # Shared by the sequence-space driver: identical experiments, result
    # columns, and architecture block.
    hp = load(out, f"{prefix}_baseline_hyperparameters")

    def experiment(name):
        j = load(out, f"{prefix}_{name}")
        frame = pd.read_parquet(out / f"{prefix}_{name}_results.parquet")
        accepted = frame[frame["solution_accepted"]]
        if accepted.empty:
            return {"frac_accepted": sig(j["frac_accepted"])}
        return {
            "frac_accepted": sig(j["frac_accepted"]),
            "k_0": sig(float(accepted["k_0"].median()), 5),
            "a_0": sig(float(accepted["a_0"].median()), 5),
            "max_train_residual_median": sci(
                float(accepted["max_train_residual"].median())
            ),
            "max_validation_residual_median": sci(
                float(accepted["max_validation_residual"].median())
            ),
            "max_benchmark_error_median": sci(
                float(accepted["max_benchmark_error"].median())
            ),
            "ex_post_present_value_gap_median": sci(
                float(accepted["ex_post_present_value_gap"].median())
            ),
            "ex_post_asset_tvc_median": sci(
                float(accepted["ex_post_asset_tvc"].median())
            ),
            "ex_post_capital_tvc_median": sci(
                float(accepted["ex_post_capital_tvc"].median())
            ),
            "solve_time_median": sig(float(accepted["solve_time"].median())),
            **final_damping_summary(accepted),
        }

    return {
        "parameters": hp["parameters"],
        "architecture": {
            "width": hp["nn_set"]["width"],
            "depth": hp["nn_set"]["depth"],
            "hidden_activation": hp["nn_set"]["hidden_activation"],
            "total_params": int(med(load(out, f"{prefix}_baseline")["total_params"])),
        },
        "selection": "minimum_identity_metric_parameter_step",
        "training_information": {"terminal_residual_count": 0},
        "n_seeds": int(load(out, f"{prefix}_baseline")["n_seeds"]),
        "experiments": {
            name: experiment(name)
            for name in (
                "baseline",
                "initial_debt",
                "initial_assets",
            )
        },
    }


def multicountry_growth_kernel(out):
    prefix = "multicountry_growth_kernel"

    def experiment(name):
        j = load(out, f"{prefix}_{name}")
        hp = load(out, f"{prefix}_{name}_hyperparameters")
        results = pd.read_parquet(out / f"{prefix}_{name}_results.parquet")
        return {
            "num_countries": len(hp["parameters"]["populations"]),
            "initial_conditions": {
                field: [sig(value, 6) for value in values]
                for field, values in hp["initial_conditions"].items()
            },
            "solution_accepted": bool(j["frac_accepted"] == 1.0),
            "validation_pass": bool(results["validation_pass"].all()),
            "num_train_points": int(med(j["num_train_points"])),
            "train_T": sig(med(j["train_T"]), 5),
            "max_train_residual": sci(med(j["max_train_residual"])),
            "max_validation_residual": sci(med(j["max_validation_residual"])),
            "max_resource_gap": sci(med(j["max_resource_gap"])),
            "max_asset_clearing": sci(med(j["max_asset_clearing"])),
            "max_benchmark_error": sci(med(j["max_benchmark_error"])),
            "k_relative_error_max": sci(med(j["k_relative_error_max"])),
            "q_relative_error_max": sci(med(j["q_relative_error_max"])),
            "c_relative_error_max": sci(med(j["c_relative_error_max"])),
            "a_scaled_error_max": sci(med(j["a_scaled_error_max"])),
            "interest_rate_scaled_error_max": sci(
                med(j["interest_rate_scaled_error_max"])
            ),
            "ex_post_asset_tvc": sci(med(j["ex_post_asset_tvc"])),
            "ex_post_capital_tvc": sci(med(j["ex_post_capital_tvc"])),
            "solve_time_median": sig(med(j["solve_time"])),
            "timing_repetitions": int(med(j["timing_repetitions"])),
            **final_damping(j),
        }

    baseline_hp = load(out, f"{prefix}_baseline_hyperparameters")
    summary = load(out, f"{prefix}_summary")
    if "bond_invariance" not in summary:
        raise RuntimeError(
            "multicountry kernel summary lacks bond_invariance; the "
            "replication.multicountry_growth_kernel driver writes it after "
            "write_summary, so rerun the driver rather than write_summary alone"
        )
    bond_invariance = summary["bond_invariance"]
    bond_invariance = {
        "configurations": bond_invariance["configurations"],
        "test_T": sig(bond_invariance["test_T"]),
        **{
            field: sci(bond_invariance[field])
            for field in (
                "max_interest_rate_gap",
                "max_k_gap",
                "max_q_gap",
                "max_gap",
                "max_interest_rate_gap_train_horizon",
                "max_k_gap_train_horizon",
                "max_q_gap_train_horizon",
                "max_gap_train_horizon",
            )
        },
    }
    payload = {
        "parameters": baseline_hp["parameters"],
        "kernel": baseline_hp["kernel_set"],
        "selection": "minimum_rkhs_norm_ridge",
        "training_information": {"terminal_residual_count": 0},
        # Max gap in (r*, k_i, q_i) paths across the three bond
        # configurations: exactly zero in the common-gamma theory, so the
        # measured gap is a solver-accuracy certificate.
        "bond_invariance": bond_invariance,
        "experiments": {
            name: experiment(name)
            for name in (
                "baseline",
                "initial_debt",
                "initial_assets",
                "three_countries",
                "heterogeneous_countries",
                "ten_countries",
            )
        },
    }
    # The implicit-AD runs are separate single-solve experiments
    # (multicountry_growth_kernel_sensitivity.py), float64 only.
    if (out / f"{prefix}_sensitivity.json").exists():
        sensitivity_json = load(out, f"{prefix}_sensitivity")
        payload["sensitivity"] = multicountry_sensitivity_payload(
            json_values(sensitivity_json)
        )
        payload["aggregate_sensitivity"] = multicountry_aggregate_payload(
            json_values(load(out, f"{prefix}_sensitivity_ten_countries")),
            reference=json_values(sensitivity_json),
        )
    return payload


# Mirrors the ladder in replication/multicountry_growth_scaling.py (kept
# jax-free here, like the sensitivity times above).
MULTICOUNTRY_SCALING_COUNTS = (2, 3, 5, 10, 15, 25, 40, 80, 100)
MULTICOUNTRY_SCALING_DENSE_MAX = 10
MULTICOUNTRY_SCALING_GAMMA_COMMON = 40


def multicountry_growth_scaling(out):
    summary = load(out, "multicountry_growth_scaling_summary")
    solve_times = {"dense": {}, "cg": {}}
    compile_solve_times = {"dense": {}, "cg": {}}
    for name, experiment in summary.items():
        solver, count = name.rsplit("_N", 1)
        solve_times[solver][count] = sig(med(experiment["solve_time"]))
        compile_solve_times[solver][count] = sig(med(experiment["compile_solve_time"]))
    cap = str(MULTICOUNTRY_SCALING_DENSE_MAX)
    train_gates = set()
    validation_diagnostics = set()
    kernel_settings = set()
    residual_atols = {}
    for count in MULTICOUNTRY_SCALING_COUNTS:
        hp = load(out, f"multicountry_growth_scaling_cg_N{count}_hyperparameters")
        train_gates.add(hp["tolerance_set"]["train_residual"])
        validation_diagnostics.add(hp["tolerance_set"]["validation_residual"])
        kernel_settings.add((hp["kernel_set"]["kernel"], hp["kernel_set"]["ell"]))
        residual_atols[str(count)] = hp["lm_set"]["residual_atol"]
        if count <= MULTICOUNTRY_SCALING_DENSE_MAX:
            dense_hp = load(
                out, f"multicountry_growth_scaling_dense_N{count}_hyperparameters"
            )
            if dense_hp != hp | {
                "lm_set": {
                    **hp["lm_set"],
                    "linear_solver": "cholesky",
                }
            }:
                raise RuntimeError(f"scaling dense and cg settings differ at N={count}")
    if train_gates != {1e-5}:
        raise RuntimeError(
            f"scaling train gates are not uniform: {sorted(train_gates)}"
        )
    if len(validation_diagnostics) != 1:
        raise RuntimeError(
            "scaling validation diagnostics are not uniform: "
            f"{sorted(validation_diagnostics)}"
        )
    if len(kernel_settings) != 1:
        raise RuntimeError(f"scaling kernel settings differ: {sorted(kernel_settings)}")
    kernel, ell = kernel_settings.pop()
    sensitivity_rung = load(
        out, f"multicountry_growth_scaling_cg_N{MULTICOUNTRY_SCALING_GAMMA_COMMON}"
    )
    # The tangent itself comes from the gamma-common driver's own stem; the
    # rung's primal solve times stay with the ladder rung it extends.
    gamma_common = load(out, "multicountry_growth_scaling_gamma_common")
    return {
        "country_counts": list(MULTICOUNTRY_SCALING_COUNTS),
        "dense_max_countries": MULTICOUNTRY_SCALING_DENSE_MAX,
        "kernel": {"family": kernel, "ell": ell},
        "solve_times": solve_times,
        "compile_solve_times": compile_solve_times,
        "warm_timing_repetitions": int(med(sensitivity_rung["timing_repetitions"])),
        "dense_to_cg_ratio_at_cap": sig(
            solve_times["dense"][cap] / solve_times["cg"][cap]
        ),
        "compile_to_warm_ratio_at_cap": sig(
            compile_solve_times["dense"][cap] / solve_times["dense"][cap]
        ),
        "train_residual_gate": train_gates.pop(),
        "max_achieved_train_residual": sci(
            max(
                med(experiment["max_train_residual"]) for experiment in summary.values()
            )
        ),
        "validation_residual_diagnostic": validation_diagnostics.pop(),
        "max_validation_residual": sci(
            max(
                med(experiment["max_validation_residual"])
                for experiment in summary.values()
            )
        ),
        "residual_atol_by_country_count": residual_atols,
        # The representative-agent CRRA comparative static, one implicit-AD
        # forward tangent through the cg solve; comparable against the
        # N=2/N=10 sensitivity tables' _common entries, with both runtimes
        # alongside.
        "gamma_common_sensitivity": {
            "num_countries": MULTICOUNTRY_SCALING_GAMMA_COMMON,
            "rate_T10": sig(med(gamma_common["rate_T10"]), 4),
            "d_rate_T10_d_gamma_common": sig(
                med(gamma_common["d_rate_T10_d_gamma_common"]), 4
            ),
            "solve_seconds": sig(med(sensitivity_rung["solve_time"])),
            "compile_solve_seconds": sig(med(sensitivity_rung["compile_solve_time"])),
            "sensitivity_seconds": sig(
                med(gamma_common["gamma_common_sensitivity_time"])
            ),
            "compile_sensitivity_seconds": sig(
                med(gamma_common["compile_gamma_common_sensitivity_time"])
            ),
        },
    }


def multicountry_growth_nn(out):
    prefix = "multicountry_growth_nn"
    experiments = {}
    experiment_data = {}
    for name in ("baseline", "large_countries"):
        summary = load(out, f"{prefix}_{name}")
        hp = load(out, f"{prefix}_{name}_hyperparameters")
        frame = pd.read_parquet(out / f"{prefix}_{name}_results.parquet")
        accepted = frame[frame["solution_accepted"]]
        if accepted.empty:
            raise RuntimeError(f"multicountry_growth_nn {name} has no accepted seeds")
        total_params = int(accepted["total_params"].median())
        residual_count = int(accepted["dae_residual_count"].median())
        experiments[name] = {
            "scenario": summary["scenario"],
            "num_countries": len(hp["parameters"]["populations"]),
            "parameters": hp["parameters"],
            "initial_conditions": hp["initial_conditions"],
            "architecture": {
                "width": hp["nn_set"]["width"],
                "depth": hp["nn_set"]["depth"],
                "hidden_activation": hp["nn_set"]["hidden_activation"],
                "total_params": total_params,
            },
            "residual_count": residual_count,
            "parameter_surplus": total_params - residual_count,
            "n_seeds": int(summary["n_seeds"]),
            "fraction_accepted": sig(summary["frac_accepted"]),
            "solve_time_median": sig(float(accepted["solve_time"].median())),
            "max_train_residual_median": sci(
                float(accepted["max_train_residual"].median())
            ),
            "max_validation_residual_median": sci(
                float(accepted["max_validation_residual"].median())
            ),
            "max_resource_gap_median": sci(
                float(accepted["max_resource_gap"].median())
            ),
            **final_damping_summary(accepted),
        }
        experiment_data[name] = (summary, hp, accepted)

    baseline_summary, baseline_hp, baseline_accepted = experiment_data["baseline"]
    experiments["baseline"].update(
        {
            "max_benchmark_error_median": sci(
                float(baseline_accepted["max_benchmark_error"].median())
            ),
            "log_relative_weight_error_median": sci(
                float(
                    (
                        baseline_accepted["log_relative_weight"]
                        - baseline_accepted["benchmark_log_relative_weight"]
                    )
                    .abs()
                    .median()
                )
            ),
        }
    )
    payload = {
        "parameters": baseline_hp["parameters"],
        "architecture": experiments["baseline"]["architecture"],
        "selection": "minimum_identity_metric_parameter_step",
        "training_information": {
            "terminal_residual_count": 0,
            "benchmark_training_residual_count": 0,
            "tvc_training_residual_count": 0,
        },
        "n_seeds": int(baseline_summary["n_seeds"]),
        "experiments": experiments,
    }
    # The implicit-AD runs are separate single-seed experiments, float64 only.
    if (out / f"{prefix}_sensitivity.json").exists():
        two_country = accepted_values(out, f"{prefix}_sensitivity")
        payload["sensitivity"] = multicountry_sensitivity_payload(
            two_country,
            reference=json_values(load(out, "multicountry_growth_kernel_sensitivity")),
        )
        payload["aggregate_sensitivity"] = multicountry_aggregate_payload(
            accepted_values(out, f"{prefix}_sensitivity_ten_countries"),
            reference=two_country,
        )
    return payload


def performance_comparisons(out, include_uno=True):
    def accepted(prefix, experiment="baseline"):
        frame = pd.read_parquet(out / f"{prefix}_{experiment}_results.parquet")
        accepted_frame = frame[frame["solution_accepted"]]
        if accepted_frame.empty:
            raise RuntimeError(f"{prefix} {experiment} has no accepted solutions")
        return accepted_frame

    asset_kernel = load(out, "asset_pricing_kernel_baseline")
    asset_nn = accepted("asset_pricing_nn")
    asset_kernel_time = med(asset_kernel["solve_time"])
    asset_nn_time = float(asset_nn["solve_time"].median())
    asset_kernel_error = med(asset_kernel["p_rel_error_mean"])
    asset_nn_error = float(asset_nn["p_rel_error_mean"].median())
    if include_uno:
        asset_uno = load(out, "asset_pricing_kernel_uno_baseline")
        asset_uno_time = med(asset_uno["solve_time"])
        asset_uno_error = med(asset_uno["p_rel_error_mean"])
    else:
        asset_uno_time = asset_uno_error = None

    growth_kernel = load(out, "neoclassical_growth_kernel_baseline")
    growth_sequential_nn = accepted("neoclassical_growth_nn")
    growth_recursive_nn = accepted("neoclassical_growth_nn_recursive", "single_ic_g0")
    growth_kernel_time = med(growth_kernel["solve_time"])
    growth_kernel_error = max(
        med(growth_kernel["test_k_rel_error_mean"]),
        med(growth_kernel["test_c_rel_error_mean"]),
    )
    if include_uno:
        growth_uno = load(out, "neoclassical_growth_kernel_uno_baseline")
        growth_uno_time = med(growth_uno["solve_time"])
        growth_uno_error = max(
            med(growth_uno["test_k_rel_error_mean"]),
            med(growth_uno["test_c_rel_error_mean"]),
        )
    else:
        growth_uno_time = growth_uno_error = None

    # Both NN formulations against the same sequence-space kernel baseline: the
    # sequential MLP shares the kernel's t -> (k, c) ansatz, the recursive one
    # learns a state-space policy rolled forward by RK4.
    def growth_nn_stats(frame, k_error_field, c_error_field):
        return (
            float(frame["solve_time"].median()),
            max(
                float(frame[k_error_field].median()),
                float(frame[c_error_field].median()),
            ),
        )

    open_kernel = load(out, "open_economy_growth_kernel_baseline")
    open_sequential_nn = accepted("open_economy_growth_nn")
    open_recursive_nn = accepted("open_economy_growth_nn_recursive")
    open_kernel_time = med(open_kernel["solve_time"])
    open_kernel_error = med(open_kernel["max_benchmark_error"])
    if include_uno:
        open_uno = load(out, "open_economy_growth_kernel_uno_baseline")
        open_uno_time = med(open_uno["solve_time"])
        open_uno_error = med(open_uno["max_benchmark_error"])
    else:
        open_uno_time = open_uno_error = None

    def open_nn_stats(frame):
        return (
            float(frame["solve_time"].median()),
            float(frame["max_benchmark_error"].median()),
        )

    multicountry_kernel = load(out, "multicountry_growth_kernel_initial_debt")
    multicountry_nn_frame = accepted("multicountry_growth_nn")
    multicountry_kernel_time = med(multicountry_kernel["solve_time"])
    multicountry_nn_time = float(multicountry_nn_frame["solve_time"].median())
    multicountry_kernel_error = med(multicountry_kernel["max_benchmark_error"])
    multicountry_nn_error = float(multicountry_nn_frame["max_benchmark_error"].median())

    # nn_variants maps a key prefix to (time, error). A lone "nn" entry keeps the
    # flat nn_* keys; neoclassical growth carries sequential and recursive
    # side by side against the one kernel baseline.
    def comparison(
        kernel_time,
        kernel_error,
        nn_variants,
        uno_time=None,
        uno_error=None,
    ):
        values = {
            "kernel_solve_time_median": sig(kernel_time),
            "kernel_error": sci(kernel_error),
        }
        for name, (nn_time, nn_error) in nn_variants.items():
            values.update(
                {
                    f"{name}_solve_time_median": sig(nn_time),
                    f"{name}_to_kernel_time_ratio": sig(nn_time / kernel_time),
                    f"{name}_error_median": sci(nn_error),
                    f"{name}_to_kernel_error_ratio": sig(nn_error / kernel_error),
                }
            )
        if uno_time is not None:
            values.update(
                {
                    "uno_solve_time_median": sig(uno_time),
                    "uno_to_kernel_time_ratio": sig(uno_time / kernel_time),
                    "uno_error": sci(uno_error),
                    "uno_to_kernel_error_ratio": sig(uno_error / kernel_error),
                }
            )
            for name, (nn_time, nn_error) in nn_variants.items():
                values.update(
                    {
                        f"{name}_to_uno_time_ratio": sig(nn_time / uno_time),
                        f"{name}_to_uno_error_ratio": sig(nn_error / uno_error),
                    }
                )
        return values

    timing_contract = {
        "kernel": "median of five cached deterministic solves",
        "nn": (
            "median across accepted seeds from 100 initializations "
            "(21 for the open-economy sequence-space NN, seven for "
            "multicountry growth), each seed allowed up to five "
            "multi-start restarts"
        ),
    }
    if include_uno:
        timing_contract["uno"] = (
            "median of five cached deterministic constrained solves"
        )
    return {
        "timing_contract": timing_contract,
        "asset_pricing": comparison(
            asset_kernel_time,
            asset_kernel_error,
            {"nn": (asset_nn_time, asset_nn_error)},
            asset_uno_time,
            asset_uno_error,
        ),
        "neoclassical_growth": comparison(
            growth_kernel_time,
            growth_kernel_error,
            {
                "sequential_nn": growth_nn_stats(
                    growth_sequential_nn,
                    "test_k_rel_error_mean",
                    "test_c_rel_error_mean",
                ),
                "recursive_nn": growth_nn_stats(
                    growth_recursive_nn,
                    "test_mean_abs_k_rel_error",
                    "test_mean_abs_c_rel_error",
                ),
            },
            growth_uno_time,
            growth_uno_error,
        ),
        "open_economy_growth": comparison(
            open_kernel_time,
            open_kernel_error,
            {
                "sequential_nn": open_nn_stats(open_sequential_nn),
                "recursive_nn": open_nn_stats(open_recursive_nn),
            },
            open_uno_time,
            open_uno_error,
        ),
        "multicountry_growth": comparison(
            multicountry_kernel_time,
            multicountry_kernel_error,
            {"nn": (multicountry_nn_time, multicountry_nn_error)},
        ),
    }


def validate_replication_contract(out, include_uno=True):
    # (headline experiment, expected seed count) per NN prefix; the recursive
    # growth driver's headline experiment is single_ic_g0, not baseline.
    nn_seed_counts = {
        "asset_pricing_nn": ("baseline", 7),
        "neoclassical_growth_nn": ("baseline", 7),
        "neoclassical_growth_nn_recursive": ("single_ic_g0", 5),
        "open_economy_growth_nn": ("baseline", 7),
        "open_economy_growth_nn_recursive": ("baseline", 5),
        "multicountry_growth_nn": ("baseline", 7),
        "stochastic_growth_nn_recursive": ("baseline", 5),
    }
    for prefix, (experiment, expected) in nn_seed_counts.items():
        summary = load(out, f"{prefix}_{experiment}")
        if summary["n_seeds"] != expected:
            raise RuntimeError(
                f"{prefix} must use {expected} seeds, got {summary['n_seeds']}"
            )

    kernel_prefixes = (
        "neoclassical_growth_kernel",
        "asset_pricing_kernel",
        "optimal_advertising_kernel",
        "human_capital_kernel",
        "open_economy_growth_kernel",
        "multicountry_growth_kernel",
    )
    if include_uno:
        kernel_prefixes += (
            "open_economy_growth_kernel_uno",
            "asset_pricing_kernel_uno",
            "neoclassical_growth_kernel_uno",
        )
    for prefix in kernel_prefixes:
        timing = pd.read_parquet(out / f"{prefix}_baseline_timing_df.parquet")
        if len(timing) != DETERMINISTIC_TIMING_REPETITIONS:
            raise RuntimeError(
                f"{prefix} must retain {DETERMINISTIC_TIMING_REPETITIONS} timing runs"
            )

    # The scaling ladder runs only in the float64 pass.
    residual_dtype = load(out, "neoclassical_growth_kernel_baseline").get(
        "residual_dtype"
    )
    if residual_dtype == "float64":
        summary = load(out, "multicountry_growth_scaling_summary")
        expected = {f"cg_N{count}" for count in MULTICOUNTRY_SCALING_COUNTS} | {
            f"dense_N{count}"
            for count in MULTICOUNTRY_SCALING_COUNTS
            if count <= MULTICOUNTRY_SCALING_DENSE_MAX
        }
        missing = sorted(expected - set(summary))
        if missing:
            raise RuntimeError(
                f"multicountry_growth_scaling summary lacks rungs {missing}; "
                "rerun replication.multicountry_growth_scaling"
            )
        for name in sorted(expected):
            if summary[name]["frac_accepted"] != 1.0:
                raise RuntimeError(
                    f"multicountry_growth_scaling {name} was not accepted"
                )


def collect(output_dir: str = ".results", include_uno: bool = True):
    out = pathlib.Path(output_dir)
    validate_replication_contract(out, include_uno)
    # The collected precision comes from the collected data itself.
    residual_dtype = load(out, "neoclassical_growth_kernel_baseline").get(
        "residual_dtype"
    )
    write_nu_rho_table(out)
    write_sequential_comparison_table(out)
    write_neoclassical_growth_comparison_table(out)
    # The sensitivity experiments run only in the float64 pass.
    if residual_dtype == "float64":
        write_neoclassical_sensitivity_table(out)
        # One table per solver and cross-section width; the kernel and NN
        # numbers stand on their own rather than being differenced.
        write_multicountry_sensitivity_table(
            out,
            "multicountry_growth_kernel_sensitivity",
            "multicountry_growth_sensitivity.tex",
        )
        write_multicountry_aggregate_table(
            out,
            "multicountry_growth_kernel_sensitivity_ten_countries",
            "multicountry_growth_aggregate_sensitivity.tex",
        )
        write_multicountry_sensitivity_table(
            out,
            "multicountry_growth_nn_sensitivity",
            "multicountry_growth_nn_sensitivity.tex",
        )
        write_multicountry_aggregate_table(
            out,
            "multicountry_growth_nn_sensitivity_ten_countries",
            "multicountry_growth_nn_aggregate_sensitivity.tex",
        )
    payload = clean(
        {
            "_meta": {
                "description": (
                    "Authoritative source for every number embedded in the "
                    "paper: each value is selected and rounded to copy/paste "
                    "form here (replication/collect_json.py) and must never be "
                    "recomputed or re-rounded downstream. Cross-seed statistics "
                    "are medians (suffix _median); deterministic solve_time values "
                    f"are medians of {DETERMINISTIC_TIMING_REPETITIONS} cached "
                    "solve(s), except the multicountry scaling ladder, which "
                    "reports the median of two warmed solves after one separately "
                    "retained first solve; error magnitudes below 0.001 are LaTeX "
                    "math-mode strings from sci()."
                ),
                "generated": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "machine": machine_info(),
                "output_dir": output_dir,
                "residual_dtype": residual_dtype,
                "include_uno": include_uno,
                "band_quantiles": [int(100 * QUANTILES[0]), int(100 * QUANTILES[-1])],
            },
            "neoclassical_growth_kernel": neoclassical_growth_kernel(out),
            **(
                {
                    "neoclassical_growth_kernel_uno": neoclassical_growth_kernel_uno(
                        out
                    ),
                    "neoclassical_growth_concave_convex_kernel_uno": (
                        neoclassical_growth_concave_convex_kernel(
                            out,
                            prefix="neoclassical_growth_concave_convex_kernel_uno",
                            require_all=False,
                        )
                    ),
                    "asset_pricing_kernel_uno": asset_pricing_kernel(
                        out, prefix="asset_pricing_kernel_uno"
                    ),
                    "open_economy_growth_kernel_uno": open_economy_growth_kernel(
                        out,
                        prefix="open_economy_growth_kernel_uno",
                        selection="minimum_rkhs_norm_constrained",
                    ),
                }
                if include_uno
                else {}
            ),
            # The all-70 basin contract is the float64 caption's claim; a
            # float32 collection reports n_basin_correct as data (platform
            # rounding moves which knife-edge point misclassifies).
            "neoclassical_growth_concave_convex_kernel": (
                neoclassical_growth_concave_convex_kernel(
                    out, require_all=residual_dtype == "float64"
                )
            ),
            # require_all=False for both NNs: solver acceptance is scored
            # separately from basin correctness, so a rejected grid point is
            # data rather than a pipeline failure.
            "neoclassical_growth_concave_convex_nn": (
                neoclassical_growth_concave_convex_kernel(
                    out,
                    prefix="neoclassical_growth_concave_convex_nn",
                    require_all=False,
                )
            ),
            "asset_pricing_kernel": asset_pricing_kernel(out),
            "asset_pricing_nn": asset_pricing_nn(out),
            "optimal_advertising_kernel": optimal_advertising_kernel(out),
            "human_capital_kernel": human_capital_kernel(out),
            "neoclassical_growth_nn": neoclassical_growth_nn(out),
            "neoclassical_growth_nn_recursive": neoclassical_growth_nn_recursive(out),
            "neoclassical_growth_kernel_recursive": neoclassical_growth_kernel_recursive(
                out
            ),
            "stochastic_growth_nn_recursive": stochastic_growth_nn_recursive(out),
            "open_economy_growth_kernel": open_economy_growth_kernel(out),
            "open_economy_growth_nn": open_economy_growth_nn_recursive(
                out, prefix="open_economy_growth_nn"
            ),
            "open_economy_growth_nn_recursive": open_economy_growth_nn_recursive(out),
            "multicountry_growth_kernel": multicountry_growth_kernel(out),
            **(
                {"multicountry_growth_scaling": multicountry_growth_scaling(out)}
                if residual_dtype == "float64"
                else {}
            ),
            "multicountry_growth_nn": multicountry_growth_nn(out),
            "performance_comparisons": performance_comparisons(out, include_uno),
        }
    )
    # allow_nan=False guarantees the output is strict JSON (any NaN raises here).
    (out / "all_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {out / 'all_results.json'}")


if __name__ == "__main__":
    jsonargparse.CLI(collect)
