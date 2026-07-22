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

from replication.utilities import QUANTILES

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
    "k_rel_error_max",
    "c_rel_error_max",
    "k_rel_error_min",
    "c_rel_error_min",
]

KERNEL_NU = {
    "matern12": 0.5,
    "matern32": 1.5,
    "matern52": 2.5,
    "rbf": "inf",
}


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


def neoclassical_growth_kernel(out):
    prefix = "neoclassical_growth_kernel"
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    baseline = load(out, f"{prefix}_baseline")

    def errors(j):
        return {
            "solution_accepted": bool(j["n_accepted"] == 1),
            "k_rel_error_mean": sci(med(j["k_rel_error_mean"])),
            "c_rel_error_mean": sci(med(j["c_rel_error_mean"])),
            "k_rel_error_max": sci(med(j["k_rel_error_max"])),
            "c_rel_error_max": sci(med(j["c_rel_error_max"])),
            "k_rel_error_min": sci(med(j["k_rel_error_min"])),
            "c_rel_error_min": sci(med(j["c_rel_error_min"])),
            "max_train_residual": sci(med(j["max_train_residual"]), 2),
            "wall_time": sig(med(j["wall_time"]), 2),
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


def neoclassical_growth_concave_convex_kernel(out):
    prefix = "neoclassical_growth_concave_convex_kernel"
    j = load(out, f"{prefix}_threshold")
    hp = load(out, f"{prefix}_threshold_hyperparameters")
    df = pd.read_parquet(out / f"{prefix}_threshold_results.parquet")
    n_total = int(len(df))
    n_accepted = int(df["solution_accepted"].sum())
    if n_accepted != n_total:
        # The QE caption states every initial condition converges.
        rejected = df.loc[~df["solution_accepted"], "k_0"].tolist()
        raise RuntimeError(f"rejected threshold solves at k_0={rejected}")
    # Empirical basin threshold from the side each path ACTUALLY settles on
    # (basin_correct = analytic side, false_positive = opposite side), so a
    # wrong-basin path moves the bracket instead of being silently dropped.
    above_thr = df["k_0"] > df["k_thr"]
    ends_high = (df["basin_correct"] & above_thr) | (df["false_positive"] & ~above_thr)
    goes_low = (df["basin_correct"] & ~above_thr) | (df["false_positive"] & above_thr)
    if not (ends_high | goes_low).all():
        unresolved = df.loc[~(ends_high | goes_low), "k_0"].tolist()
        raise RuntimeError(f"unclassified threshold paths at k_0={unresolved}")
    k_0_low_max = float(df.loc[goes_low, "k_0"].max())
    k_0_high_min = float(df.loc[ends_high, "k_0"].min())
    return {
        "parameters": {
            "A": hp["parameters"]["A"],
            "b_1": hp["parameters"]["b_1"],
            "b_2": hp["parameters"]["b_2"],
        },
        "threshold": {
            "k_0_grid": {
                "min": sig(df["k_0"].min()),
                "max": sig(df["k_0"].max()),
                "n": n_total,
            },
            "n_accepted": n_accepted,
            "n_basin_correct": int(df["basin_correct"].sum()),
            "k_thr": sig(float(df["k_thr"].iloc[0]), 6),
            "k_ss_low": sig(float(df["k_ss_low"].iloc[0]), 4),
            "k_ss_high": sig(float(df["k_ss_high"].iloc[0]), 4),
            # Bracketing grid points around the empirical basin split.
            "k_0_threshold_bracket": [sig(k_0_low_max, 4), sig(k_0_high_min, 4)],
            "k_0_threshold": sig(0.5 * (k_0_low_max + k_0_high_min), 4),
        },
        "n_seeds": int(j["n_seeds"]),
    }


def asset_pricing_kernel(out):
    prefix = "asset_pricing_kernel"
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
                "wall_time": sig(med(j["wall_time"]), 2),
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
                "train_time_median": sig(float(accepted["train_time"].median())),
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
        "wall_time": sig(med(j["wall_time"]), 2),
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
            "lambda_p": hp["parameters"]["lambda_p"],
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
        "wall_time": sig(med(j["wall_time"]), 2),
    }


def neoclassical_growth_nn_recursive(out):
    prefix = "neoclassical_growth_nn_recursive"

    # Medians over ACCEPTED seeds only (a rejected attempt burns the full
    # training budget, so all-seed medians would mix in outliers).
    baseline_json = load(out, f"{prefix}_baseline")
    baseline_df = pd.read_parquet(out / f"{prefix}_baseline_results.parquet")
    accepted = baseline_df[baseline_df["solution_accepted"]]
    hp = load(out, f"{prefix}_baseline_hyperparameters")

    bf_hp = load(out, f"{prefix}_butterfly_basin_hyperparameters")
    bf = load(out, f"{prefix}_butterfly_basin")
    bf_df = pd.read_parquet(out / f"{prefix}_butterfly_basin_results.parquet")
    frac_by_k0 = bf_df.groupby("k_0")["basin_correct"].mean()

    return {
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
                "test_mean_abs_k_rel_error_median": sci(
                    float(accepted["test_mean_abs_k_rel_error"].median())
                ),
                "test_mean_abs_c_rel_error_median": sci(
                    float(accepted["test_mean_abs_c_rel_error"].median())
                ),
                "test_euler_mean_abs_median": sci(
                    float(accepted["test_euler_mean_abs"].median())
                ),
                "max_log_tvc_residual_median": sig(
                    float(accepted["max_log_tvc_residual"].median()), 4
                ),
                "train_time_median": sig(float(accepted["train_time"].median())),
                "steps_median": sig(float(accepted["steps"].median())),
                "num_collocation_states": int(
                    med(baseline_json["num_collocation_states"])
                ),
                "ode_solver": hp["data_set"]["ode_solver"],
                "train_steps": hp["data_set"]["train_steps"],
                "k_ss": sig(med(baseline_json["k_ss"]), 4),
            }
        },
        "butterfly": {
            "parameters": {
                "A": bf_hp["parameters"]["A"],
                "b_1": bf_hp["parameters"]["b_1"],
                "b_2": bf_hp["parameters"]["b_2"],
            },
            "steady_states": {
                "k_thr": sig(med(bf["k_thr"]), 6),
                "k_ss_low": sig(med(bf["k_ss_low"]), 4),
                "k_ss_high": sig(med(bf["k_ss_high"]), 4),
            },
            "k_0_grid": {
                "min": sig(float(bf_df["k_0"].min())),
                "max": sig(float(bf_df["k_0"].max())),
                "n": int(bf_df["k_0"].nunique()),
            },
            "n_solver_converged": int(bf_df["solution_accepted"].sum()),
            "frac_solver_converged": sig(float(bf_df["solution_accepted"].mean())),
            "frac_basin_correct": sig(float(bf_df["basin_correct"].mean())),
            "frac_false_positive": sig(float(bf_df["false_positive"].mean())),
            "frac_correct_by_k_0": {
                f"{k_0:g}": sig(float(value)) for k_0, value in frac_by_k0.items()
            },
        },
    }


def collect(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    write_nu_rho_table(out)
    payload = clean(
        {
            "_meta": {
                "description": (
                    "Authoritative source for every number embedded in the "
                    "paper: each value is selected and rounded to copy/paste "
                    "form here (replication/collect_json.py) and must never be "
                    "recomputed or re-rounded downstream. Cross-seed statistics "
                    "are medians (suffix _median); error magnitudes below 0.001 "
                    "are LaTeX math-mode strings from sci()."
                ),
                "generated": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "machine": machine_info(),
                "output_dir": output_dir,
                "band_quantiles": [int(100 * QUANTILES[0]), int(100 * QUANTILES[-1])],
            },
            "neoclassical_growth_kernel": neoclassical_growth_kernel(out),
            "neoclassical_growth_concave_convex_kernel": (
                neoclassical_growth_concave_convex_kernel(out)
            ),
            "asset_pricing_kernel": asset_pricing_kernel(out),
            "asset_pricing_nn": asset_pricing_nn(out),
            "optimal_advertising_kernel": optimal_advertising_kernel(out),
            "human_capital_kernel": human_capital_kernel(out),
            "neoclassical_growth_nn_recursive": neoclassical_growth_nn_recursive(out),
        }
    )
    # allow_nan=False guarantees the output is strict JSON (any NaN raises here).
    (out / "all_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {out / 'all_results.json'}")


if __name__ == "__main__":
    jsonargparse.CLI(collect)
