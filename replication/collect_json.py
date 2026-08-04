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


def neoclassical_growth_kernel_uno(out):
    prefix = "neoclassical_growth_kernel_uno"
    hp = load(out, f"{prefix}_baseline_hyperparameters")

    def errors(j):
        return {
            "solution_accepted": bool(j["n_accepted"] == 1),
            "k_rel_error_mean": sci(med(j["k_rel_error_mean"])),
            "c_rel_error_mean": sci(med(j["c_rel_error_mean"])),
            "k_rel_error_max": sci(med(j["k_rel_error_max"])),
            "c_rel_error_max": sci(med(j["c_rel_error_max"])),
            "max_train_residual": sci(med(j["max_train_residual"]), 2),
            "wall_time": sig(med(j["wall_time"]), 2),
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
        nu = experiment_hp["kernel_set"]["nu"]
        robustness_table[name if name != "baseline" else "nu05_ell10"] = {
            "nu": "inf" if math.isinf(nu) else nu,
            "ell": ell,
            **errors(result),
        }

    baseline_kernel = hp["kernel_set"]
    return {
        "kernel": {
            "nu": baseline_kernel["nu"],
            "ell": baseline_kernel["ell"],
            "sigma": baseline_kernel["sigma"],
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
                "wall_time": sig(med(j["wall_time"]), 2),
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
                "train_time_median": sig(float(accepted["train_time"].median())),
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
        "wall_time": sig(med(j["wall_time"]), 2),
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
        "wall_time": sig(med(j["wall_time"]), 2),
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

    return {
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
                "test_mean_abs_k_rel_error_median": sci(
                    float(accepted["test_mean_abs_k_rel_error"].median())
                ),
                "test_mean_abs_c_rel_error_median": sci(
                    float(accepted["test_mean_abs_c_rel_error"].median())
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
                "train_time_median": sig(float(accepted["train_time"].median())),
                "steps_median": sig(float(accepted["steps"].median())),
                **final_damping_summary(accepted),
                "num_train_points": int(med(baseline_json["num_train_points"])),
            }
        },
    }


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
            "train_time_median": sig(float(sample["train_time"].median())),
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
            "width": hp["nn_set"]["width"],
            "depth": hp["nn_set"]["depth"],
            "hidden_activation": hp["nn_set"]["hidden_activation"],
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
        "solver": {"ridge": hp["lm_set"]["ridge"]},
        "experiments": {name: experiment(name) for name in RECURSIVE_EXPERIMENTS},
    }


def open_economy_growth_kernel(
    out,
    prefix="open_economy_growth_kernel",
    selection="minimum_shifted_rkhs_metric",
):
    hp = load(out, f"{prefix}_baseline_hyperparameters")
    baseline = load(out, f"{prefix}_baseline")

    def experiment(name):
        j = load(out, f"{prefix}_{name}")
        return {
            "solution_accepted": bool(j["frac_accepted"] == 1.0),
            "r_star": sig(med(j["r_star_resolved"]), 5),
            "k_0": sig(med(j["k_0"]), 5),
            "a_0": sig(med(j["a_0"]), 5),
            "num_train_points": int(med(j["num_train_points"])),
            "train_T": sig(med(j["train_T"]), 5),
            "max_train_residual": sci(med(j["max_train_residual"])),
            "max_validation_residual": sci(med(j["max_validation_residual"])),
            "max_benchmark_error": sci(med(j["max_benchmark_error"])),
            "ex_post_present_value_gap": sci(med(j["ex_post_present_value_gap"])),
            "ex_post_asset_tvc": sci(med(j["ex_post_asset_tvc"])),
            "ex_post_capital_tvc": sci(med(j["ex_post_capital_tvc"])),
            "wall_time_median": sig(med(j["wall_time"])),
            "timing_repetitions": int(med(j["timing_repetitions"])),
            **maybe_final_damping(j),
        }

    return {
        "parameters": hp["parameters"],
        "kernel": hp["kernel_set"],
        "selection": selection,
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
            "train_time_median": sig(float(accepted["train_time"].median())),
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
        return {
            "num_countries": len(hp["parameters"]["populations"]),
            "initial_conditions": hp["initial_conditions"],
            "solution_accepted": bool(j["frac_accepted"] == 1.0),
            "num_train_points": int(med(j["num_train_points"])),
            "train_T": sig(med(j["train_T"]), 5),
            "max_train_residual": sci(med(j["max_train_residual"])),
            "max_validation_residual": sci(med(j["max_validation_residual"])),
            "max_asset_clearing_residual": sci(med(j["max_asset_clearing_residual"])),
            "max_rate_consistency_residual": sci(
                med(j["max_rate_consistency_residual"])
            ),
            "max_benchmark_error": sci(med(j["max_benchmark_error"])),
            "ex_post_asset_tvc": sci(med(j["ex_post_asset_tvc"])),
            "ex_post_capital_tvc": sci(med(j["ex_post_capital_tvc"])),
            "wall_time_median": sig(med(j["wall_time"])),
            "timing_repetitions": int(med(j["timing_repetitions"])),
            **final_damping(j),
        }

    baseline_hp = load(out, f"{prefix}_baseline_hyperparameters")
    return {
        "parameters": baseline_hp["parameters"],
        "kernel": baseline_hp["kernel_set"],
        "selection": "minimum_shifted_rkhs_metric",
        "training_information": {"terminal_residual_count": 0},
        "experiments": {
            name: experiment(name)
            for name in (
                "baseline",
                "initial_debt",
                "initial_assets",
                "asymmetric_capital",
                "three_countries",
                "heterogeneous_countries",
            )
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
                "dynamic_output_count": int(
                    accepted["dynamic_network_output_count"].median()
                ),
            },
            "residual_count": residual_count,
            "parameter_surplus": total_params - residual_count,
            "n_seeds": int(summary["n_seeds"]),
            "fraction_accepted": sig(summary["frac_accepted"]),
            "fraction_equation_diagnostics_passed": sig(
                float(accepted["equation_diagnostics_pass"].mean())
            ),
            "train_time_median": sig(float(accepted["train_time"].median())),
            "max_train_residual_median": sci(
                float(accepted["max_train_residual"].median())
            ),
            "max_validation_residual_median": sci(
                float(accepted["max_validation_residual"].median())
            ),
            "max_asset_clearing_residual_median": sci(
                float(accepted["max_asset_clearing_residual"].median())
            ),
            "max_rate_consistency_residual_median": sci(
                float(accepted["max_rate_consistency_residual"].median())
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
    return {
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


def performance_comparisons(out):
    def accepted(prefix):
        frame = pd.read_parquet(out / f"{prefix}_baseline_results.parquet")
        accepted_frame = frame[frame["solution_accepted"]]
        if accepted_frame.empty:
            raise RuntimeError(f"{prefix} baseline has no accepted solutions")
        return accepted_frame

    asset_kernel = load(out, "asset_pricing_kernel_baseline")
    asset_uno = load(out, "asset_pricing_kernel_uno_baseline")
    asset_nn = accepted("asset_pricing_nn")
    asset_kernel_time = med(asset_kernel["wall_time"])
    asset_nn_time = float(asset_nn["train_time"].median())
    asset_kernel_error = med(asset_kernel["p_rel_error_mean"])
    asset_uno_time = med(asset_uno["wall_time"])
    asset_uno_error = med(asset_uno["p_rel_error_mean"])
    asset_nn_error = float(asset_nn["p_rel_error_mean"].median())

    growth_kernel = load(out, "neoclassical_growth_kernel_baseline")
    growth_uno = load(out, "neoclassical_growth_kernel_uno_baseline")
    growth_sequential_nn = accepted("neoclassical_growth_nn")
    growth_recursive_nn = accepted("neoclassical_growth_nn_recursive")
    growth_kernel_time = med(growth_kernel["wall_time"])
    growth_kernel_error = max(
        med(growth_kernel["k_rel_error_mean"]),
        med(growth_kernel["c_rel_error_mean"]),
    )
    growth_uno_time = med(growth_uno["wall_time"])
    growth_uno_error = max(
        med(growth_uno["k_rel_error_mean"]),
        med(growth_uno["c_rel_error_mean"]),
    )

    # Both NN formulations against the same sequence-space kernel baseline: the
    # sequential MLP shares the kernel's t -> (k, c) ansatz, the recursive one
    # learns a state-space policy rolled forward by RK4.
    def growth_nn_stats(frame):
        return (
            float(frame["train_time"].median()),
            max(
                float(frame["test_mean_abs_k_rel_error"].median()),
                float(frame["test_mean_abs_c_rel_error"].median()),
            ),
        )

    open_kernel = load(out, "open_economy_growth_kernel_baseline")
    open_uno = load(out, "open_economy_growth_kernel_uno_baseline")
    open_sequential_nn = accepted("open_economy_growth_nn")
    open_recursive_nn = accepted("open_economy_growth_nn_recursive")
    open_kernel_time = med(open_kernel["wall_time"])
    open_uno_time = med(open_uno["wall_time"])
    open_kernel_error = med(open_kernel["max_benchmark_error"])
    open_uno_error = med(open_uno["max_benchmark_error"])

    def open_nn_stats(frame):
        return (
            float(frame["train_time"].median()),
            float(frame["max_benchmark_error"].median()),
        )

    multicountry_kernel = load(out, "multicountry_growth_kernel_initial_debt")
    multicountry_nn_frame = accepted("multicountry_growth_nn")
    multicountry_kernel_time = med(multicountry_kernel["wall_time"])
    multicountry_nn_time = float(multicountry_nn_frame["train_time"].median())
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

    return {
        "timing_contract": {
            "kernel": "median of five cached deterministic solves",
            "uno": "median of five cached deterministic constrained solves",
            "nn": (
                "median across accepted seeds from 100 initializations "
                "(21 for the open-economy sequence-space NN, five for "
                "multicountry growth), each seed allowed up to five "
                "multi-start restarts"
            ),
        },
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
                "sequential_nn": growth_nn_stats(growth_sequential_nn),
                "recursive_nn": growth_nn_stats(growth_recursive_nn),
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


def validate_replication_contract(out):
    nn_seed_counts = {
        "asset_pricing_nn": 100,
        "neoclassical_growth_nn": 100,
        "neoclassical_growth_nn_recursive": 100,
        "open_economy_growth_nn": 21,
        "open_economy_growth_nn_recursive": 100,
        "multicountry_growth_nn": 5,
    }
    for prefix, expected in nn_seed_counts.items():
        summary = load(out, f"{prefix}_baseline")
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
        "open_economy_growth_kernel_uno",
        "multicountry_growth_kernel",
        "asset_pricing_kernel_uno",
        "neoclassical_growth_kernel_uno",
    )
    for prefix in kernel_prefixes:
        timing = pd.read_parquet(out / f"{prefix}_baseline_timing_df.parquet")
        if len(timing) != 5:
            raise RuntimeError(f"{prefix} must retain five timing runs")


def collect(output_dir: str = ".results"):
    out = pathlib.Path(output_dir)
    validate_replication_contract(out)
    write_nu_rho_table(out)
    payload = clean(
        {
            "_meta": {
                "description": (
                    "Authoritative source for every number embedded in the "
                    "paper: each value is selected and rounded to copy/paste "
                    "form here (replication/collect_json.py) and must never be "
                    "recomputed or re-rounded downstream. Cross-seed statistics "
                    "are medians (suffix _median); deterministic wall_time values "
                    "are medians of five cached solves; error magnitudes below "
                    "0.001 are LaTeX math-mode strings from sci()."
                ),
                "generated": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "machine": machine_info(),
                "output_dir": output_dir,
                "band_quantiles": [int(100 * QUANTILES[0]), int(100 * QUANTILES[-1])],
            },
            "neoclassical_growth_kernel": neoclassical_growth_kernel(out),
            "neoclassical_growth_kernel_uno": neoclassical_growth_kernel_uno(out),
            "neoclassical_growth_concave_convex_kernel": (
                neoclassical_growth_concave_convex_kernel(out)
            ),
            "neoclassical_growth_concave_convex_kernel_uno": (
                neoclassical_growth_concave_convex_kernel(
                    out,
                    prefix="neoclassical_growth_concave_convex_kernel_uno",
                    require_all=False,
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
            "asset_pricing_kernel_uno": asset_pricing_kernel(
                out, prefix="asset_pricing_kernel_uno"
            ),
            "asset_pricing_nn": asset_pricing_nn(out),
            "optimal_advertising_kernel": optimal_advertising_kernel(out),
            "human_capital_kernel": human_capital_kernel(out),
            "neoclassical_growth_nn": neoclassical_growth_nn(out),
            "neoclassical_growth_nn_recursive": neoclassical_growth_nn_recursive(out),
            "neoclassical_growth_kernel_recursive": neoclassical_growth_kernel_recursive(
                out
            ),
            "open_economy_growth_kernel": open_economy_growth_kernel(out),
            "open_economy_growth_kernel_uno": open_economy_growth_kernel(
                out,
                prefix="open_economy_growth_kernel_uno",
                selection="minimum_rkhs_norm_constrained",
            ),
            "open_economy_growth_nn": open_economy_growth_nn_recursive(
                out, prefix="open_economy_growth_nn"
            ),
            "open_economy_growth_nn_recursive": open_economy_growth_nn_recursive(out),
            "multicountry_growth_kernel": multicountry_growth_kernel(out),
            "multicountry_growth_nn": multicountry_growth_nn(out),
            "performance_comparisons": performance_comparisons(out),
        }
    )
    # allow_nan=False guarantees the output is strict JSON (any NaN raises here).
    (out / "all_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {out / 'all_results.json'}")


if __name__ == "__main__":
    jsonargparse.CLI(collect)
