from __future__ import annotations

import inspect
import json
import pathlib

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

QUANTILES = [0.1, 0.5, 0.9]
DETERMINISTIC_TIMING_REPETITIONS = 1

# Shared kinked-production basin experiment: the k_0 grid crosses the
# concave-convex threshold and spans both basins of attraction.
BASIN_K0_GRID = np.linspace(0.5, 4.0, 15)
RESIDUAL_NORM_COLUMNS = (
    "ode_residual_rms",
    "algebraic_residual_rms",
    "dae_residual_rms",
)


def compare_residual_norm_frames(kernel_df, nn_df, quality_T):
    kernel_quality = kernel_df.loc[kernel_df["t"] <= quality_T].reset_index(drop=True)
    nn_quality = nn_df.loc[nn_df["t"] <= quality_T].reset_index(drop=True)
    if not np.array_equal(kernel_quality["t"].to_numpy(), nn_quality["t"].to_numpy()):
        raise ValueError("kernel and NN quality grids must contain identical dates")

    comparison = pd.DataFrame({"t": kernel_quality["t"]})
    summary = {"quality_T": quality_T, "quality_point_count": len(comparison)}
    for name in RESIDUAL_NORM_COLUMNS:
        kernel_values = kernel_quality[name].to_numpy()
        nn_values = nn_quality[name].to_numpy()
        pointwise_ratio = np.divide(
            nn_values,
            kernel_values,
            out=np.full_like(nn_values, np.inf),
            where=kernel_values > 0.0,
        )
        total = nn_values + kernel_values
        symmetric_gap = np.divide(
            2.0 * np.abs(nn_values - kernel_values),
            total,
            out=np.zeros_like(total),
            where=total > 0.0,
        )
        comparison[f"kernel_{name}"] = kernel_values
        comparison[f"nn_{name}"] = nn_values
        comparison[f"nn_to_kernel_{name}_ratio"] = pointwise_ratio
        comparison[f"symmetric_relative_{name}_gap"] = symmetric_gap

        kernel_grid_rms = float(np.sqrt(np.mean(np.square(kernel_values))))
        nn_grid_rms = float(np.sqrt(np.mean(np.square(nn_values))))
        summary[f"kernel_quality_grid_{name}"] = kernel_grid_rms
        summary[f"nn_quality_grid_{name}"] = nn_grid_rms
        summary[f"nn_to_kernel_quality_grid_{name}_ratio"] = (
            nn_grid_rms / kernel_grid_rms if kernel_grid_rms > 0.0 else np.inf
        )
    return summary, comparison


def configure_process(cache_path=".jax_cache"):
    # Driver-side jax setup: the same always-on x64 + highest-matmul settings
    # every solver pins at its own entry (use_float64 only selects the working
    # dtype), plus the persistent compilation cache. Must run before the first
    # jax compilation; low thresholds force caching of our sub-second
    # compiles, which the defaults skip. Cache entries are keyed by program
    # hash, so both precisions coexist in one cache dir.
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", cache_path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def with_option(options, key, value):
    # Set a possibly dotted key ("data_set.train_points") as the nested dict
    # jsonargparse.parse_object expects, merging with any existing group.
    result = dict(options)
    if "." in key:
        head, rest = key.split(".", 1)
        result[head] = with_option(dict(result.get(head) or {}), rest, value)
    else:
        result[key] = value
    return result


# One solver call with the CLI's option semantics: nested dicts are
# instantiated into the driver's settings dataclasses by jsonargparse.
def run_solver(func, options=None):
    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    parser.add_function_arguments(func)
    cfg = parser.parse_object(options or {})
    return func(**parser.instantiate_classes(cfg).as_dict())


def ensemble(func, n_seeds, start_seed=0, options=None):
    options = options or {}
    stochastic = "seed" in inspect.signature(func).parameters
    if stochastic and n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    seeds = (
        range(start_seed, start_seed + n_seeds)
        if stochastic
        else range(DETERMINISTIC_TIMING_REPETITIONS)
    )

    rows = []
    frames: dict[str, list[pd.DataFrame]] = {}
    hyperparameters = None
    for seed in seeds:
        run_options = {**options, "seed": seed} if stochastic else options
        results, extra = run_solver(func, run_options)
        rows.append({**results, "seed": seed} if stochastic else dict(results))
        # Required of every solver (collect_json depends on the saved file); identical
        # across seeds (the runtime-folded settings), so the first suffices.
        if hyperparameters is None:
            hyperparameters = extra.pop("hyperparameters")
        for key, value in extra.items():
            if isinstance(value, pd.DataFrame):
                if stochastic:
                    frames.setdefault(key, []).append(value.assign(seed=seed))
                elif seed == 0:
                    frames.setdefault(key, []).append(value)
    results_df = pd.DataFrame(rows)
    if not stochastic:
        timing_columns = [
            name
            for name in results_df.select_dtypes("number")
            if name.endswith(("_time", "_seconds"))
        ]
        timing_df = results_df[timing_columns].copy()
        timing_df.insert(0, "timing_run", np.arange(len(timing_df)))
        frames["timing_df"] = [timing_df]
        representative = results_df.iloc[0].copy()
        for name in timing_columns:
            representative[name] = results_df[name].median()
        representative["timing_repetitions"] = DETERMINISTIC_TIMING_REPETITIONS
        results_df = pd.DataFrame([representative])
    frames_df = {key: pd.concat(dfs, ignore_index=True) for key, dfs in frames.items()}
    return results_df, frames_df, hyperparameters


def sweep(func, param, values, n_seeds=1, start_seed=0, options=None, retries=()):
    # Grid sweep: rows = grid x seeds, and every per-solve frame (test_df etc.)
    # is concatenated with the swept parameter as a column so figures can plot
    # full trajectories per grid point. param may be dotted
    # ("data_set.train_points"); the added column uses the leaf name. A solve
    # that raises becomes a rejected row rather than aborting the sweep.
    # retries: option-dict overrides tried in order when a grid point is
    # rejected (e.g. an alternative solver preset); the first fully accepted
    # attempt wins, otherwise the last attempt's rows are kept.
    options = options or {}
    column = param.split(".")[-1]
    rows, frames = [], {}
    hyperparameters = None
    for value in values:
        try:
            for extra in ({}, *retries):
                results_df, frames_df, run_hyperparameters = ensemble(
                    func,
                    n_seeds,
                    start_seed,
                    with_option({**options, **extra}, param, value),
                )
                if results_df["solution_accepted"].all():
                    break
        except Exception as exc:  # surface the failure as data, keep sweeping
            print(f"sweep {param}={value}: exception {type(exc).__name__}: {exc}")
            rows.append(
                pd.DataFrame(
                    [
                        {
                            column: value,
                            "solution_accepted": False,
                            "rejection_reason": f"exception:{type(exc).__name__}",
                        }
                    ]
                )
            )
            continue
        results_df[column] = value
        rows.append(results_df)
        for key, df in frames_df.items():
            frames.setdefault(key, []).append(df.assign(**{column: value}))
        if hyperparameters is None:
            hyperparameters = run_hyperparameters
    results = pd.concat(rows, ignore_index=True)
    frames_out = {key: pd.concat(dfs, ignore_index=True) for key, dfs in frames.items()}
    return results, frames_out, hyperparameters


def summary_stats(series, collapse_tol=None):
    stats = {
        "median": float(series.median()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "q10": float(series.quantile(0.1)),
        "q90": float(series.quantile(0.9)),
    }
    if collapse_tol is not None:
        # Collapse to a single value when the series is constant up to numerical
        # noise (std at/below ~sqrt(machine eps), scaled by magnitude) or
        # degenerate (NaN std, e.g. an all-NaN column or a single row). The
        # median is the representative value. `not (std > tol)` catches NaN.
        std, median = stats["std"], stats["median"]
        if not (std > collapse_tol * max(1.0, abs(median))):
            return median
    return stats


def save_experiment(
    name, results_df, frames, output_dir, meta, prefix, hyperparameters=None
):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{name}"
    results_df.to_parquet(out / f"{stem}_results.parquet")
    for key, df in frames.items():
        df.to_parquet(out / f"{stem}_{key}.parquet")
    if hyperparameters is not None:
        (out / f"{stem}_hyperparameters.json").write_text(
            json.dumps(hyperparameters, indent=2)
        )

    numeric = results_df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    collapse_tol = float(np.sqrt(np.finfo(np.float64).eps))
    col_stats = {
        col: summary_stats(results_df[col], collapse_tol) for col in numeric.columns
    }
    header = {
        "experiment": name,
        **meta,
        "n_accepted": int(results_df["solution_accepted"].sum()),
        "frac_accepted": float(results_df["solution_accepted"].mean()),
    }
    # Full distributions first, then the collapsed constant scalars at the end.
    expanded = {k: v for k, v in col_stats.items() if isinstance(v, dict)}
    collapsed = {k: v for k, v in col_stats.items() if not isinstance(v, dict)}
    scalars = {**header, **expanded, **collapsed}
    (out / f"{stem}.json").write_text(json.dumps(scalars, indent=2))


def write_summary(output_dir, prefix, experiments=None):
    out = pathlib.Path(output_dir)
    summary = {}
    for jf in sorted(out.glob(f"{prefix}_*.json")):
        if jf.name == f"{prefix}_summary.json" or jf.name.endswith(
            "_hyperparameters.json"
        ):
            continue
        d = json.loads(jf.read_text())
        experiment = d.get("experiment")
        if experiment is None or jf.name != f"{prefix}_{experiment}.json":
            continue
        if experiments is not None and d["experiment"] not in experiments:
            continue
        summary[d["experiment"]] = d
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))


# --- kinked (concave-convex) production geometry ------------------------------


def kinked_thresholds(a=1 / 3, delta=0.1, rho_hat=0.11, A=0.5, b_1=3.0, b_2=2.5):
    # Branch switch k_thr where k^a = b_1 k^a - b_2, and the per-branch steady
    # states from f'(k_ss) = delta + rho_hat (ex-post classification only).
    k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a)
    k_ss_low = ((delta + rho_hat) / (A * a)) ** (1.0 / (a - 1.0))
    k_ss_high = ((delta + rho_hat) / (A * a * b_1)) ** (1.0 / (a - 1.0))
    return k_thr, k_ss_low, k_ss_high


def classify_path_basin(k_path, k_0, k_thr, k_ss_low, k_ss_high, tol_frac=0.05):
    # Finite-horizon solutions (kernel and sequential-NN paths to test_T) may
    # end mid-transition near the threshold, so a settle-ball test alone cannot
    # apply. A path that ends inside the ball (tol_frac of the kink) around a
    # steady state counts as settled there; otherwise classify by the side of
    # the threshold the path ends on AND whether its tail is still moving
    # toward that side's steady state (a diverging tail fails).
    k_path = np.asarray(k_path, dtype=float)
    k_end, k_prev = float(k_path[-1]), float(k_path[-2])
    tol = tol_frac * k_thr
    if abs(k_end - k_ss_low) < tol:
        ends_high, resolved = False, True
    elif abs(k_end - k_ss_high) < tol:
        ends_high, resolved = True, True
    else:
        ends_high = k_end > k_thr
        target = k_ss_high if ends_high else k_ss_low
        resolved = abs(k_end - target) < abs(k_prev - target)
    basin_correct = bool(resolved and ends_high == (k_0 > k_thr))
    false_positive = bool(resolved and not basin_correct)
    return {
        "basin_correct": basin_correct,
        "false_positive": false_positive,
        "basin_failed": not (basin_correct or false_positive),
    }


# --- plotting -------------------------------------------------------------------


def style(figsize=(15, 10), fontsize=14, ticksize=14):
    # Mathtext (not usetex) matches the committed QE figures and avoids
    # LaTeX-rendering quirks in log-axis exponents.
    plt.rcParams.update(
        {
            "font.family": "serif",
            # ASCII hyphen for minus signs: some serif fonts lack U+2212, which
            # otherwise drops minus signs from tick labels.
            "axes.unicode_minus": False,
            "figure.figsize": figsize,
            "figure.dpi": 80,
            "figure.edgecolor": "k",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
        }
    )


def dedup_legend(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend(by.values(), by.keys(), **kw)


def add_zoom_inset(ax, t, lines, center, zoom_loc, width_frac=0.01, zoom=3):
    # Inset zoom over index window zoom_loc = [i0, i1] of the plotted arrays,
    # centered vertically on the benchmark series (QE figure style).
    i0, i1 = zoom_loc
    ave_value = 0.5 * (center[i0] + center[i1])
    window_width = width_frac * ave_value
    axins = zoomed_inset_axes(
        ax,
        zoom,
        loc="center",
        bbox_to_anchor=(0.5, 0.7, -0.3, -0.3),
        bbox_transform=ax.transAxes,
    )
    lo = max(i0 - 1, 0)
    for series, kwargs in lines:
        axins.plot(t[lo : i1 + 1], series[lo : i1 + 1], **kwargs)
    axins.set_xlim(t[i0], t[i1])
    axins.set_ylim(ave_value - window_width, ave_value + window_width)
    axins.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axins.tick_params(axis="y", labelsize=8)
    mark_inset(ax, axins, loc1=1, loc2=3, linewidth=0.7, ls="--", ec="0.5")


def plot_neoclassical_growth(
    test_df,
    train_T,
    output_path,
    k_rel_error_ylim=(1e-6, 2e-2),
    c_rel_error_ylim=(1e-6, 2e-2),
    zoom=True,
    zoom_loc=(50, 60),
    fontsize=14,
    ticksize=14,
):
    # 2x2 QE growth figure: (k, k rel error, c, c rel error) vs benchmark with
    # the train/extrapolation boundary and optional zoom insets.
    style(figsize=(15, 10), fontsize=fontsize, ticksize=ticksize)
    t = test_df["t"].to_numpy()
    k_hat = test_df["k"].to_numpy()
    c_hat = test_df["c"].to_numpy()
    k_benchmark = test_df["k_benchmark"].to_numpy()
    c_benchmark = test_df["c_benchmark"].to_numpy()

    plt.figure(figsize=(15, 10))
    ax_capital = plt.subplot(2, 2, 1)
    plt.plot(t, k_hat, color="k", label=r"$\hat{x}(t)$: Kernel Approximation")
    plt.plot(
        t, k_benchmark, linestyle="--", color="k", label=r"$x(t)$: Benchmark Solution"
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.ylabel("Capital: $x(t)$")
    plt.xlabel("Time")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(
        t,
        test_df["k_rel_error"].to_numpy(),
        color="k",
        label=r"$\varepsilon_x(t)$: Relative Errors for $x(t)$",
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.yscale("log")
    plt.ylim(*k_rel_error_ylim)
    plt.xlabel("Time")
    plt.legend()

    ax_consumption = plt.subplot(2, 2, 3)
    plt.plot(t, c_hat, color="b", label=r"$\hat{y}(t)$: Kernel Approximation")
    plt.plot(
        t, c_benchmark, linestyle="--", color="b", label=r"$y(t)$: Benchmark Solution"
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.ylabel("Consumption: $y(t)$")
    plt.xlabel("Time")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(
        t,
        test_df["c_rel_error"].to_numpy(),
        color="b",
        label=r"$\varepsilon_y(t)$: Relative Errors for $y(t)$",
    )
    plt.axvline(
        x=train_T, color="k", linestyle=":", label="Extrapolation/Interpolation"
    )
    plt.yscale("log")
    plt.ylim(*c_rel_error_ylim)
    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    if zoom:
        add_zoom_inset(
            ax_capital,
            t,
            [(k_hat, dict(color="k")), (k_benchmark, dict(linestyle="--", color="k"))],
            k_benchmark,
            zoom_loc,
        )
        add_zoom_inset(
            ax_consumption,
            t,
            [(c_hat, dict(color="b")), (c_benchmark, dict(linestyle="--", color="b"))],
            c_benchmark,
            zoom_loc,
        )
    plt.savefig(output_path, format="pdf")
    plt.close()
