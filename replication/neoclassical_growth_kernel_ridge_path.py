from __future__ import annotations

import math

import jsonargparse
import numpy as np
import pandas as pd

from models import neoclassical_growth_kernel_uno
from models.kernels import integrated_kernel_matrices, rkhs_norm_squared
from replication.neoclassical_growth_kernel import PREFIX, SOLVER
from replication.utilities import configure_process, run_solver, save_experiment

# Ridge path by kernel on the equally spaced baseline dates: the ridge solves
# of the robustness table at every weight, stopping on the ridge objective's
# gradient as in the ridge grid, and the constrained solve (exact collocation,
# minimum RKHS norm) that is their ridgeless limit. Nothing gates; the
# constrained row of a kernel whose Gram matrix is numerically singular at
# these settings is reported as it comes.
KERNELS = [
    ("matern12", 10.0),
    ("matern32", 10.0),
    ("matern52", 10.0),
    ("rbf", 2.0),
]
RIDGES = [1e-8, 1e-10, 1e-12, 1e-14]
GTOL_RELATIVE = 1e-4
GTOL_FLOOR = 1e-13
LM_OVERRIDES = dict(atol=1.0, max_steps=300)
TOLERANCE_OVERRIDES = dict(gate_train=False, gate_validation=False)
CONSTRAINED_SOLVER = neoclassical_growth_kernel_uno.neoclassical_growth_kernel_uno
COLUMNS = (
    "kernel_key",
    "kernel",
    "ell",
    "solver",
    "ridge",
    "test_k_rel_error_max",
    "test_c_rel_error_max",
    "test_k_rel_error_abs",
    "test_c_rel_error_abs",
    "max_train_residual",
    "mean_validation_residual",
    "rkhs_norm_squared",
    "gram_condition",
    "c_0",
    "final_grad_norm",
    "steps",
    "solve_time",
    "solver_status",
    "solution_accepted",
    "rejection_reason",
)


def kernel_key(kernel, ell):
    nu = {"matern12": "nu05", "matern32": "nu15", "matern52": "nu25", "rbf": "rbf"}
    return f"{nu[kernel]}_ell{ell:g}"


# Maxima over the test window excluding the pinned t = 0 point, as in
# collect_json.rel_error_extrema.
def rel_error_max(test_df):
    df = test_df[test_df["t"] > 0.0]
    return float(df["k_rel_error"].abs().max()), float(df["c_rel_error"].abs().max())


def run(output_dir: str = ".results"):
    configure_process()
    rows = []
    hyperparameters = None
    print(
        f"{'kernel':>12} {'ridge':>12} {'k max':>9} {'c max':>9} {'resid':>9} "
        f"{'norm2':>9} {'steps':>5} {'status':>18}"
    )
    for kernel, ell in KERNELS:
        kernel_set = dict(kernel=kernel, ell=ell)
        for ridge in RIDGES:
            results, extras = run_solver(
                SOLVER,
                dict(
                    kernel_set=kernel_set,
                    ridge_set=dict(ridge=ridge),
                    lm_set=dict(
                        gtol=max(GTOL_RELATIVE * ridge, GTOL_FLOOR), **LM_OVERRIDES
                    ),
                    tolerance_set=TOLERANCE_OVERRIDES,
                ),
            )
            if hyperparameters is None:
                hyperparameters = {
                    key: value
                    for key, value in results.items()
                    if isinstance(value, dict)
                }
                hyperparameters["kernel_set"] = None
                hyperparameters["ridge_set"]["ridge"] = None
                hyperparameters["lm_set"]["gtol"] = None
                hyperparameters["ridge_path"] = dict(
                    kernels=[dict(kernel=k, ell=e) for k, e in KERNELS],
                    ridges=RIDGES,
                    train_points=int(results["data_set"]["train_points"]),
                    train_T=results["data_set"]["train_T"],
                    test_T=results["data_set"]["test_T"],
                    lm_overrides=dict(
                        gtol_relative=GTOL_RELATIVE,
                        gtol_floor=GTOL_FLOOR,
                        **LM_OVERRIDES,
                    ),
                    tolerance_overrides=TOLERANCE_OVERRIDES,
                )
            theta = extras["result"].x
            K = extras["solve"]["residual_args"].K
            norm2 = float(
                sum(
                    rkhs_norm_squared(alpha, K)
                    for alpha in (theta.alpha_mu, theta.alpha_c, theta.alpha_k)
                )
            )
            k_max, c_max = rel_error_max(extras["test_df"])
            row = {
                "kernel_key": kernel_key(kernel, ell),
                "kernel": kernel,
                "ell": ell,
                "solver": "ridge",
                "ridge": ridge,
                "test_k_rel_error_max": k_max,
                "test_c_rel_error_max": c_max,
                "test_k_rel_error_abs": results["test_k_rel_error_abs"],
                "test_c_rel_error_abs": results["test_c_rel_error_abs"],
                "max_train_residual": results["max_train_residual"],
                "mean_validation_residual": results["mean_validation_residual"],
                "rkhs_norm_squared": norm2,
                "gram_condition": float(np.linalg.cond(np.asarray(K))),
                "c_0": results["c_0"],
                "final_grad_norm": float(extras["result"].info.grad_norm),
                "steps": results["steps"],
                "solve_time": results["solve_time"],
                "solver_status": results["solver_status"],
                "solution_accepted": results["solution_accepted"],
                "rejection_reason": results["rejection_reason"],
            }
            rows.append(row)
            print(
                f"{row['kernel_key']:>12} {ridge:>12.0e} {k_max:>9.2e} {c_max:>9.2e} "
                f"{row['max_train_residual']:>9.2e} {norm2:>9.4f} "
                f"{row['steps']:>5} {row['solver_status']:>18}"
            )
        results, extras = run_solver(
            CONSTRAINED_SOLVER,
            dict(kernel_set=kernel_set, tolerance_set=TOLERANCE_OVERRIDES),
        )
        t = extras["train_df"]["t"].to_numpy()
        K, _ = integrated_kernel_matrices(t, t, kernel, 1.0, ell)
        k_max, c_max = rel_error_max(extras["test_df"])
        row = {
            "kernel_key": kernel_key(kernel, ell),
            "kernel": kernel,
            "ell": ell,
            "solver": "constrained",
            "ridge": math.nan,
            "test_k_rel_error_max": k_max,
            "test_c_rel_error_max": c_max,
            "test_k_rel_error_abs": results["test_k_rel_error_abs"],
            "test_c_rel_error_abs": results["test_c_rel_error_abs"],
            "max_train_residual": results["max_train_residual"],
            "mean_validation_residual": results["mean_validation_residual"],
            "rkhs_norm_squared": results["rkhs_norm_squared"],
            "gram_condition": float(np.linalg.cond(np.asarray(K))),
            "c_0": results["c_0"],
            "final_grad_norm": math.nan,
            "steps": -1,
            "solve_time": results["solve_time"],
            "solver_status": f"{results['solver_status']}/{results['solution_status']}",
            "solution_accepted": results["solution_accepted"],
            "rejection_reason": results["rejection_reason"],
        }
        rows.append(row)
        print(
            f"{row['kernel_key']:>12} {'constrained':>12} {k_max:>9.2e} {c_max:>9.2e} "
            f"{row['max_train_residual']:>9.2e} {row['rkhs_norm_squared']:>9.4f} "
            f"{'':>5} {row['solver_status']:>18}"
        )
    results_df = pd.DataFrame(rows, columns=COLUMNS)
    save_experiment(
        "ridge_path",
        results_df,
        {},
        output_dir,
        dict(n_seeds=1, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    print(f"ridge_path -> {output_dir}/ ({len(results_df)} solves)")


if __name__ == "__main__":
    jsonargparse.CLI(run)
