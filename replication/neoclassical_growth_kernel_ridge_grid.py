from __future__ import annotations

import jsonargparse
import numpy as np
import pandas as pd

from models.kernels import rkhs_norm_squared
from replication.neoclassical_growth_kernel import PREFIX, SOLVER
from replication.utilities import configure_process, run_solver, save_experiment

# Ridge weight x collocation count grid on the baseline growth model: the
# ridge is the fixed weight on the RKHS norm in every solve (no continuation),
# dates are i.i.d. uniform on [0, train_T] with t = 0 pinned, and each
# (ridge, N) cell is an ensemble of draws. The solve stops on the ridge
# objective's gradient alone (atol is never binding on the 1/sqrt(3N)-scaled
# residual): gtol = 1e-4 * ridge resolves the min-norm selection to ~1e-3
# relative accuracy for a solution seminorm of order one (nlls-gram's
# calibration), floored at 1e-13, the float64 roundoff floor of the
# gradient, which binds at ridge <= 1e-9. Nothing gates, so every solve is
# reported as data.
RIDGES = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
TRAIN_POINTS = [11, 21, 51, 101, 201]
N_SEEDS = 7
GTOL_RELATIVE = 1e-4
GTOL_FLOOR = 1e-13
LM_OVERRIDES = dict(atol=1.0, max_steps=300)
TOLERANCE_OVERRIDES = dict(gate_train=False, gate_validation=False)
COLUMNS = (
    "ridge",
    "train_points",
    "seed",
    "test_k_rel_error_abs",
    "test_c_rel_error_abs",
    "k_distance_to_smallest_ridge",
    "c_distance_to_smallest_ridge",
    "k_distance_to_next_ridge",
    "c_distance_to_next_ridge",
    "max_train_residual",
    "rkhs_norm_squared",
    "gram_condition",
    "final_grad_norm",
    "penalty_grad_norm",
    "steps",
    "solve_time",
    "solver_status",
    "solution_accepted",
    "rejection_reason",
)


def run(output_dir: str = ".results", train_T: float = 100.0):
    configure_process()
    rows = []
    hyperparameters = None
    print(
        f"{'ridge':>8} {'N':>4} {'seed':>4} {'k err':>9} {'resid':>9} "
        f"{'norm2':>9} {'steps':>5} {'status':>10}"
    )
    for n in TRAIN_POINTS:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            dates = [0.0] + sorted(rng.uniform(0.0, train_T, n - 1).tolist())
            # Solves at every ridge on the same dates, smallest ridge first:
            # the paired distances are to the smallest-ridge solve and to the
            # next smaller ridge on the grid (zero at the smallest ridge).
            paths = {}
            for previous, ridge in zip([None, *sorted(RIDGES)], sorted(RIDGES)):
                results, extras = run_solver(
                    SOLVER,
                    dict(
                        data_set=dict(train_T=train_T, train_points_list=dates),
                        ridge_set=dict(ridge=ridge),
                        lm_set=dict(
                            gtol=max(GTOL_RELATIVE * ridge, GTOL_FLOOR),
                            **LM_OVERRIDES,
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
                    hyperparameters["data_set"]["train_points_list"] = None
                    hyperparameters["ridge_set"]["ridge"] = None
                    hyperparameters["lm_set"]["gtol"] = None
                    hyperparameters["ridge_grid"] = dict(
                        train_T=train_T,
                        ridges=RIDGES,
                        train_points=TRAIN_POINTS,
                        n_seeds=N_SEEDS,
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
                test_df = extras["test_df"]
                paths[ridge] = (test_df["k"].to_numpy(), test_df["c"].to_numpy())
                k_ref, c_ref = paths[min(RIDGES)]
                k_next, c_next = paths[previous if previous is not None else ridge]
                info = extras["result"].info
                row = {
                    "ridge": ridge,
                    "train_points": n,
                    "seed": seed,
                    "test_k_rel_error_abs": results["test_k_rel_error_abs"],
                    "test_c_rel_error_abs": results["test_c_rel_error_abs"],
                    "k_distance_to_smallest_ridge": float(
                        np.mean(np.abs(paths[ridge][0] - k_ref) / np.abs(k_ref))
                    ),
                    "c_distance_to_smallest_ridge": float(
                        np.mean(np.abs(paths[ridge][1] - c_ref) / np.abs(c_ref))
                    ),
                    "k_distance_to_next_ridge": float(
                        np.mean(np.abs(paths[ridge][0] - k_next) / np.abs(k_next))
                    ),
                    "c_distance_to_next_ridge": float(
                        np.mean(np.abs(paths[ridge][1] - c_next) / np.abs(c_next))
                    ),
                    "max_train_residual": results["max_train_residual"],
                    "rkhs_norm_squared": norm2,
                    "gram_condition": float(np.linalg.cond(np.asarray(K))),
                    "final_grad_norm": float(info.grad_norm),
                    "penalty_grad_norm": float(info.penalty_grad_norm),
                    "steps": results["steps"],
                    "solve_time": results["solve_time"],
                    "solver_status": results["solver_status"],
                    "solution_accepted": results["solution_accepted"],
                    "rejection_reason": results["rejection_reason"],
                }
                rows.append(row)
                print(
                    f"{ridge:>8.0e} {n:>4} {seed:>4} "
                    f"{row['test_k_rel_error_abs']:>9.2e} "
                    f"{row['max_train_residual']:>9.2e} {norm2:>9.3e} "
                    f"{row['steps']:>5} {row['solver_status']:>10}"
                )
    results_df = pd.DataFrame(rows, columns=COLUMNS)
    save_experiment(
        "ridge_grid",
        results_df,
        {},
        output_dir,
        dict(n_seeds=N_SEEDS, residual_dtype="float64"),
        prefix=PREFIX,
        hyperparameters=hyperparameters,
    )
    print(f"ridge_grid -> {output_dir}/ ({len(results_df)} solves)")


if __name__ == "__main__":
    jsonargparse.CLI(run)
