import unittest

import pandas as pd

from extra.replication.neoclassical_growth_recursive_comparison import (
    apply_common_validation,
    method_summary,
    wide_comparison,
)


class RecursiveGrowthComparisonTests(unittest.TestCase):
    def test_common_gate_reports_but_does_not_reject_bad_diagnostics(self):
        results = pd.DataFrame(
            {
                "solution_accepted": [True, True],
                "solver_converged": [True, True],
                "max_train_euler": [1e-7, 1e-2],
                "max_test_euler": [1e-6, 1e-1],
                "test_path_ok": [True, False],
                "transversality_ok": [True, False],
                "test_mean_abs_k_rel_error": [1e-5, 1e-1],
                "test_mean_abs_c_rel_error": [1e-5, 1e-1],
                "solve_time": [0.2, 0.3],
                "total_fit_time": [0.25, 0.35],
                "test_pooled_median_abs_k_rel_error": [1e-5, 1e-1],
                "test_pooled_median_abs_c_rel_error": [1e-5, 1e-1],
                "test_median_trajectory_mean_abs_k_rel_error": [2e-5, 2e-1],
                "test_median_trajectory_mean_abs_c_rel_error": [2e-5, 2e-1],
                "test_euler_mean_abs": [1e-7, 1e-2],
            }
        )
        validated = apply_common_validation(results)

        # Bad Euler residuals, paths, transversality, and benchmark errors are
        # reported columns; only solver convergence gates acceptance.
        self.assertListEqual(
            validated["common_validation_accepted"].tolist(), [True, True]
        )
        summary = method_summary("recursive_nn", validated)
        self.assertEqual(summary["frac_accepted"], 1.0)
        self.assertEqual(summary["max_test_euler"], (1e-6 + 1e-1) / 2)

    def test_common_gate_rejects_a_nonconverged_solve(self):
        results = pd.DataFrame(
            {
                "solution_accepted": [True, False],
                "solver_converged": [True, False],
                "max_train_euler": [1e-7, 1e-7],
                "max_test_euler": [1e-6, 1e-6],
                "test_path_ok": [True, True],
                "transversality_ok": [True, True],
                "test_mean_abs_k_rel_error": [1e-5, 1e-5],
                "test_mean_abs_c_rel_error": [1e-5, 1e-5],
            }
        )

        validated = apply_common_validation(results)

        self.assertListEqual(
            validated["common_validation_accepted"].tolist(), [True, False]
        )
        self.assertListEqual(
            validated["native_solution_accepted"].tolist(), [True, False]
        )

    def test_wide_json_row_keeps_method_identities(self):
        nn = {
            "method": "recursive_nn",
            "solution_accepted": True,
            "num_runs": 3,
            "frac_accepted": 1.0,
            "optimizer_solve_time": 0.3,
            "total_fit_time": 0.4,
            "test_pooled_median_abs_k_rel_error": 1e-6,
            "test_pooled_median_abs_c_rel_error": 2e-6,
            "test_median_trajectory_mean_abs_k_rel_error": 2e-6,
            "test_median_trajectory_mean_abs_c_rel_error": 4e-6,
            "test_euler_mean_abs": 1e-7,
            "max_train_euler": 1e-7,
            "max_test_euler": 1e-6,
        }
        kernel = {
            **nn,
            "method": "fixed_recursive_kernel",
            "total_fit_time": 0.2,
            "test_median_trajectory_mean_abs_k_rel_error": 4e-6,
            "test_median_trajectory_mean_abs_c_rel_error": 8e-6,
        }
        wide = wide_comparison([nn, kernel], 3.0, 1.0)

        self.assertEqual(len(wide), 1)
        self.assertEqual(wide.loc[0, "nn_total_fit_time"], 0.4)
        self.assertEqual(wide.loc[0, "kernel_total_fit_time"], 0.2)
        self.assertEqual(wide.loc[0, "kernel_to_nn_fit_time_ratio"], 0.5)
        self.assertEqual(wide.loc[0, "kernel_to_nn_k_error_ratio"], 2.0)
        self.assertNotIn("method", wide.columns)

    def test_common_gate_rejects_an_incomplete_collocation_path(self):
        results = pd.DataFrame(
            {
                "solution_accepted": [True, True],
                "solver_converged": [True, True],
                "collocation_path_ok": [True, False],
                "max_train_euler": [1e-7, 1e-7],
                "max_test_euler": [1e-6, 1e-6],
                "test_path_ok": [True, True],
                "transversality_ok": [True, True],
                "test_mean_abs_k_rel_error": [1e-5, 1e-5],
                "test_mean_abs_c_rel_error": [1e-5, 1e-5],
            }
        )

        validated = apply_common_validation(results)

        self.assertListEqual(
            validated["common_validation_accepted"].tolist(), [True, False]
        )


if __name__ == "__main__":
    unittest.main()
