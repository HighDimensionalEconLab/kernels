import unittest

import pandas as pd

from extra.replication.neoclassical_growth_recursive_comparison import (
    apply_common_validation,
    method_summary,
    wide_comparison,
)


class RecursiveGrowthComparisonTests(unittest.TestCase):
    def test_common_gate_rejects_bad_benchmark_path_for_either_method(self):
        results = pd.DataFrame(
            {
                "solution_accepted": [True, True],
                "solver_converged": [True, True],
                "max_train_euler": [1e-7, 1e-7],
                "test_path_ok": [True, True],
                "transversality_ok": [True, True],
                "test_mean_abs_k_rel_error": [1e-5, 1e-1],
                "test_mean_abs_c_rel_error": [1e-5, 1e-1],
                "solve_time": [0.2, 0.3],
                "total_fit_time": [0.25, 0.35],
                "test_pooled_median_abs_k_rel_error": [1e-5, 1e-1],
                "test_pooled_median_abs_c_rel_error": [1e-5, 1e-1],
                "test_median_trajectory_mean_abs_k_rel_error": [2e-5, 2e-1],
                "test_median_trajectory_mean_abs_c_rel_error": [2e-5, 2e-1],
                "test_euler_mean_abs": [1e-7, 1e-7],
            }
        )
        validated = apply_common_validation(results, 2e-5, 5e-3)

        self.assertListEqual(
            validated["common_validation_accepted"].tolist(), [True, False]
        )
        summary = method_summary("recursive_nn", validated)
        self.assertEqual(summary["frac_accepted"], 0.5)
        self.assertEqual(summary["total_fit_time"], 0.25)

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


if __name__ == "__main__":
    unittest.main()
