import unittest

import jax
import jax.numpy as jnp
import numpy as np

from experiments.multicountry_trade_growth_lls_testbed import (
    LinearSolveSettings,
    RandomCaseSettings,
    build_linearized_lls_case,
    dense_cholesky,
    random_trade_growth_inputs,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class MulticountryTradeGrowthLLSTestbedTests(unittest.TestCase):
    def test_random_cases_are_reproducible_and_clear_world_assets(self):
        settings = RandomCaseSettings(num_countries=4, seed=17)
        first = random_trade_growth_inputs(settings)
        second = random_trade_growth_inputs(settings)

        self.assertEqual(first, second)
        parameters, k_0, a_0 = first
        populations = np.asarray(parameters.populations)
        self.assertEqual(parameters.num_countries, 4)
        self.assertTrue(np.all(np.asarray(k_0) > 0.0))
        self.assertLess(abs(float(populations @ np.asarray(a_0))), 1e-14)
        self.assertTrue(np.all(np.diag(parameters.trade_costs) == 1.0))
        self.assertTrue(
            np.all(np.asarray(parameters.trade_costs)[~np.eye(4, dtype=bool)] > 1.0)
        )

    def test_linearized_operator_layout_and_dense_reference(self):
        case = build_linearized_lls_case(
            RandomCaseSettings(
                num_countries=2,
                train_points=3,
                train_T=10.0,
                seed=3,
            )
        )
        self.assertEqual(case.residual_size, (12 * 2 + 1) * 3)
        self.assertEqual(case.parameter_size, case.residual_size + 2 * 2)
        self.assertEqual(case.residual_time_block_size, 12 * 2 + 1)
        self.assertEqual(case.unknown_layout.dynamic_function_count, 8)
        self.assertEqual(case.unknown_layout.total_size, case.parameter_size)

        residual_time_major = case.to_time_major(case.residual)
        np.testing.assert_allclose(
            case.to_native_order(residual_time_major),
            case.residual,
            rtol=0.0,
            atol=0.0,
        )

        linear_matrix = case.dense_dual_matrix()
        vector = jax.random.normal(
            jax.random.key(11), (case.residual_size,), dtype=jnp.float64
        )
        np.testing.assert_allclose(
            case.dual_matvec(vector),
            linear_matrix @ vector,
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            linear_matrix,
            linear_matrix.T,
            rtol=1e-11,
            atol=1e-11,
        )

        dual_solution, iterations, converged = dense_cholesky(
            case, case.residual, LinearSolveSettings()
        )
        relative_residual = jnp.linalg.norm(
            case.dual_matvec(dual_solution) - case.residual
        ) / jnp.linalg.norm(case.residual)
        self.assertEqual(int(iterations), 1)
        self.assertTrue(bool(converged))
        self.assertLess(float(relative_residual), 1e-8)


if __name__ == "__main__":
    unittest.main()
