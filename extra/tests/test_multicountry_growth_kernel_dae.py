import unittest

import jax
import jax.numpy as jnp
import numpy as np

from extra.multicountry_growth_kernel_dae import (
    multicountry_dae_path_values,
    multicountry_growth_kernel_dae,
)
from multicountry_growth import (
    MulticountryParameters,
    multicountry_reference_mu,
    multicountry_steady_state,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class MulticountryGrowthKernelDAETests(unittest.TestCase):
    def test_steady_state_residuals(self):
        parameters = MulticountryParameters(
            populations=(1.0, 2.0, 1.5),
            productivities=(1.0, 1.1, 0.9),
        )
        mu = multicountry_reference_mu(parameters)
        steady = multicountry_steady_state(mu, parameters)
        zeros = jnp.zeros((1, parameters.num_countries))
        values = multicountry_dae_path_values(
            k=steady["k"][None, :],
            a=steady["a"][None, :],
            q=steady["q"][None, :],
            mu=steady["marginal_wealth"][None, :],
            consumption=steady["c"][None, :],
            investment_rate=(steady["x"] / steady["k"])[None, :],
            interest_rate=jnp.asarray((parameters.r,)),
            dlog_k_dt=zeros,
            da_dt=zeros,
            dlog_q_dt=zeros,
            dlog_mu_dt=zeros,
            dlog_consumption_dt=zeros,
            dlog_investment_rate_dt=zeros,
            parameters=parameters,
        )
        for name in (
            "capital",
            "assets",
            "tobin_q",
            "marginal_wealth_euler",
            "consumption_foc",
            "investment_foc",
            "resource",
            "resource_drift",
            "asset_clearing",
        ):
            np.testing.assert_allclose(values[name], 0.0, atol=1e-14)

    def test_two_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.4),
            productivities=(1.05, 0.95),
        )
        results, extras = multicountry_growth_kernel_dae(parameters=parameters)
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(
            results["equation_diagnostics_pass"],
            results["equation_diagnostic_reason"],
        )
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["stopping_reason"], "converged")
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(
            results["differential_residual_count"],
            8 * results["num_train_points"],
        )
        self.assertEqual(
            results["country_algebraic_residual_count"],
            4 * results["num_train_points"],
        )
        self.assertEqual(
            results["global_algebraic_residual_count"],
            results["num_train_points"] + 1,
        )
        self.assertEqual(
            results["dae_residual_count"],
            13 * results["num_train_points"] + 1,
        )
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 8)
        self.assertEqual(results["dynamic_rkhs_function_count"], 8)
        self.assertEqual(results["country_algebraic_rkhs_function_count"], 4)
        self.assertEqual(results["common_algebraic_rkhs_function_count"], 1)
        self.assertEqual(results["rkhs_function_count"], 13)
        self.assertEqual(results["free_initial_value_count"], 9)
        self.assertEqual(results["metric_selection_dimension"], 8)
        self.assertEqual(results["linear_solver"], "auto")
        self.assertEqual(results["selection"], "minimum_rkhs_norm_ridge")
        self.assertEqual(
            results["global_closure"],
            "explicit_assets_and_index_consistent_clearing",
        )
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        self.assertLess(results["max_consumption_foc_residual"], 5e-3)
        self.assertLess(results["max_investment_foc_residual"], 5e-3)
        self.assertLess(results["max_benchmark_error"], 2e-3)
        self.assertLess(
            abs(
                results["log_relative_weight"]
                - results["benchmark_log_relative_weight"]
            ),
            1e-5,
        )
        self.assertTrue((extras["test_df"][["c_1", "c_2"]] > 0.0).all().all())
        self.assertTrue((extras["test_df"][["s_1", "s_2"]] > 0.0).all().all())
        self.assertTrue((extras["test_df"]["interest_rate"] > 0.0).all())

    def test_three_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.3, 0.8),
            productivities=(1.1, 0.95, 1.0),
        )
        results, extras = multicountry_growth_kernel_dae(
            parameters=parameters,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(
            results["equation_diagnostics_pass"],
            results["equation_diagnostic_reason"],
        )
        self.assertFalse(results["benchmark_available"])
        self.assertEqual(results["num_countries"], 3)
        self.assertEqual(results["rkhs_function_count"], 19)
        self.assertEqual(results["free_initial_value_count"], 13)
        self.assertEqual(results["metric_selection_dimension"], 12)
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 12)
        # Countries 2 and 3 share a_0 but draw different k_0, so their
        # welfare weights must be positive and genuinely distinct (the old
        # equalities held only under identical initial capital).
        self.assertGreater(results["relative_weight_2"], 0.0)
        self.assertGreater(results["relative_weight_3"], 0.0)
        self.assertNotAlmostEqual(
            results["relative_weight_2"],
            results["relative_weight_3"],
            places=3,
        )
        self.assertIn("consumption_foc_3", extras["test_df"])
        self.assertIn("investment_foc_3", extras["test_df"])


if __name__ == "__main__":
    unittest.main()
