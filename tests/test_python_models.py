import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from nlls_gram import matern_state_space, metric_from_state_space

from asset_pricing_kernel import asset_pricing_kernel
from asset_pricing_kernel_nlls import DataSettings as APDataSettings
from asset_pricing_kernel_nlls import KernelSettings as APKernelSettings
from asset_pricing_kernel_nlls import (
    NLLSGramSettings,
    asset_pricing_kernel_nlls,
)
from human_capital_kernel import human_capital_initial_conditions, human_capital_kernel
from human_capital_kernel_nlls import human_capital_kernel_nlls
from kernels import (
    gaussian_kernel,
    integrated_kernel_matrices,
    matern_kernel_0p5,
    matern_kernel_1p5,
    matern_kernel_2p5,
    matern_kernel_matvec_pair,
    normalize_kernel,
)
from multicountry_growth import (
    MulticountryParameters,
    multicountry_allocations,
    multicountry_output_scale,
    multicountry_path_values,
    multicountry_reference_marginal_wealth,
    multicountry_steady_state,
)
from multicountry_growth_kernel_nlls import multicountry_growth_kernel_nlls
from multicountry_trade_growth import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    multicountry_trade_allocations,
    multicountry_trade_path_values,
    static_trade_equilibrium,
    symmetric_trade_growth_parameters,
)
from multicountry_trade_growth_kernel_nlls import (
    DataSettings as MulticountryTradeDataSettings,
)
from multicountry_trade_growth_kernel_nlls import (
    NLLSGramSettings as MulticountryTradeNLLSGramSettings,
)
from multicountry_trade_growth_kernel_nlls import (
    multicountry_trade_growth_kernel_nlls,
)
from multicountry_trade_growth_nn import DataSettings as MulticountryTradeNNDataSettings
from multicountry_trade_growth_nn import (
    NLLSGramSettings as MulticountryTradeNNNLLSGramSettings,
)
from multicountry_trade_growth_nn import (
    SequentialNNSettings,
    multicountry_trade_growth_nn,
)
from neoclassical_growth_concave_convex_kernel import (
    neoclassical_growth_concave_convex_kernel,
)
from neoclassical_growth_concave_convex_kernel_nlls import (
    neoclassical_growth_concave_convex_kernel_nlls,
)
from neoclassical_growth_kernel import (
    DataSettings as GrowthDataSettings,
)
from neoclassical_growth_kernel import (
    KernelSettings,
    neoclassical_growth_kernel,
)
from neoclassical_growth_kernel_nlls import (
    DataSettings as GrowthNLLSDataSettings,
)
from neoclassical_growth_kernel_nlls import (
    KernelSettings as GrowthNLLSKernelSettings,
)
from neoclassical_growth_kernel_nlls import (
    NLLSGramSettings as GrowthNLLSSettings,
)
from neoclassical_growth_kernel_nlls import (
    neoclassical_growth_kernel_nlls,
)
from neoclassical_growth_nn import (
    DataSettings as GrowthNNDataSettings,
)
from neoclassical_growth_nn import (
    GrowthPolicySettings as GrowthNNSettings,
)
from neoclassical_growth_nn import (
    StoppingCriteria as GrowthNNStoppingCriteria,
)
from neoclassical_growth_nn import (
    neoclassical_growth_nn,
)
from neoclassical_growth_nn_integrated import (
    DataSettings as IntegratedNNDataSettings,
)
from neoclassical_growth_nn_integrated import (
    GrowthDerivativeSettings as IntegratedNNSettings,
)
from neoclassical_growth_nn_integrated import (
    StoppingCriteria as IntegratedNNStoppingCriteria,
)
from neoclassical_growth_nn_integrated import (
    neoclassical_growth_nn_integrated,
)
from neoclassical_growth_nn_recursive_i import (
    DataSettings as RecursiveIDataSettings,
)
from neoclassical_growth_nn_recursive_i import (
    StoppingCriteria as RecursiveIStoppingCriteria,
)
from neoclassical_growth_nn_recursive_i import (
    neoclassical_growth_nn_recursive_i,
)
from open_economy_growth import (
    ModelParameters as OpenEconomyParameters,
)
from open_economy_growth import (
    stable_tail_terms as open_economy_stable_tail_terms,
)
from open_economy_growth import (
    steady_state as open_economy_steady_state,
)
from open_economy_growth_benchmark import open_economy_growth_benchmark
from open_economy_growth_kernel_nlls import (
    DataSettings as OpenEconomyDataSettings,
)
from open_economy_growth_kernel_nlls import open_economy_growth_kernel_nlls
from open_economy_growth_nn_recursive import (
    DataSettings as OpenEconomyNNDataSettings,
)
from open_economy_growth_nn_recursive import evaluate_path as evaluate_open_economy_nn
from open_economy_growth_nn_recursive import open_economy_growth_nn_recursive
from optimal_advertising_kernel import optimal_advertising_kernel
from optimal_advertising_kernel_nlls import optimal_advertising_kernel_nlls

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

FAST_GROWTH_DATA = GrowthDataSettings(train_T=10.0, train_points=11, test_T=10.0)
FAST_GROWTH_NLLS = GrowthNLLSDataSettings(train_T=10.0, train_points=11, test_T=10.0)


class KernelSmokeTests(unittest.TestCase):
    def test_multicountry_growth_math(self):
        parameters = MulticountryParameters(
            populations=(1.0, 2.0, 1.5),
            productivities=(1.0, 1.1, 0.9),
        )
        marginal_wealth = multicountry_reference_marginal_wealth(parameters)
        steady = multicountry_steady_state(marginal_wealth, parameters)
        allocations = multicountry_allocations(
            steady["k"],
            steady["q"],
            marginal_wealth,
            parameters.rho,
            parameters,
        )
        np.testing.assert_allclose(allocations["capital_drift"], 0.0, atol=1e-14)
        np.testing.assert_allclose(allocations["tobin_q_drift"], 0.0, atol=1e-14)
        np.testing.assert_allclose(
            allocations["marginal_wealth_drift"], 0.0, atol=1e-14
        )
        self.assertAlmostEqual(float(steady["resource_gap"]), 0.0, places=13)
        self.assertAlmostEqual(float(steady["asset_clearing_gap"]), 0.0, places=13)
        np.testing.assert_allclose(steady["relative_welfare_weights"], 1.0, atol=1e-14)

        log_state = jnp.log(
            jnp.concatenate(
                [
                    steady["k"] * jnp.asarray((0.9, 1.1, 1.05)),
                    steady["q"] * jnp.asarray((1.1, 0.95, 1.02)),
                    marginal_wealth * jnp.asarray((0.95, 1.08, 1.03)),
                ]
            )
        )
        log_direction = jnp.asarray(
            (0.02, -0.01, 0.03, -0.04, 0.01, 0.02, 0.03, -0.02, 0.01)
        )
        k, q, marginal_wealth_test = jnp.split(jnp.exp(log_state), 3)
        dlog_k_dt, dlog_q_dt, dlog_m_dt = jnp.split(log_direction, 3)
        values = multicountry_path_values(
            k=k[None, :],
            a=jnp.zeros((1, 3)),
            tobin_q=q[None, :],
            marginal_wealth=marginal_wealth_test[None, :],
            interest_rate=jnp.asarray((parameters.rho,)),
            dlog_k_dt=dlog_k_dt[None, :],
            da_dt=jnp.zeros((1, 3)),
            dlog_q_dt=dlog_q_dt[None, :],
            dlog_marginal_wealth_dt=dlog_m_dt[None, :],
            parameters=parameters,
        )
        populations = jnp.asarray(parameters.populations)
        flow_scale = multicountry_output_scale(parameters)

        def normalized_resource(logs):
            k_jvp, q_jvp, m_jvp = jnp.split(jnp.exp(logs), 3)
            allocation_jvp = multicountry_allocations(
                k_jvp,
                q_jvp,
                m_jvp,
                parameters.rho,
                parameters,
            )
            return (
                jnp.sum(
                    populations
                    * (allocation_jvp["resources"] - allocation_jvp["consumption"])
                )
                / flow_scale
            )

        _, resource_jvp = jax.jvp(
            normalized_resource,
            (log_state,),
            (log_direction,),
        )
        self.assertAlmostEqual(
            float(values["resource_drift"][0]),
            float(resource_jvp),
            places=13,
        )

    def test_multicountry_growth_kernel_smoke(self):
        results, extras = multicountry_growth_kernel_nlls(a_0=(-0.5, 0.5))
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(
            results["dae_residual_count"],
            9 * results["num_train_points"] + 1,
        )
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 3)
        self.assertEqual(results["dynamic_rkhs_function_count"], 8)
        self.assertEqual(
            results["algebraic_rate_value_count"], results["num_train_points"]
        )
        self.assertEqual(results["free_initial_value_count"], 4)
        self.assertEqual(results["metric_selection_dimension"], 3)
        self.assertEqual(
            results["global_closure"],
            "explicit_assets_and_algebraic_clearing",
        )
        self.assertEqual(results["selection"], "minimum_shifted_rkhs_metric")
        self.assertEqual(results["stable_roots"], 2)
        self.assertEqual(results["unstable_roots"], 3)
        self.assertEqual(results["center_roots"], 1)
        self.assertGreater(results["relative_weight"], 1.0)
        self.assertLess(results["long_run_a_1"], results["a_0_1"])
        self.assertGreater(results["minimum_rate_denominator"], 0.0)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(abs(results["initial_resource_drift_residual"]), 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        self.assertLess(results["max_rate_consistency_residual"], 5e-5)
        self.assertLess(results["max_benchmark_error"], 2e-3)
        self.assertLess(
            abs(
                results["log_relative_weight"]
                - results["benchmark_log_relative_weight"]
            ),
            1e-5,
        )
        self.assertLess(results["benchmark_boundary_residual"], 1e-7)
        self.assertAlmostEqual(
            results["rkhs_norm_total"],
            results["rkhs_norm_k_1"]
            + results["rkhs_norm_k_2"]
            + results["rkhs_norm_a_1"]
            + results["rkhs_norm_a_2"]
            + results["rkhs_norm_q_1"]
            + results["rkhs_norm_q_2"]
            + results["rkhs_norm_m_1"]
            + results["rkhs_norm_m_2"],
            places=10,
        )
        self.assertTrue((extras["test_df"]["k_1"] > 0.0).all())
        self.assertTrue((extras["test_df"]["k_2"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c_1"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c_2"] > 0.0).all())
        self.assertLess(extras["test_df"]["asset_clearing_residual"].abs().max(), 5e-5)

    def test_multicountry_growth_three_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.0, 1.0),
            productivities=(1.0, 1.0, 1.0),
        )
        results, extras = multicountry_growth_kernel_nlls(
            a_0=(-0.5, 0.25, 0.25),
            parameters=parameters,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertFalse(results["benchmark_available"])
        self.assertEqual(results["num_countries"], 3)
        self.assertEqual(results["dynamic_rkhs_function_count"], 12)
        self.assertEqual(results["free_initial_value_count"], 6)
        self.assertEqual(results["metric_selection_dimension"], 5)
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 5)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        self.assertAlmostEqual(
            results["relative_weight_2"],
            results["relative_weight_3"],
            places=6,
        )
        self.assertAlmostEqual(
            results["terminal_a_2"], results["terminal_a_3"], places=6
        )
        self.assertIn("k_3", extras["test_df"])
        self.assertIn("marginal_wealth_3", extras["test_df"])
        self.assertIn("resource_drift_residual", extras["test_df"])

        with self.assertRaisesRegex(ValueError, "must sum to zero"):
            multicountry_growth_kernel_nlls(
                a_0=(-0.5, 0.25, 0.0),
                parameters=parameters,
            )

    def test_multicountry_trade_static_equilibrium(self):
        for num_countries in (2, 3):
            parameters = symmetric_trade_growth_parameters(num_countries)
            k = jnp.ones(num_countries)
            tobin_q = jnp.full(num_countries, 1.0 / parameters.installation_elasticity)
            marginal_wealth = jnp.ones(num_countries)
            values, diagnostics = static_trade_equilibrium(
                k,
                tobin_q,
                marginal_wealth,
                parameters,
            )
            self.assertLess(diagnostics["max_residual"], 1e-10)
            self.assertGreater(diagnostics["minimum_jacobian_singular_value"], 1e-3)
            np.testing.assert_allclose(
                values["trade_shares"].sum(axis=1), 1.0, atol=1e-13
            )
            np.testing.assert_allclose(values["budget_trade_gap"], 0.0, atol=1e-13)
            np.testing.assert_allclose(values["world_trade_balance"], 0.0, atol=1e-13)
            np.testing.assert_allclose(values["wage"], values["wage"][0], atol=1e-13)
            np.testing.assert_allclose(
                values["sector_capital"].sum(axis=0),
                k * jnp.asarray(parameters.populations),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                values["sector_labor"].sum(axis=0),
                parameters.populations,
                atol=1e-12,
            )
            self.assertTrue(
                np.all(np.diag(values["trade_shares"]) > values["trade_shares"][0, -1])
            )

    def test_multicountry_trade_dynamic_shell(self):
        parameters = symmetric_trade_growth_parameters(3)
        k = jnp.ones(3)
        tobin_q = jnp.full(3, 1.0 / parameters.installation_elasticity)
        marginal_wealth = jnp.ones(3)
        static, _ = static_trade_equilibrium(
            k,
            tobin_q,
            marginal_wealth,
            parameters,
        )
        k_path = k[None, :]
        q_path = tobin_q[None, :]
        marginal_wealth_path = marginal_wealth[None, :]
        interest_rate = jnp.asarray((parameters.rho,))
        log_static = static["log_static"][None, :, :]
        allocations = multicountry_trade_allocations(
            k_path,
            q_path,
            marginal_wealth_path,
            interest_rate,
            log_static,
            parameters,
        )
        a = jnp.zeros((1, 3))
        da_dt = (
            interest_rate[:, None] * a
            + allocations["factor_income"]
            - allocations["final_expenditure"]
        )
        values = multicountry_trade_path_values(
            k=k_path,
            a=a,
            tobin_q=q_path,
            marginal_wealth=marginal_wealth_path,
            interest_rate=interest_rate,
            log_static=log_static,
            dlog_k_dt=allocations["capital_drift"] / k_path,
            da_dt=da_dt,
            dlog_q_dt=allocations["tobin_q_drift"] / q_path,
            dlog_marginal_wealth_dt=jnp.zeros((1, 3)),
            parameters=parameters,
        )
        for name in (
            "capital",
            "assets",
            "tobin_q",
            "marginal_wealth_euler",
            "world_flow",
            "asset_clearing",
        ):
            np.testing.assert_allclose(values[name], 0.0, atol=1e-12)
        np.testing.assert_allclose(values["static_residuals"], 0.0, atol=1e-12)
        np.testing.assert_allclose(values["budget_trade_gap"], 0.0, atol=1e-12)

    def test_multicountry_trade_three_country_joint_kernel_solve(self):
        parameters = symmetric_trade_growth_parameters(3)
        results, frames = multicountry_trade_growth_kernel_nlls(
            a_0=(-0.2, 0.1, 0.1),
            parameters=parameters,
            data_set=MulticountryTradeDataSettings(
                train_T=15.0,
                train_points=6,
                sampling="uniform",
                dense_T=8.0,
                tail_points=2,
                test_T=8.0,
                test_points=9,
                validation_points=13,
                rank_audit=True,
            ),
            lm_set=MulticountryTradeNLLSGramSettings(
                max_steps=30,
                residual_atol=1e-7,
            ),
        )

        self.assertTrue(results["solution_accepted"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(results["dynamic_rkhs_function_count"], 12)
        self.assertEqual(results["metric_selection_dimension"], 6)
        self.assertEqual(results["unknown_count"] - results["dae_residual_count"], 6)
        self.assertEqual(results["dae_jacobian_rank"], results["dae_residual_count"])
        self.assertGreater(results["dae_jacobian_min_singular_value"], 1e-8)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_validation_residual"], 1e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-8)
        self.assertLess(results["max_rate_consistency_residual"], 1e-8)
        self.assertEqual(results["backend"], jax.default_backend())
        self.assertTrue(np.isfinite(results["ex_post_discounted_assets_decay_ratio"]))
        self.assertTrue(np.isfinite(results["ex_post_asset_tvc_decay_ratio"]))
        self.assertTrue(np.isfinite(results["ex_post_capital_tvc_decay_ratio"]))
        self.assertTrue(results["ex_post_discounted_assets_decay_ratio_applicable"])
        self.assertTrue(results["ex_post_asset_tvc_decay_ratio_applicable"])
        self.assertTrue(results["ex_post_capital_tvc_decay_ratio_applicable"])
        self.assertAlmostEqual(results["terminal_a_2"], results["terminal_a_3"], 7)
        for name in (
            "capital",
            "assets",
            "tobin_q",
            "marginal_wealth",
            "consumption",
            "investment_rate",
            "trade_balance",
            "current_account",
            *STATIC_BLOCK_NAMES,
        ):
            self.assertLess(results[f"max_country_2_3_{name}_gap"], 2e-7)
        self.assertLess(results["max_country_2_3_trade_share_permutation_gap"], 1e-7)
        self.assertAlmostEqual(
            results["terminal_steady_trade_gap_1"],
            results["terminal_current_account_1"],
            5,
        )
        self.assertTrue(np.isfinite(results["plateau_interest_rate_gap"]))
        self.assertLess(
            np.max(np.abs(frames["validation_df"]["consumption_basket_log_price"])),
            1e-4,
        )

    def test_multicountry_trade_three_country_sequential_nn_solve(self):
        results, frames = multicountry_trade_growth_nn(
            a_0=(-0.2, 0.1, 0.1),
            parameters=symmetric_trade_growth_parameters(3),
            seed=0,
            data_set=MulticountryTradeNNDataSettings(
                train_T=15.0,
                train_points=6,
                sampling="uniform",
                dense_T=8.0,
                tail_points=2,
                test_T=8.0,
                test_points=9,
                validation_points=13,
            ),
            nn_set=SequentialNNSettings(
                width=16,
                depth=2,
                input_scale=10.0,
            ),
            lm_set=MulticountryTradeNNNLLSGramSettings(
                max_steps=60,
                residual_atol=1e-10,
            ),
            verbose=False,
        )

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(results["initial_residual_count"], 0)
        self.assertEqual(results["network_output_count"], 37)
        self.assertEqual(results["algebraic_network_output_count"], 25)
        self.assertEqual(results["dae_residual_count"], 37 * 6)
        self.assertGreater(results["unknown_count"], results["dae_residual_count"])
        self.assertEqual(
            results["selection"],
            "minimum_identity_metric_parameter_step",
        )
        self.assertLess(results["max_initial_residual"], 1e-12)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_validation_residual"], 1e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        self.assertLess(results["max_rate_consistency_residual"], 1e-5)
        self.assertTrue(np.isfinite(results["ex_post_discounted_assets"]))
        self.assertTrue(np.isfinite(results["ex_post_asset_tvc"]))
        self.assertTrue(np.isfinite(results["ex_post_capital_tvc"]))
        for name in (
            "capital",
            "assets",
            "tobin_q",
            "marginal_wealth",
            "consumption",
            "investment_rate",
            "trade_balance",
            "current_account",
            *STATIC_BLOCK_NAMES,
        ):
            self.assertLess(results[f"max_country_2_3_{name}_gap"], 2e-4)
        self.assertLess(
            results["max_country_2_3_trade_share_permutation_gap"],
            2e-4,
        )
        self.assertLess(
            np.max(np.abs(frames["validation_df"]["consumption_basket_log_price"])),
            1e-4,
        )

    def test_multicountry_trade_heterogeneous_static_equilibrium(self):
        parameters = TradeGrowthParameters(
            populations=(1.0, 2.0, 1.5),
            intermediate_productivities=(1.0, 1.1, 0.9),
            consumption_productivities=(1.0, 1.05, 0.95),
            investment_productivities=(1.0, 0.9, 1.1),
            consumption_value_added_shares=(0.75, 0.7, 0.8),
            intermediate_value_added_shares=(0.5, 0.55, 0.45),
            investment_value_added_shares=(0.5, 0.45, 0.55),
            trade_costs=(
                (1.0, 1.3, 1.8),
                (1.5, 1.0, 1.4),
                (1.2, 1.6, 1.0),
            ),
        )
        values, diagnostics = static_trade_equilibrium(
            k=jnp.asarray((1.0, 0.8, 1.3)),
            tobin_q=jnp.full(3, 1.0 / parameters.installation_elasticity),
            marginal_wealth=jnp.asarray((1.0, 1.1, 0.9)),
            parameters=parameters,
        )
        self.assertLess(diagnostics["max_residual"], 1e-10)
        self.assertGreater(diagnostics["minimum_jacobian_singular_value"], 1e-3)
        np.testing.assert_allclose(values["trade_shares"].sum(axis=1), 1.0, atol=1e-13)
        np.testing.assert_allclose(values["budget_trade_gap"], 0.0, atol=1e-13)
        np.testing.assert_allclose(values["world_trade_balance"], 0.0, atol=1e-13)
        self.assertFalse(np.allclose(values["trade_shares"], values["trade_shares"].T))

    def test_open_economy_growth_math_and_benchmark(self):
        parameters = OpenEconomyParameters()
        steady = open_economy_steady_state(parameters)
        tail = open_economy_stable_tail_terms(parameters)
        self.assertAlmostEqual(float(steady["s"]), parameters.delta, places=14)
        self.assertAlmostEqual(
            float(steady["q"]),
            1.0 / parameters.installation_elasticity,
            places=14,
        )
        self.assertLess(float(tail["stable_root"]), 0.0)
        self.assertGreater(float(tail["unstable_root"]), 0.0)

        evaluate, diagnostics = open_economy_growth_benchmark(
            1.0,
            0.0,
            np.linspace(0.0, 80.0, 81),
            parameters=parameters,
        )
        values = evaluate(np.linspace(0.0, 50.0, 101))
        self.assertLess(abs(diagnostics["a_0_gap"]), 1e-8)
        self.assertLess(abs(diagnostics["stable_residual"]), 1e-10)
        self.assertLess(abs(diagnostics["present_value_residual"]), 1e-10)
        self.assertGreater(values["x"].min(), 0.0)

    def test_open_economy_growth_nlls_smoke(self):
        results, extras = open_economy_growth_kernel_nlls()
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(results["dae_residual_count"], 3 * results["num_train_points"])
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 2)
        self.assertEqual(results["selection"], "minimum_shifted_rkhs_metric")
        self.assertEqual(results["initialization"], "zero_coordinate")
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_benchmark_error"], 1e-3)
        self.assertLess(results["ex_post_present_value_gap"], 1e-3)
        self.assertLess(results["ex_post_discounted_assets"], 1e-2)
        self.assertLess(results["ex_post_asset_tvc"], 1e-2)
        self.assertLess(results["ex_post_capital_tvc"], 1e-2)
        self.assertAlmostEqual(
            results["rkhs_norm_total"],
            results["rkhs_norm_k"] + results["rkhs_norm_a"] + results["rkhs_norm_q"],
            places=10,
        )
        self.assertGreaterEqual(
            results["shifted_metric_norm_squared"], results["rkhs_norm_total"]
        )
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["q"] > 0.0).all())
        self.assertTrue((extras["test_df"]["x"] > 0.0).all())

        short_horizon, _ = open_economy_growth_kernel_nlls(
            data_set=OpenEconomyDataSettings(
                train_T=80.0,
                train_points=86,
                tail_points=5,
                benchmark_T=120.0,
            )
        )
        self.assertTrue(short_horizon["solution_accepted"])
        self.assertFalse(short_horizon["benchmark_match"])
        self.assertGreater(short_horizon["ex_post_asset_tvc"], 0.1)

    def test_open_economy_growth_nn_recursive_smoke(self):
        parameters = OpenEconomyParameters()
        data_set = OpenEconomyNNDataSettings(
            diagnostic_T=50.0,
            diagnostic_points=51,
            diagnostic_steps=50,
        )
        results, extras = open_economy_growth_nn_recursive(
            parameters=parameters,
            verbose=False,
            measure_compilation=False,
            data_set=data_set,
        )

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(results["dae_residual_count"], 2 * results["num_train_points"])
        self.assertEqual(results["domain_residual_count"], results["num_train_points"])
        self.assertEqual(results["residual_count"], 3 * results["num_train_points"])
        self.assertGreater(results["unknown_count"], results["residual_count"])
        self.assertEqual(results["selection"], "minimum_identity_metric_parameter_step")
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_benchmark_error"], 1e-3)
        self.assertGreater(results["ex_post_asset_tvc"], 0.1)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["q"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())
        self.assertTrue((extras["test_df"]["x"] > 0.0).all())

        times = jnp.linspace(0.0, data_set.test_T, data_set.test_points)
        output_ss = float(open_economy_steady_state(parameters)["output"])
        asset_shift = 0.5 * output_ss
        shifted = evaluate_open_economy_nn(
            extras["model"],
            times,
            round(data_set.test_T / data_set.dt),
            jnp.asarray(1.0),
            jnp.asarray(asset_shift),
            parameters,
            data_set,
        )
        baseline = extras["test_df"]
        np.testing.assert_allclose(shifted["k"], baseline["k"], atol=1e-12)
        np.testing.assert_allclose(shifted["x"], baseline["x"], atol=1e-12)
        np.testing.assert_allclose(
            shifted["current_account"], baseline["current_account"], atol=1e-12
        )
        np.testing.assert_allclose(
            shifted["a"] - baseline["a"], asset_shift, atol=1e-12
        )
        np.testing.assert_allclose(
            shifted["c"] - baseline["c"],
            parameters.interest_rate * asset_shift,
            atol=1e-12,
        )

    def test_integrated_kernels_match_quadrature(self):
        # (32.161, 5.829) is where the removed adaptive-quadrature versions
        # silently missed their tolerance by ~2.6e-4.
        kernel_fns = {
            "matern12": matern_kernel_0p5,
            "matern32": matern_kernel_1p5,
            "matern52": matern_kernel_2p5,
            "rbf": gaussian_kernel,
        }
        geometries = ((0.0, 3.0), (5.0, 1.0), (10.0, 8.0), (32.161, 5.829))
        for name, kernel_fn in kernel_fns.items():
            for ell in (2.0, 10.0):
                for t_i, t_j in geometries:
                    with self.subTest(kernel=name, ell=ell, t_i=t_i, t_j=t_j):
                        _, K_tilde = integrated_kernel_matrices(
                            jnp.array([t_i]), jnp.array([t_j]), name, 1.3, ell
                        )
                        reference = scipy.integrate.quad(
                            lambda s: float(kernel_fn(s, t_j, 1.3, ell)),
                            0.0,
                            t_i,
                            points=[t_j] if 0.0 < t_j < t_i else None,
                            epsabs=1e-13,
                            epsrel=1e-13,
                            limit=200,
                        )[0]
                        self.assertLess(abs(float(K_tilde[0, 0]) - reference), 1e-11)

    def test_neoclassical_growth_smoke(self):
        results, extras = neoclassical_growth_kernel(data_set=FAST_GROWTH_DATA)

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["k_rel_error_mean"], 2e-2)
        self.assertLess(results["c_rel_error_mean"], 2e-2)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_neoclassical_growth_rbf_smoke(self):
        # Gaussian/RBF limit with a short length scale (well conditioned).
        results, _ = neoclassical_growth_kernel(
            kernel_set=KernelSettings(nu=math.inf, ell=2.0),
            data_set=FAST_GROWTH_DATA,
        )

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["k_rel_error_mean"], 2e-2)

    def test_concave_convex_smoke(self):
        results, extras = neoclassical_growth_concave_convex_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["solve_time"], 0.5)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_concave_convex_nlls_smoke(self):
        results, extras = neoclassical_growth_concave_convex_kernel_nlls()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["p_lower_violation"], 1e-3)
        self.assertLess(results["p_upper_violation"], 1e-3)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_neoclassical_growth_nlls_smoke(self):
        results, extras = neoclassical_growth_kernel_nlls(data_set=FAST_GROWTH_NLLS)

        self.assertEqual(results["linear_solver_resolved"], "cholesky")
        self.assertEqual(results["residuals_method_resolved"], "dense")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["k_rel_error_mean"], 2e-2)
        self.assertLess(results["c_rel_error_mean"], 2e-2)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

        # matrix_free recovers the same selection as dense at scale.
        data_set = GrowthNLLSDataSettings(train_points=501)
        dense, _ = neoclassical_growth_kernel_nlls(
            data_set=data_set, lm_set=GrowthNLLSSettings(linear_solver="cholesky")
        )
        structured, _ = neoclassical_growth_kernel_nlls(
            data_set=data_set, lm_set=GrowthNLLSSettings(linear_solver="cg")
        )
        self.assertEqual(structured["linear_solver_resolved"], "cg")
        self.assertEqual(structured["residuals_method_resolved"], "matrix_free")
        self.assertTrue(structured["solution_accepted"])
        self.assertLess(abs(structured["c_0"] - dense["c_0"]), 1e-4)

    def test_neoclassical_growth_nlls_matern32_auto_stays_dense(self):
        # The matern32 iterative window from asset pricing fails the growth
        # gates at scale (measured at n >= 501): auto must not pick cg. On
        # the dense branch it prefers qr for the smooth Materns at
        # n >= DENSE_QR_MIN_POINTS (the ill-conditioned regime where qr
        # converges in 11 steps at n = 101 while cholesky caps out at 50).
        results, _ = neoclassical_growth_kernel_nlls(
            kernel_set=GrowthNLLSKernelSettings(kernel="matern32"),
            data_set=GrowthNLLSDataSettings(train_points=101),
        )
        self.assertEqual(results["linear_solver_resolved"], "qr")
        self.assertEqual(results["residuals_method_resolved"], "dense")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["steps"], 50)

        results, _ = neoclassical_growth_kernel_nlls(
            kernel_set=GrowthNLLSKernelSettings(kernel="matern32"),
            data_set=GrowthNLLSDataSettings(train_points=501),
        )
        self.assertEqual(results["linear_solver_resolved"], "qr")
        self.assertEqual(results["residuals_method_resolved"], "dense")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])

    def test_neoclassical_growth_nlls_kernels_smoke(self):
        for kernel_set, error_bar in [
            (GrowthNLLSKernelSettings(kernel="matern32"), 2e-2),
            (GrowthNLLSKernelSettings(kernel="matern52"), 2e-2),
            (GrowthNLLSKernelSettings(kernel="rbf", ell=2.0), 2e-2),
        ]:
            with self.subTest(kernel=kernel_set.kernel):
                results, _ = neoclassical_growth_kernel_nlls(
                    kernel_set=kernel_set, data_set=FAST_GROWTH_NLLS
                )
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["k_rel_error_mean"], error_bar)

    def test_asset_pricing_smoke(self):
        results, extras = asset_pricing_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-10)
        self.assertLess(results["p_rel_error_mean"], 5e-3)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

    def test_matern_matvecs_match_dense(self):
        rng = np.random.default_rng(7)
        t = jnp.asarray(np.sort(rng.uniform(0.0, 50.0, 501)))
        v = jnp.asarray(rng.standard_normal(501))
        for name in ("matern12", "matern32", "matern52"):
            K, K_tilde = integrated_kernel_matrices(t, t, name, 1.3, 10.0)
            K = (K + K.T) / 2
            for parallel in (False, True):
                with self.subTest(kernel=name, parallel=parallel):
                    k_v, k_tilde_v = matern_kernel_matvec_pair(
                        name, t, v, 1.3, 10.0, parallel=parallel
                    )
                    self.assertLess(float(jnp.max(jnp.abs(K @ v - k_v))), 1e-10)
                    self.assertLess(
                        float(jnp.max(jnp.abs(K_tilde @ v - k_tilde_v))), 1e-10
                    )

    def test_state_space_metric_matches_dense(self):
        # The nlls-gram state-space metric replaced the tinygp qsm solve;
        # keep the dense cross-check of the exact plumbing this repo uses.
        rng = np.random.default_rng(8)
        t = jnp.linspace(0.0, 50.0, 300)
        v = jnp.asarray(rng.standard_normal(300))
        for name, nu in (("matern12", 0.5), ("matern32", 1.5), ("matern52", 2.5)):
            with self.subTest(kernel=name):
                jitter = 1e-8 * 1.3**2
                K, _ = integrated_kernel_matrices(t, t, name, 1.3, 10.0)
                K_jittered = np.asarray((K + K.T) / 2) + jitter * np.eye(300)
                metric = metric_from_state_space(
                    t, *matern_state_space(1.3, 10.0, nu), nugget=jitter
                )
                solution = metric.solve(v)
                reference = np.linalg.solve(K_jittered, np.asarray(v))
                self.assertLess(
                    float(np.max(np.abs(np.asarray(solution) - reference)))
                    / float(np.max(np.abs(reference))),
                    1e-6,
                )
                norm_reference = float(np.sqrt(np.asarray(v) @ K_jittered @ v))
                self.assertLess(
                    abs(float(metric.norm(v)) - norm_reference) / norm_reference,
                    1e-10,
                )

    def test_asset_pricing_nlls_smoke(self):
        reference, extras = asset_pricing_kernel_nlls()

        self.assertEqual(reference["linear_solver_resolved"], "cholesky")
        self.assertEqual(reference["residuals_method_resolved"], "dense")
        self.assertTrue(reference["solution_accepted"], reference["rejection_reason"])
        self.assertLess(reference["max_train_residual"], 1e-10)
        self.assertLess(reference["p_rel_error_mean"], 5e-3)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

        # matrix_free must recover the same min-norm selection as the dense
        # default, not just interpolate.
        lm_set = NLLSGramSettings(linear_solver="cg", residuals_method="matrix_free")
        results, extras = asset_pricing_kernel_nlls(lm_set=lm_set)

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-10)
        self.assertLess(results["p_rel_error_mean"], 5e-3)
        self.assertLess(abs(results["p_0"] - reference["p_0"]), 1e-9)
        self.assertLess(abs(results["rkhs_norm_p"] - reference["rkhs_norm_p"]), 1e-9)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

    def test_asset_pricing_nlls_kernels_smoke(self):
        # Every named kernel solves on the dense path; the smoother kernels
        # need their documented working configurations (rbf: short length
        # scale -- its Gram at ell = 10 is numerically singular for every
        # solver, IPOPT included).
        cases = {
            "matern32": (APKernelSettings(kernel="matern32"), 2e-2),
            "matern52": (APKernelSettings(kernel="matern52"), 2e-2),
            "rbf": (APKernelSettings(kernel="rbf", ell=2.0), 1e-1),
        }
        for name, (kernel_set, error_bar) in cases.items():
            with self.subTest(kernel=name):
                results, _ = asset_pricing_kernel_nlls(kernel_set=kernel_set)

                self.assertEqual(results["linear_solver_resolved"], "cholesky")
                self.assertEqual(results["residuals_method_resolved"], "dense")
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["max_train_residual"], 1e-8)
                self.assertLess(results["p_rel_error_mean"], error_bar)

        # matern32 matrix_free agrees with its dense selection.
        kernel_set = APKernelSettings(kernel="matern32")
        data_set = APDataSettings(train_T=50.0, train_points=1001)
        dense, _ = asset_pricing_kernel_nlls(
            kernel_set=kernel_set,
            data_set=data_set,
            lm_set=NLLSGramSettings(linear_solver="cholesky"),
        )
        structured, _ = asset_pricing_kernel_nlls(
            kernel_set=kernel_set,
            data_set=data_set,
            lm_set=NLLSGramSettings(linear_solver="cg"),
        )
        self.assertTrue(structured["solution_accepted"])
        self.assertLess(abs(structured["p_0"] - dense["p_0"]), 1e-8)

    def test_normalize_kernel_aliases(self):
        cases = [
            (0.5, "matern12"),
            (1.5, "matern32"),
            (2.5, "matern52"),
            (math.inf, "rbf"),
            ("gaussian", "rbf"),
            ("se", "rbf"),
            ("matern_1_2", "matern12"),
            ("matern2.5", "matern52"),
        ]
        for alias, expected in cases:
            self.assertEqual(normalize_kernel(alias), expected)
        with self.assertRaises(ValueError):
            normalize_kernel("matern72")

    def test_asset_pricing_nlls_eps_contract(self):
        # The unified shift applies identically across configurations --
        # every one factors K + eps I and weights p_0 by eps -- so their
        # min-M-norm selections agree, matern12 included (the newly unified
        # case: its matrix_free metric is now state-space, not the old
        # nugget-free tridiagonal precision).
        data_set = APDataSettings(train_T=50.0, train_points=501)
        eps = 1e-6
        for kernel in ("matern12", "matern32"):
            with self.subTest(kernel=kernel):
                kernel_set = APKernelSettings(kernel=kernel)
                dense, _ = asset_pricing_kernel_nlls(
                    kernel_set=kernel_set,
                    data_set=data_set,
                    lm_set=NLLSGramSettings(linear_solver="cholesky", metric_eps=eps),
                )
                structured, _ = asset_pricing_kernel_nlls(
                    kernel_set=kernel_set,
                    data_set=data_set,
                    lm_set=NLLSGramSettings(linear_solver="cg", metric_eps=eps),
                )
                self.assertTrue(structured["solution_accepted"])
                self.assertLess(abs(structured["p_0"] - dense["p_0"]), 1e-8)

        with self.assertRaises(ValueError):
            asset_pricing_kernel_nlls(lm_set=NLLSGramSettings(metric_eps=0.0))

    def test_asset_pricing_nlls_auto_dispatch(self):
        small, _ = asset_pricing_kernel_nlls()
        self.assertEqual(small["linear_solver_resolved"], "cholesky")
        self.assertEqual(small["residuals_method_resolved"], "dense")

        large, _ = asset_pricing_kernel_nlls(
            data_set=APDataSettings(train_T=50.0, train_points=501)
        )
        self.assertEqual(large["linear_solver_resolved"], "cg")
        self.assertEqual(large["residuals_method_resolved"], "matrix_free")
        self.assertTrue(large["solution_accepted"])

        rbf, _ = asset_pricing_kernel_nlls(
            kernel_set=APKernelSettings(kernel="rbf", ell=2.0),
            data_set=APDataSettings(train_T=50.0, train_points=501),
        )
        self.assertEqual(rbf["linear_solver_resolved"], "cholesky")
        self.assertEqual(rbf["residuals_method_resolved"], "dense")

        with self.assertRaises(ValueError):
            asset_pricing_kernel_nlls(
                kernel_set=APKernelSettings(kernel="matern52"),
                lm_set=NLLSGramSettings(residuals_method="matrix_free"),
            )

        # Point lists are sorted at intake (collocation is permutation-
        # invariant), so an unsorted list resolves and solves exactly like
        # its sorted version -- matrix_free at this scale.
        points = np.linspace(0.0, 50.0, 1600)
        shuffled = points.copy()
        shuffled[10], shuffled[11] = shuffled[11], shuffled[10]
        unsorted, _ = asset_pricing_kernel_nlls(
            data_set=APDataSettings(train_points_list=[float(p) for p in shuffled])
        )
        sorted_run, _ = asset_pricing_kernel_nlls(
            data_set=APDataSettings(train_points_list=[float(p) for p in points])
        )
        self.assertEqual(unsorted["linear_solver_resolved"], "cg")
        self.assertEqual(unsorted["residuals_method_resolved"], "matrix_free")
        self.assertTrue(unsorted["solution_accepted"], unsorted["rejection_reason"])
        self.assertEqual(unsorted["p_0"], sorted_run["p_0"])

        # Duplicate points are the one remaining obstruction: auto falls
        # back to dense, and explicit matrix_free refuses.
        duplicated = [float(p) for p in points[:-1]] + [float(points[-2])]
        dup, _ = asset_pricing_kernel_nlls(
            data_set=APDataSettings(train_points_list=duplicated)
        )
        self.assertEqual(dup["linear_solver_resolved"], "cholesky")
        self.assertEqual(dup["residuals_method_resolved"], "dense")
        with self.assertRaises(ValueError):
            asset_pricing_kernel_nlls(
                data_set=APDataSettings(train_points_list=duplicated),
                lm_set=NLLSGramSettings(residuals_method="matrix_free"),
            )

    def test_nlls_qr_matches_cholesky(self):
        # qr factors the whitened Jacobian instead of the dual Gram; both
        # dense solvers must land on the same min-M-norm selection.
        cholesky_ap, _ = asset_pricing_kernel_nlls(
            lm_set=NLLSGramSettings(linear_solver="cholesky")
        )
        qr_ap, _ = asset_pricing_kernel_nlls(
            lm_set=NLLSGramSettings(linear_solver="qr")
        )
        self.assertEqual(qr_ap["linear_solver_resolved"], "qr")
        self.assertEqual(qr_ap["residuals_method_resolved"], "dense")
        self.assertTrue(qr_ap["solution_accepted"], qr_ap["rejection_reason"])
        self.assertLess(abs(qr_ap["p_0"] - cholesky_ap["p_0"]), 1e-8)

        cholesky_gr, _ = neoclassical_growth_kernel_nlls(
            lm_set=GrowthNLLSSettings(linear_solver="cholesky")
        )
        qr_gr, _ = neoclassical_growth_kernel_nlls(
            lm_set=GrowthNLLSSettings(linear_solver="qr")
        )
        self.assertEqual(qr_gr["linear_solver_resolved"], "qr")
        self.assertEqual(qr_gr["residuals_method_resolved"], "dense")
        self.assertTrue(qr_gr["solution_accepted"], qr_gr["rejection_reason"])
        self.assertLess(abs(qr_gr["c_0"] - cholesky_gr["c_0"]), 1e-4)

    def test_nlls_cg_falls_back_to_dense_residuals(self):
        # Explicit cg on a matrix-free-ineligible kernel: residuals auto
        # falls back to dense (no error), exercising cg with dense residual
        # matrices and the Sherman-Morrison preconditioner.
        results, _ = asset_pricing_kernel_nlls(
            kernel_set=APKernelSettings(kernel="rbf", ell=2.0),
            lm_set=NLLSGramSettings(linear_solver="cg"),
        )
        self.assertEqual(results["linear_solver_resolved"], "cg")
        self.assertEqual(results["residuals_method_resolved"], "dense")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])

    def test_nlls_float32_residuals_use_float64_metric(self):
        results, frames = asset_pricing_kernel_nlls(use_float64=False)

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["residual_dtype"], "float32")
        self.assertEqual(results["metric_dtype"], "float64")
        self.assertEqual(set(frames["train_df"].dtypes.astype(str)), {"float32"})

    def test_optimal_advertising_smoke(self):
        results, extras = optimal_advertising_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertAlmostEqual(results["gamma"], 0.32)
        self.assertTrue((extras["test_df"]["u"] > 0.0).all())

    def test_optimal_advertising_nlls_smoke(self):
        results, extras = optimal_advertising_kernel_nlls()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertAlmostEqual(results["mu_0"], 0.8835154879257, places=10)
        self.assertAlmostEqual(results["u_0"], 0.2650546463776, places=10)
        self.assertAlmostEqual(results["gamma"], 0.32)
        self.assertTrue((extras["test_df"]["u"] > 0.0).all())

    def test_human_capital_smoke(self):
        h_0, c_0, residual, _ = human_capital_initial_conditions()
        self.assertAlmostEqual(float(h_0), 1.3745155888757778, places=10)
        self.assertLess(float(jnp.linalg.norm(residual, ord=jnp.inf)), 1e-12)

        results, extras = human_capital_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["num_train_points"], 81)
        self.assertAlmostEqual(results["h_0"], 1.3745155888757778, places=10)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_test_feasibility"], 1e-3)
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_human_capital_nlls_smoke(self):
        results, extras = human_capital_kernel_nlls()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["num_train_points"], 81)
        self.assertAlmostEqual(results["h_0"], 1.3745155888757778, places=10)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_validation_residual"], 4e-4)
        self.assertLess(results["max_test_feasibility"], 4e-4)
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())


class NNSmokeTests(unittest.TestCase):
    def tearDown(self):
        # These solves retain large reverse-over-forward executables on GPU;
        # release them before compiling the next model family.
        jax.clear_caches()

    def test_neoclassical_growth_nn_smoke(self):
        results, frames = neoclassical_growth_nn(
            verbose=False,
            measure_compilation=False,
            data_set=GrowthNNDataSettings(
                train_T=10.0,
                train_points=11,
                test_T=10.0,
                test_points=21,
                test_tvc_T=50.0,
            ),
            nn_set=GrowthNNSettings(width=16, depth=2),
            stop_set=GrowthNNStoppingCriteria(
                max_steps=40,
                train_loss_threshold=1e-10,
                num_attempts=1,
            ),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)
        self.assertTrue((frames["test_df"]["k"] > 0.0).all())
        self.assertIn("nn_set", frames["hyperparameters"])

    def test_neoclassical_growth_nn_integrated_rff_smoke(self):
        results, frames = neoclassical_growth_nn_integrated(
            verbose=False,
            measure_compilation=False,
            data_set=IntegratedNNDataSettings(
                train_T=10.0,
                train_points=11,
                test_T=10.0,
                test_points=21,
                test_tvc_T=50.0,
                integration_substeps=4,
            ),
            nn_set=IntegratedNNSettings(
                approximation="rff",
                width=1024,
                rff_kernel="matern12",
                rff_radius=10.0,
                rff_sigma=1.0,
            ),
            stop_set=IntegratedNNStoppingCriteria(
                max_steps=120,
                train_loss_threshold=1e-8,
                num_attempts=1,
            ),
        )

        self.assertEqual(results["approximation"], "rff")
        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)

    def test_recursive_i_smoke(self):
        results, extras = neoclassical_growth_nn_recursive_i(
            verbose=False,
            measure_compilation=False,
            data_set=RecursiveIDataSettings(
                dt=2.0, train_T=20.0, train_tvc_T=100.0, test_T=20.0
            ),
            stop_set=RecursiveIStoppingCriteria(max_steps=200, num_attempts=1),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertTrue(results["transversality_ok"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)
        self.assertGreater(results["min_test_c"], 0.0)
        self.assertIn("i_nn", extras["policy_df"].columns)


if __name__ == "__main__":
    unittest.main()
