import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate

from asset_pricing_kernel import KernelSettings as APKernelSettings
from asset_pricing_kernel import SolverSettings, asset_pricing_kernel
from asset_pricing_nn import asset_pricing_nn
from human_capital_kernel import human_capital_kernel
from kernels import (
    gaussian_kernel,
    integrated_kernel_matrices,
    matern_kernel_0p5,
    matern_kernel_1p5,
    matern_kernel_2p5,
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
from multicountry_growth_kernel import multicountry_growth_kernel
from multicountry_growth_nn import multicountry_growth_nn
from neoclassical_growth_concave_convex_kernel import (
    neoclassical_growth_concave_convex_kernel,
)
from neoclassical_growth_kernel import DataSettings as GrowthDataSettings
from neoclassical_growth_kernel import KernelSettings, neoclassical_growth_kernel
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
from neoclassical_growth_nn_recursive import DataSettings as RecursiveNNDataSettings
from neoclassical_growth_nn_recursive import (
    StoppingCriteria as RecursiveNNStoppingCriteria,
)
from neoclassical_growth_nn_recursive import neoclassical_growth_nn_recursive
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
from open_economy_growth_kernel import (
    DataSettings as OpenEconomyDataSettings,
)
from open_economy_growth_kernel import open_economy_growth_kernel
from open_economy_growth_nn_recursive import DataSettings as OpenEconomyNNDataSettings
from open_economy_growth_nn_recursive import evaluate_path as evaluate_open_economy_nn
from open_economy_growth_nn_recursive import open_economy_growth_nn_recursive
from optimal_advertising_kernel import optimal_advertising_kernel

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

FAST_GROWTH_DATA = GrowthDataSettings(train_T=10.0, train_points=11, test_T=10.0)


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
        results, extras = multicountry_growth_kernel(a_0=(-0.5, 0.5))
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["linear_solver"], "auto")
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

    def test_multicountry_growth_nn_smoke(self):
        results, extras = multicountry_growth_nn(
            a_0=(-0.5, 0.5),
            seed=0,
            verbose=False,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(
            results["equation_diagnostics_pass"],
            results["equation_diagnostic_reason"],
        )
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["stopping_reason"], "early_stopping_met")
        self.assertEqual(results["attempt"], 1)
        self.assertEqual(results["linear_solver"], "auto")
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(results["benchmark_training_residual_count"], 0)
        self.assertEqual(results["tvc_training_residual_count"], 0)
        self.assertEqual(results["dynamic_network_output_count"], 8)
        self.assertEqual(results["calculated_algebraic_value_count"], 5)
        self.assertEqual(
            results["dae_residual_count"],
            9 * results["num_train_points"] + 1,
        )
        self.assertGreater(results["total_params"], 500)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        self.assertLess(results["max_rate_consistency_residual"], 5e-5)
        self.assertLess(results["max_benchmark_error"], 5e-4)
        self.assertLess(
            abs(
                results["log_relative_weight"]
                - results["benchmark_log_relative_weight"]
            ),
            1e-5,
        )
        initial = extras["test_df"].iloc[0]
        self.assertAlmostEqual(initial["k_1"], 1.0, places=14)
        self.assertAlmostEqual(initial["k_2"], 1.0, places=14)
        self.assertAlmostEqual(initial["a_1"], -0.5, places=14)
        self.assertAlmostEqual(initial["a_2"], 0.5, places=14)

    def test_multicountry_growth_nn_three_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.0, 1.0),
            productivities=(1.0, 1.0, 1.0),
        )
        results, extras = multicountry_growth_nn(
            a_0=(-0.5, 0.25, 0.25),
            parameters=parameters,
            seed=0,
            verbose=False,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(
            results["equation_diagnostics_pass"],
            results["equation_diagnostic_reason"],
        )
        self.assertFalse(results["benchmark_available"])
        self.assertEqual(results["num_countries"], 3)
        self.assertEqual(results["dynamic_network_output_count"], 12)
        self.assertEqual(results["calculated_algebraic_value_count"], 7)
        self.assertAlmostEqual(
            results["relative_weight_2"],
            results["relative_weight_3"],
            delta=1e-6,
        )
        self.assertIn("k_3", extras["test_df"])
        self.assertIn("marginal_wealth_3", extras["test_df"])

    def test_multicountry_growth_three_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.0, 1.0),
            productivities=(1.0, 1.0, 1.0),
        )
        results, extras = multicountry_growth_kernel(
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
            results["terminal_a_2"], results["terminal_a_3"], delta=1e-5
        )
        self.assertIn("k_3", extras["test_df"])
        self.assertIn("marginal_wealth_3", extras["test_df"])
        self.assertIn("resource_drift_residual", extras["test_df"])

        with self.assertRaisesRegex(ValueError, "must sum to zero"):
            multicountry_growth_kernel(
                a_0=(-0.5, 0.25, 0.0),
                parameters=parameters,
            )

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

    def test_open_economy_growth_kernel_smoke(self):
        baseline_parameters = OpenEconomyParameters()
        parameters = OpenEconomyParameters(r_star=baseline_parameters.rho)
        results, extras = open_economy_growth_kernel(parameters=parameters)
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["stopping_reason"], "converged")
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(
            results["differential_residual_count"],
            4 * results["num_train_points"],
        )
        self.assertEqual(
            results["algebraic_residual_count"], results["num_train_points"]
        )
        self.assertEqual(results["dae_residual_count"], 5 * results["num_train_points"])
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 3)
        self.assertEqual(results["dynamic_rkhs_function_count"], 4)
        self.assertEqual(results["algebraic_rkhs_function_count"], 1)
        self.assertEqual(results["free_initial_value_count"], 3)
        self.assertEqual(results["metric_selection_dimension"], 3)
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
            results["rkhs_norm_k"]
            + results["rkhs_norm_a"]
            + results["rkhs_norm_q"]
            + results["rkhs_norm_c"]
            + results["rkhs_norm_mu"],
            places=10,
        )
        self.assertGreaterEqual(
            results["shifted_metric_norm_squared"], results["rkhs_norm_total"]
        )
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["q"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())
        self.assertTrue((extras["test_df"]["x"] > 0.0).all())
        self.assertLess(extras["train_df"]["marginal_wealth"].abs().max(), 1e-6)
        self.assertLess(extras["train_df"]["consumption_foc"].abs().max(), 1e-6)

        short_horizon, _ = open_economy_growth_kernel(
            data_set=OpenEconomyDataSettings(
                train_T=80.0,
                train_points=86,
                tail_points=5,
                benchmark_T=120.0,
            )
        )
        self.assertFalse(short_horizon["solution_accepted"])
        self.assertIn("solver_status", short_horizon["rejection_reason"])
        self.assertFalse(short_horizon["benchmark_match"])
        self.assertGreater(short_horizon["ex_post_asset_tvc"], 0.1)

    def test_open_economy_growth_kernel_initial_assets(self):
        parameters = OpenEconomyParameters()
        output_ss = float(open_economy_steady_state(parameters)["output"])
        results, _ = open_economy_growth_kernel(
            a_0=0.5 * output_ss,
            parameters=parameters,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["stopping_reason"], "converged")
        self.assertLess(results["steps"], 100)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_benchmark_error"], 1e-3)

    def test_open_economy_growth_kernel_nonstationary_consumption(self):
        parameters = OpenEconomyParameters(r_star=0.05)
        results, extras = open_economy_growth_kernel(parameters=parameters)
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["stopping_reason"], "converged")
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(
            results["differential_residual_count"],
            4 * results["num_train_points"],
        )
        self.assertEqual(
            results["algebraic_residual_count"], results["num_train_points"]
        )
        self.assertEqual(results["dae_residual_count"], 5 * results["num_train_points"])
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 3)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["c_relative_error_max"], 2e-4)
        self.assertLess(results["max_benchmark_error"], 2e-2)

        path = extras["train_df"]
        realized_growth = math.log(path["c"].iloc[-1] / path["c"].iloc[0]) / (
            path["t"].iloc[-1] - path["t"].iloc[0]
        )
        expected_growth = (parameters.interest_rate - parameters.rho) / parameters.gamma
        self.assertAlmostEqual(realized_growth, expected_growth, places=5)
        self.assertLess(path["marginal_wealth"].abs().max(), 1e-6)
        self.assertLess(path["consumption_foc"].abs().max(), 1e-6)

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

    def test_concave_convex_kernel_smoke(self):
        results, extras = neoclassical_growth_concave_convex_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["p_lower_violation"], 1e-3)
        self.assertLess(results["p_upper_violation"], 1e-3)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_neoclassical_growth_kernel_smoke(self):
        results, extras = neoclassical_growth_kernel(data_set=FAST_GROWTH_DATA)

        self.assertEqual(results["linear_solver"], "auto")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["k_rel_error_mean"], 2e-2)
        self.assertLess(results["c_rel_error_mean"], 2e-2)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_neoclassical_growth_kernel_families(self):
        for kernel_set, error_bar in [
            (KernelSettings(kernel="matern32"), 2e-2),
            (KernelSettings(kernel="matern52"), 2e-2),
            (KernelSettings(kernel="rbf", ell=2.0), 2e-2),
        ]:
            with self.subTest(kernel=kernel_set.kernel):
                results, _ = neoclassical_growth_kernel(
                    kernel_set=kernel_set, data_set=FAST_GROWTH_DATA
                )
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["k_rel_error_mean"], error_bar)

    def test_asset_pricing_kernel_smoke(self):
        reference, extras = asset_pricing_kernel()

        self.assertEqual(reference["linear_solver"], "auto")
        self.assertTrue(reference["solution_accepted"], reference["rejection_reason"])
        self.assertLess(reference["max_train_residual"], 1e-10)
        self.assertLess(reference["p_rel_error_mean"], 5e-3)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

    def test_asset_pricing_kernel_families(self):
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
                results, _ = asset_pricing_kernel(kernel_set=kernel_set)

                self.assertEqual(results["linear_solver"], "auto")
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["max_train_residual"], 1e-8)
                self.assertLess(results["p_rel_error_mean"], error_bar)

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

    def test_asset_pricing_metric_shift_must_be_positive(self):
        with self.assertRaises(ValueError):
            asset_pricing_kernel(lm_set=SolverSettings(metric_eps=0.0))

    def test_float32_residuals_use_float64_metric(self):
        results, frames = asset_pricing_kernel(use_float64=False)

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["residual_dtype"], "float32")
        self.assertEqual(results["metric_dtype"], "float64")
        self.assertEqual(set(frames["train_df"].dtypes.astype(str)), {"float32"})

    def test_optimal_advertising_kernel_smoke(self):
        results, extras = optimal_advertising_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertAlmostEqual(results["mu_0"], 0.8835154879257, places=10)
        self.assertAlmostEqual(results["u_0"], 0.2650546463776, places=10)
        self.assertAlmostEqual(results["gamma"], 0.32)
        self.assertTrue((extras["test_df"]["u"] > 0.0).all())

    def test_human_capital_kernel_smoke(self):
        results, extras = human_capital_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["num_train_points"], 81)
        self.assertAlmostEqual(results["h_0"], 1.3745155888757778, places=10)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_validation_residual"], 4e-4)
        self.assertLess(results["max_test_feasibility"], 4e-4)
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())


class NNSmokeTests(unittest.TestCase):
    def test_neoclassical_growth_nn_smoke(self):
        results, frames = neoclassical_growth_nn(
            verbose=False,
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

    def test_asset_pricing_nn_smoke(self):
        results, frames = asset_pricing_nn(
            verbose=False,
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertEqual(results["selection"], "minimum_identity_metric_parameter_step")
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(results["tvc_training_residual_count"], 0)
        self.assertEqual(results["benchmark_training_residual_count"], 0)
        self.assertLess(results["p_rel_error_mean"], 5e-3)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertIn("bubble_coefficient", frames["test_df"])

    def test_recursive_nn_smoke(self):
        results, extras = neoclassical_growth_nn_recursive(
            verbose=False,
            data_set=RecursiveNNDataSettings(
                dt=2.0, train_T=20.0, train_tvc_T=100.0, test_T=20.0
            ),
            stop_set=RecursiveNNStoppingCriteria(max_steps=200, num_attempts=1),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertTrue(results["transversality_ok"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)
        self.assertGreater(results["min_test_c"], 0.0)
        self.assertIn("i_nn", extras["policy_df"].columns)

    def test_recursive_nn_butterfly_smoke(self):
        results, _ = neoclassical_growth_nn_recursive(
            k_0=1.5,
            A=0.5,
            b_1=3.0,
            b_2=2.5,
            verbose=False,
            data_set=RecursiveNNDataSettings(
                dt=2.0,
                train_T=20.0,
                train_tvc_T=100.0,
                test_T=20.0,
                settle_T=200.0,
            ),
            stop_set=RecursiveNNStoppingCriteria(max_steps=200, num_attempts=1),
        )

        self.assertFalse(results["solution_accepted"])
        self.assertEqual(results["stopping_reason"], "max_steps")
        self.assertTrue(results["basin_correct"])
        self.assertFalse(results["false_positive"])
        self.assertLess(
            abs(results["k_settled"] - results["k_ss_low"]),
            0.05 * results["k_thr"],
        )


if __name__ == "__main__":
    unittest.main()
