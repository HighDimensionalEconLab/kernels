import dataclasses
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
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    multicountry_allocations,
    multicountry_path_values,
    multicountry_reference_marginal_wealth,
    multicountry_steady_state,
    weighted_gini,
)
from multicountry_growth_kernel import DataSettings as MulticountryDataSettings
from multicountry_growth_kernel import SolverSettings as MulticountrySolverSettings
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
from open_economy_growth_kernel_uno import open_economy_growth_kernel_uno
from open_economy_growth_nn_recursive import DataSettings as OpenEconomyNNDataSettings
from open_economy_growth_nn_recursive import evaluate_path as evaluate_open_economy_nn
from open_economy_growth_nn_recursive import open_economy_growth_nn_recursive
from optimal_advertising_kernel import optimal_advertising_kernel

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

FAST_GROWTH_DATA = GrowthDataSettings(train_T=10.0, train_points=11, test_T=10.0)


class KernelSmokeTests(unittest.TestCase):
    def test_weighted_gini(self):
        # Hand values: equal weights on v=(1,3) give dispersion 1 and mean
        # 2, so Gini 1/4; weights (2,1) on v=(1,2) give 4/9 over 8/3 = 1/6.
        self.assertAlmostEqual(
            float(weighted_gini(jnp.array([1.0, 3.0]), (1.0, 1.0))), 0.25, places=14
        )
        self.assertAlmostEqual(
            float(weighted_gini(jnp.array([1.0, 2.0]), (2.0, 1.0))),
            1.0 / 6.0,
            places=14,
        )
        # Scale invariance and time-axis vectorization: paths with constant
        # cross-country ratios have a constant Gini.
        values = jnp.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(
            float(weighted_gini(7.0 * values, (1.0, 2.0, 0.5))),
            float(weighted_gini(values, (1.0, 2.0, 0.5))),
            places=14,
        )
        growth = jnp.exp(0.03 * jnp.arange(5.0))[:, None]
        path_gini = weighted_gini(growth * values, (1.0, 2.0, 0.5))
        self.assertEqual(path_gini.shape, (5,))
        np.testing.assert_allclose(path_gini, path_gini[0], rtol=1e-13)

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

        def resource(logs):
            k_jvp, q_jvp, m_jvp = jnp.split(jnp.exp(logs), 3)
            allocation_jvp = multicountry_allocations(
                k_jvp,
                q_jvp,
                m_jvp,
                parameters.rho,
                parameters,
            )
            return jnp.sum(
                populations
                * (allocation_jvp["resources"] - allocation_jvp["consumption"])
            )

        _, resource_jvp = jax.jvp(
            resource,
            (log_state,),
            (log_direction,),
        )
        self.assertAlmostEqual(
            float(values["resource_drift"][0]),
            float(resource_jvp),
            places=13,
        )

    def test_multicountry_growth_kernel_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.4),
            productivities=(1.05, 0.95),
        )
        results, extras = multicountry_growth_kernel(parameters=parameters)
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["linear_solver"], "auto")
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["terminal_residual_count"], 0)
        self.assertEqual(
            results["dae_residual_count"],
            9 * results["num_train_points"] + 1,
        )
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 4)
        self.assertEqual(results["dynamic_rkhs_function_count"], 8)
        self.assertEqual(results["algebraic_rkhs_function_count"], 1)
        self.assertEqual(results["free_initial_value_count"], 5)
        self.assertEqual(results["metric_selection_dimension"], 4)
        self.assertEqual(
            results["global_closure"],
            "explicit_assets_and_algebraic_clearing",
        )
        self.assertEqual(results["selection"], "minimum_rkhs_norm_ridge")
        self.assertEqual(results["stable_roots"], 2)
        self.assertEqual(results["unstable_roots"], 3)
        self.assertEqual(results["center_roots"], 1)
        # Drawn heterogeneous k_0 makes the welfare-weight ordering
        # draw-dependent; correctness is pinned by the benchmark
        # log-relative-weight agreement below.
        self.assertGreater(results["relative_weight"], 0.0)
        self.assertNotAlmostEqual(results["relative_weight"], 1.0, places=3)
        # The long-run NFA position moves genuinely away from the initial
        # one (the sign of the drift depends on the drawn capital
        # heterogeneity, so no ordering is asserted).
        self.assertNotAlmostEqual(results["long_run_a_1"], results["a_0_1"], places=2)
        # By t=100 the paths are flat and at the theory steady state
        # (bounds are 3-5x the measured values at this draw).
        self.assertLess(results["long_run_c_flatness"], 0.01)
        self.assertLess(results["long_run_k_flatness"], 0.02)
        self.assertLess(results["long_run_a_flatness"], 0.01)
        self.assertLess(results["long_run_k_theory_gap"], 0.03)
        self.assertLess(results["long_run_mpk_equalization_gap"], 0.02)
        self.assertLess(results["long_run_interest_rate_gap"], 0.03)
        # Exact one-good invariants at solver accuracy: constant consumption
        # ratios (common Euler drift) and consumption proportional to
        # initial total wealth a_0 + PV(resources).
        self.assertLess(results["long_run_consumption_ratio_drift"], 1e-5)
        self.assertLess(results["long_run_wealth_proportionality_gap"], 1e-3)
        self.assertTrue(results["long_run_c_rank_matches_wealth"])
        self.assertEqual(results["long_run_wealth_rank_violation"], 0.0)
        # At this draw the k_0 heterogeneity dominates the debt draw:
        # country 1 holds MORE initial debt yet higher total wealth, so its
        # long-run consumption is higher -- the a_0-only consumption
        # ordering fails while the wealth ordering is exact. The long-run
        # NFA ranking still follows the initial debt ranking here.
        self.assertLess(results["a_0_1"], results["a_0_2"])
        self.assertGreater(results["initial_wealth_1"], results["initial_wealth_2"])
        self.assertGreater(results["long_run_c_1"], results["long_run_c_2"])
        self.assertFalse(results["long_run_c_rank_matches_a_0"])
        self.assertTrue(results["long_run_terminal_a_rank_matches_a_0"])
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
            + results["rkhs_norm_m_2"]
            + results["rkhs_norm_interest_rate"],
            places=10,
        )
        self.assertTrue((extras["test_df"]["k_1"] > 0.0).all())
        self.assertTrue((extras["test_df"]["k_2"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c_1"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c_2"] > 0.0).all())
        self.assertLess(extras["test_df"]["asset_clearing_residual"].abs().max(), 5e-5)

    def test_multicountry_growth_qr_configuration(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.3, 0.8),
            productivities=(1.1, 0.95, 1.0),
        )
        data_set = MulticountryDataSettings(
            train_T=40.0,
            train_points=24,
            dense_T=20.0,
            tail_points=5,
            test_T=20.0,
            test_points=21,
            validation_points=31,
            benchmark_T=50.0,
            benchmark_points=51,
        )
        lm_set = MulticountrySolverSettings(
            max_steps=2,
            linear_solver="qr",
            geodesic_acceleration=False,
        )
        results, _ = multicountry_growth_kernel(
            parameters=parameters,
            data_set=data_set,
            lm_set=lm_set,
        )
        self.assertEqual(results["linear_solver"], "qr")
        self.assertEqual(results["residual_atol"], lm_set.residual_atol)
        self.assertFalse(results["solution_accepted"])
        self.assertIn("solver_status", results["rejection_reason"])

        with self.assertRaisesRegex(ValueError, "linear_solver must be one of"):
            multicountry_growth_kernel(
                parameters=parameters,
                lm_set=MulticountrySolverSettings(linear_solver="normal_cg"),
            )

    def test_multicountry_growth_cg_configuration(self):
        # The matrix-free path with the block-eigen preconditioner solves the
        # small-grid N=3 problem to the same acceptance gates as the dense
        # default, and reports its configuration. 36 collocation points is
        # the coarsest grid whose rate-consistency gate holds under the
        # drawn heterogeneous initial conditions.
        parameters = MulticountryParameters(
            populations=(1.0, 1.3, 0.8),
            productivities=(1.1, 0.95, 1.0),
        )
        data_set = MulticountryDataSettings(
            train_T=40.0,
            train_points=36,
            dense_T=20.0,
            tail_points=8,
            test_T=20.0,
            test_points=21,
            validation_points=31,
            benchmark_T=50.0,
            benchmark_points=51,
        )
        outputs = {}
        for solver_name in ("auto", "cg"):
            results, _ = multicountry_growth_kernel(
                parameters=parameters,
                data_set=data_set,
                lm_set=MulticountrySolverSettings(linear_solver=solver_name),
                sensitivity_time=10.0,
            )
            self.assertTrue(results["solution_accepted"], results["rejection_reason"])
            outputs[solver_name] = results
        cg = outputs["cg"]
        self.assertEqual(cg["linear_solver"], "cg")
        self.assertEqual(cg["preconditioner"], "block_eigen")
        self.assertEqual(cg["cg_maxiter"], MulticountrySolverSettings().cg_maxiter)
        self.assertAlmostEqual(
            cg["rkhs_norm_total"],
            outputs["auto"]["rkhs_norm_total"],
            delta=1e-6 * max(outputs["auto"]["rkhs_norm_total"], 1.0),
        )
        # The implicit-AD comparative statics agree across the dense and
        # matrix-free AD paths (the cg tangent/cotangent solve runs the
        # block-eigen-preconditioned CG to tolerance at zero damping).
        for name in ("alpha", "gamma", "delta", "rho", "installation_elasticity"):
            self.assertAlmostEqual(
                cg[f"d_rate_d_{name}"],
                outputs["auto"][f"d_rate_d_{name}"],
                delta=max(1e-6, 1e-3 * abs(outputs["auto"][f"d_rate_d_{name}"])),
            )
        for country in (1, 2, 3):
            for name in ("productivity", "population", "k_0"):
                self.assertAlmostEqual(
                    cg[f"d_rate_d_{name}_{country}"],
                    outputs["auto"][f"d_rate_d_{name}_{country}"],
                    delta=max(
                        1e-6,
                        1e-3 * abs(outputs["auto"][f"d_rate_d_{name}_{country}"]),
                    ),
                )

    def test_multicountry_growth_sensitivity(self):
        # Implicit-AD comparative statics of the global rate at t=10 with
        # respect to every structural parameter and initial condition,
        # validated against central finite differences of full re-solves.
        # The AD-vs-FD gap is the Gauss-Newton implicit-sensitivity bias at
        # a NON-interpolating stationary point (the dropped curvature terms
        # scale with the collocation residual: measured 7e-3 relative at
        # max_train ~ 9e-7 on this grid, collapsing to 3e-4 at ~1e-8 on a
        # near-steady problem), so the bound is 2e-2 -- FD itself is stable
        # to ~1e-4 across a 50x step range.
        parameters = MulticountryParameters(
            populations=(1.0, 1.4),
            productivities=(1.05, 0.95),
        )
        k_0, a_0 = draw_initial_conditions(InitialConditionSettings(), parameters)
        data_set = MulticountryDataSettings(
            train_T=40.0,
            train_points=36,
            dense_T=20.0,
            tail_points=8,
            test_T=20.0,
            test_points=21,
            validation_points=31,
            benchmark_T=50.0,
            benchmark_points=51,
        )

        def rate_at_10(**overrides):
            call = dict(k_0=k_0, a_0=a_0, parameters=parameters, data_set=data_set)
            call.update(overrides)
            _, extras = multicountry_growth_kernel(**call)
            return float(extras["model"](jnp.asarray([10.0]))["interest_rate"][0])

        results, extras = multicountry_growth_kernel(
            k_0=k_0,
            a_0=a_0,
            parameters=parameters,
            data_set=data_set,
            sensitivity_time=10.0,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["sensitivity_time"], 10.0)
        self.assertGreater(results["interest_rate_at_sensitivity_time"], 0.0)
        sensitivity_df = extras["sensitivity_df"]
        self.assertEqual(len(sensitivity_df), 5 + 4 * 2)
        self.assertTrue(np.all(np.isfinite(sensitivity_df["value"])))

        # Central FD of full re-solves against the reverse-mode values.
        h_rho = 4e-4
        fd_rho = (
            rate_at_10(
                parameters=dataclasses.replace(parameters, rho=parameters.rho + h_rho)
            )
            - rate_at_10(
                parameters=dataclasses.replace(parameters, rho=parameters.rho - h_rho)
            )
        ) / (2.0 * h_rho)
        self.assertAlmostEqual(
            results["d_rate_d_rho"], fd_rho, delta=2e-2 * abs(fd_rho)
        )
        h_productivity = 1e-3
        fd_productivity = (
            rate_at_10(
                parameters=dataclasses.replace(
                    parameters, productivities=(1.05 + h_productivity, 0.95)
                )
            )
            - rate_at_10(
                parameters=dataclasses.replace(
                    parameters, productivities=(1.05 - h_productivity, 0.95)
                )
            )
        ) / (2.0 * h_productivity)
        self.assertAlmostEqual(
            results["d_rate_d_productivity_1"],
            fd_productivity,
            delta=2e-2 * abs(fd_productivity),
        )
        h_capital = 1e-3 * k_0[0]
        fd_capital = (
            rate_at_10(k_0=(k_0[0] + h_capital, k_0[1]))
            - rate_at_10(k_0=(k_0[0] - h_capital, k_0[1]))
        ) / (2.0 * h_capital)
        self.assertAlmostEqual(
            results["d_rate_d_k_0_1"], fd_capital, delta=2e-2 * abs(fd_capital)
        )

        # Redistribution neutrality: identical homothetic preferences make
        # the global rate invariant to feasible reshuffles of the initial
        # NFA positions, so the projected a_0 gradient is ~0 (five orders
        # below the other sensitivities) and sums to zero exactly by the
        # demeaning projection. The FD check runs along a feasible
        # direction (population-weighted sum preserved).
        self.assertLess(abs(results["d_rate_d_a_0_1"]), 1e-5)
        self.assertLess(abs(results["d_rate_d_a_0_2"]), 1e-5)
        self.assertAlmostEqual(
            results["d_rate_d_a_0_1"] + results["d_rate_d_a_0_2"], 0.0, places=12
        )
        h_asset = 1e-3
        direction = (1.0, -1.0 / 1.4)
        fd_asset = (
            rate_at_10(
                a_0=(a_0[0] + h_asset * direction[0], a_0[1] + h_asset * direction[1])
            )
            - rate_at_10(
                a_0=(a_0[0] - h_asset * direction[0], a_0[1] - h_asset * direction[1])
            )
        ) / (2.0 * h_asset)
        ad_asset = (
            results["d_rate_d_a_0_1"] * direction[0]
            + results["d_rate_d_a_0_2"] * direction[1]
        )
        self.assertLess(abs(fd_asset), 2e-5)
        self.assertLess(abs(ad_asset), 2e-5)

    def test_multicountry_growth_nn_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.4),
            productivities=(1.05, 0.95),
        )
        results, extras = multicountry_growth_nn(
            parameters=parameters,
            seed=0,
            verbose=False,
            long_run_diagnostics=True,
        )
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(
            results["equation_diagnostics_pass"],
            results["equation_diagnostic_reason"],
        )
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["stopping_reason"], "early_stopping_met")
        # Optional long-run diagnostics mirror the kernel driver's: theory
        # steady state reached by t=100 and the sharing invariants hold
        # (the ratio drift is looser than the kernel's, tracking the NN's
        # higher train residual; measured 2e-5 at this seed).
        self.assertLess(results["long_run_k_theory_gap"], 0.03)
        self.assertLess(results["long_run_consumption_ratio_drift"], 1e-4)
        self.assertLess(results["long_run_wealth_proportionality_gap"], 1e-3)
        self.assertTrue(results["long_run_c_rank_matches_wealth"])
        self.assertEqual(results["long_run_wealth_rank_violation"], 0.0)
        self.assertEqual(results["attempt"], 1)
        self.assertEqual(results["linear_solver"], "auto")
        self.assertEqual(results["nn_width"], 32)
        self.assertEqual(results["nn_depth"], 3)
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
        # 2e-5 under the drawn heterogeneous initial conditions (measured
        # 1.17e-5 at the default seed; 1e-5 was the symmetric-k_0 margin).
        self.assertLess(results["max_train_residual"], 2e-5)
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
        # The paths pin the DRAWN initial conditions exactly.
        self.assertAlmostEqual(initial["k_1"], results["k_0_1"], places=12)
        self.assertAlmostEqual(initial["k_2"], results["k_0_2"], places=12)
        self.assertNotAlmostEqual(results["k_0_1"], results["k_0_2"], places=3)
        self.assertAlmostEqual(initial["a_1"], results["a_0_1"], places=12)
        self.assertAlmostEqual(initial["a_2"], results["a_0_2"], places=12)
        self.assertNotAlmostEqual(results["a_0_1"], results["a_0_2"], places=3)

    def test_multicountry_growth_nn_three_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.3, 0.8),
            productivities=(1.1, 0.95, 1.0),
        )
        results, extras = multicountry_growth_nn(
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
        # Distinct drawn k_0 across countries 2 and 3: weights positive and
        # genuinely different (the old equality assumed identical capital).
        self.assertGreater(results["relative_weight_2"], 0.0)
        self.assertGreater(results["relative_weight_3"], 0.0)
        self.assertNotAlmostEqual(
            results["relative_weight_2"],
            results["relative_weight_3"],
            places=3,
        )
        self.assertIn("k_3", extras["test_df"])
        self.assertIn("marginal_wealth_3", extras["test_df"])

    def test_multicountry_growth_three_country_smoke(self):
        parameters = MulticountryParameters(
            populations=(1.0, 1.3, 0.8),
            productivities=(1.1, 0.95, 1.0),
        )
        results, extras = multicountry_growth_kernel(parameters=parameters)
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertFalse(results["benchmark_available"])
        self.assertEqual(results["num_countries"], 3)
        self.assertEqual(results["dynamic_rkhs_function_count"], 12)
        self.assertEqual(results["algebraic_rkhs_function_count"], 1)
        self.assertEqual(results["free_initial_value_count"], 7)
        self.assertEqual(results["metric_selection_dimension"], 6)
        self.assertEqual(results["unknown_count"], results["dae_residual_count"] + 6)
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        # Long-run validity at t=100: theory steady state reached, exact
        # sharing invariants at solver accuracy, consumption ranked by
        # initial total wealth.
        self.assertLess(results["long_run_k_theory_gap"], 0.03)
        self.assertLess(results["long_run_consumption_ratio_drift"], 1e-5)
        self.assertLess(results["long_run_wealth_proportionality_gap"], 1e-3)
        self.assertTrue(results["long_run_c_rank_matches_wealth"])
        self.assertEqual(results["long_run_wealth_rank_violation"], 0.0)
        self.assertTrue(results["long_run_terminal_a_rank_matches_a_0"])
        # Drawn k_0 and a_0 differ across countries, so the welfare weights
        # must be positive and genuinely distinct (the old equality held
        # only under identical initial conditions).
        self.assertGreater(results["relative_weight_2"], 0.0)
        self.assertGreater(results["relative_weight_3"], 0.0)
        self.assertNotAlmostEqual(
            results["relative_weight_2"],
            results["relative_weight_3"],
            places=3,
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
        self.assertEqual(results["selection"], "minimum_rkhs_norm_ridge")
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
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["q"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())
        self.assertTrue((extras["test_df"]["x"] > 0.0).all())
        self.assertLess(extras["train_df"]["marginal_wealth"].abs().max(), 1e-6)
        self.assertLess(extras["train_df"]["consumption_foc"].abs().max(), 2e-6)

        short_horizon, _ = open_economy_growth_kernel(
            data_set=OpenEconomyDataSettings(
                train_T=80.0,
                train_points=86,
                tail_points=5,
                benchmark_T=120.0,
            )
        )
        self.assertTrue(
            short_horizon["solution_accepted"], short_horizon["rejection_reason"]
        )
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

    def test_open_economy_growth_kernel_uno_matches_dae_formulation(self):
        results, extras = open_economy_growth_kernel_uno()
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["solver_status"], "SUCCESS")
        self.assertEqual(results["solution_status"], "FEASIBLE_KKT_POINT")
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
        self.assertEqual(results["selection"], "minimum_rkhs_norm_constrained")
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 1e-2)
        self.assertLess(results["max_benchmark_error"], 1.5e-3)
        self.assertAlmostEqual(
            results["objective_value"],
            results["rkhs_norm_total"],
            places=12,
        )
        self.assertNotIn("metric_eps", results)
        self.assertNotIn("shifted_metric_norm_squared", results)
        self.assertNotIn("metric_eps", extras["hyperparameters"]["solver_set"])
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["q"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

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
        self.assertLess(results["max_validation_residual"], 1e-2)
        self.assertLess(results["c_relative_error_max"], 2e-4)
        self.assertLess(results["max_benchmark_error"], 2e-2)

        path = extras["train_df"]
        realized_growth = math.log(path["c"].iloc[-1] / path["c"].iloc[0]) / (
            path["t"].iloc[-1] - path["t"].iloc[0]
        )
        expected_growth = (parameters.interest_rate - parameters.rho) / parameters.gamma
        self.assertAlmostEqual(realized_growth, expected_growth, places=5)
        self.assertLess(path["marginal_wealth"].abs().max(), 1e-6)
        self.assertLess(path["consumption_foc"].abs().max(), 2e-6)

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

        self.assertEqual(results["linear_solver"], "cholesky")
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

        self.assertEqual(reference["linear_solver"], "cholesky")
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

                self.assertEqual(results["linear_solver"], "cholesky")
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

    def test_asset_pricing_ridge_must_be_positive(self):
        with self.assertRaises(ValueError):
            asset_pricing_kernel(lm_set=SolverSettings(ridge=0.0))

    def test_optimal_advertising_kernel_smoke(self):
        results, extras = optimal_advertising_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertAlmostEqual(results["mu_0"], 0.8835154870013, places=10)
        self.assertAlmostEqual(results["u_0"], 0.2650546436059, places=10)
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
