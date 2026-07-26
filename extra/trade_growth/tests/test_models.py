import unittest

import jax
import jax.numpy as jnp
import numpy as np

from extra.trade_growth.comparative_statics import (
    build_multicountry_trade_comparative_statics,
)
from extra.trade_growth.kernel import (
    DataSettings,
    SolverSettings,
    multicountry_trade_growth_kernel,
)
from extra.trade_growth.model import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    multicountry_trade_allocations,
    multicountry_trade_path_values,
    static_trade_equilibrium,
    symmetric_trade_growth_parameters,
)
from extra.trade_growth.nn import DataSettings as MulticountryTradeNNDataSettings
from extra.trade_growth.nn import SequentialNNSettings, multicountry_trade_growth_nn
from extra.trade_growth.nn import SolverSettings as NNSolverSettings

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class KernelSmokeTests(unittest.TestCase):
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
        results, frames = multicountry_trade_growth_kernel(
            a_0=(-0.2, 0.1, 0.1),
            parameters=parameters,
            data_set=DataSettings(
                train_T=15.0,
                train_points=6,
                sampling="uniform",
                dense_T=8.0,
                tail_points=2,
                test_T=8.0,
                test_points=9,
                validation_points=13,
                quality_T=10.0,
                rank_audit=True,
            ),
            lm_set=SolverSettings(
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
        self.assertEqual(results["quality_T"], 10.0)
        self.assertEqual(results["quality_validation_point_count"], 9)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_validation_residual"], 1e-3)
        self.assertGreaterEqual(
            results["max_full_horizon_validation_residual"],
            results["max_validation_residual"],
        )
        self.assertLess(results["max_asset_clearing_residual"], 5e-8)
        self.assertGreaterEqual(
            results["max_full_horizon_asset_clearing_residual"],
            results["max_asset_clearing_residual"],
        )
        self.assertLess(results["max_rate_consistency_residual"], 1e-8)
        self.assertGreaterEqual(
            results["max_full_horizon_rate_consistency_residual"],
            results["max_rate_consistency_residual"],
        )
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
        quality_frame = frames["validation_df"].query("t <= 10.0")
        self.assertLess(
            np.max(np.abs(quality_frame["consumption_basket_log_price"])),
            1e-4,
        )
        for name in (
            "ode_residual_rms",
            "algebraic_residual_rms",
            "dae_residual_rms",
        ):
            self.assertTrue(np.isfinite(quality_frame[name]).all())
            self.assertGreaterEqual(quality_frame[name].min(), 0.0)
            self.assertAlmostEqual(
                results[f"max_validation_{name}"], quality_frame[name].max()
            )
        np.testing.assert_allclose(
            quality_frame["dae_residual_rms"] ** 2,
            (
                12.0 * quality_frame["ode_residual_rms"] ** 2
                + 25.0 * quality_frame["algebraic_residual_rms"] ** 2
            )
            / 37.0,
            rtol=1e-12,
            atol=1e-16,
        )

    def test_multicountry_trade_iterative_whitened_endgame_api(self):
        results, _ = multicountry_trade_growth_kernel(
            a_0=(-0.1, 0.1),
            parameters=symmetric_trade_growth_parameters(2),
            data_set=DataSettings(
                train_T=8.0,
                train_points=4,
                sampling="uniform",
                dense_T=4.0,
                tail_points=1,
                test_T=4.0,
                test_points=5,
                validation_points=7,
                quality_T=4.0,
            ),
            lm_set=SolverSettings(
                linear_solver="gram_cg",
                max_steps=1,
                residual_atol=1e-7,
                cg_maxiter_min=20,
                cg_maxiter_max=50,
                recycle_rank=4,
                two_phase_endgame=True,
                phase2_max_steps=1,
                phase2_maxiter=100,
            ),
        )

        self.assertEqual(results["linear_solver_resolved"], "gram_cg")
        self.assertTrue(results["two_phase_endgame"])
        self.assertEqual(results["phase2_steps"], 1)
        self.assertTrue(np.isfinite(results["phase2_build_time"]))
        self.assertTrue(np.isfinite(results["phase2_solve_time"]))

    def test_multicountry_trade_structural_parameter_implicit_ad(self):
        parameters = symmetric_trade_growth_parameters(2)
        problem = build_multicountry_trade_comparative_statics(
            k_0=(1.0, 0.9),
            a_0=(-0.1, 0.1),
            parameters=parameters,
            data_set=DataSettings(
                train_T=10.0,
                train_points=5,
                sampling="uniform",
                dense_T=5.0,
                tail_points=1,
                test_T=10.0,
                test_points=11,
                validation_points=21,
                quality_T=10.0,
            ),
            lm_set=SolverSettings(
                init_damping=1e-8,
                residual_atol=1e-12,
                max_steps=40,
            ),
            statistic_times=(0.0, 5.0, 10.0),
        )
        p = problem.structural_parameters
        p_dot = jax.tree.map(jnp.zeros_like, p)
        p_dot = {**p_dot, "rho": p["rho"]}
        result, result_dot = jax.jvp(problem.solve, (p,), (p_dot,))

        def residual(theta, parameters_pytree):
            return problem.solver.residual_fn(theta, None, parameters_pytree)[0]

        _, linearized_residual = jax.jvp(
            residual,
            (result.x, p),
            (result_dot.x, p_dot),
        )
        self.assertEqual(int(result.status), 1)
        self.assertLess(float(jnp.max(jnp.abs(residual(result.x, p)))), 1e-10)
        self.assertLess(float(jnp.max(jnp.abs(linearized_residual))), 1e-8)
        self.assertTrue(bool(jnp.all(jnp.isfinite(result_dot.aux))))
        self.assertIn("initial_log_capital[1]", problem.parameter_names)
        self.assertIn("initial_aggregate_assets_free[1]", problem.parameter_names)

        step = 1e-3
        p_plus = jax.tree.map(lambda value, dot: value + step * dot, p, p_dot)
        p_minus = jax.tree.map(lambda value, dot: value - step * dot, p, p_dot)
        theta_plus = jax.tree.map(
            lambda value, dot: value + step * dot,
            result.x,
            result_dot.x,
        )
        theta_minus = jax.tree.map(
            lambda value, dot: value - step * dot,
            result.x,
            result_dot.x,
        )
        result_plus = problem.solver.solve(
            theta_plus,
            p=p_plus,
            max_steps=40,
            atol=1e-14,
            max_steps_is_success=False,
        )
        result_minus = problem.solver.solve(
            theta_minus,
            p=p_minus,
            max_steps=40,
            atol=1e-14,
            max_steps_is_success=False,
        )
        finite_difference = (result_plus.aux - result_minus.aux) / (2.0 * step)
        relative_error = jnp.linalg.norm(
            finite_difference - result_dot.aux
        ) / jnp.linalg.norm(result_dot.aux)
        self.assertLess(float(relative_error), 1e-4)

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
                quality_T=10.0,
            ),
            nn_set=SequentialNNSettings(
                width=16,
                depth=2,
                input_scale=10.0,
            ),
            lm_set=NNSolverSettings(
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
        self.assertEqual(results["quality_T"], 10.0)
        self.assertEqual(results["quality_validation_point_count"], 9)
        self.assertLess(results["max_initial_residual"], 1e-12)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_validation_residual"], 1e-3)
        self.assertGreaterEqual(
            results["max_full_horizon_validation_residual"],
            results["max_validation_residual"],
        )
        self.assertLess(results["max_asset_clearing_residual"], 5e-5)
        self.assertGreaterEqual(
            results["max_full_horizon_asset_clearing_residual"],
            results["max_asset_clearing_residual"],
        )
        self.assertLess(results["max_rate_consistency_residual"], 1e-5)
        self.assertGreaterEqual(
            results["max_full_horizon_rate_consistency_residual"],
            results["max_rate_consistency_residual"],
        )
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
        quality_frame = frames["validation_df"].query("t <= 10.0")
        self.assertLess(
            np.max(np.abs(quality_frame["consumption_basket_log_price"])), 1e-4
        )
        for name in (
            "ode_residual_rms",
            "algebraic_residual_rms",
            "dae_residual_rms",
        ):
            self.assertTrue(np.isfinite(quality_frame[name]).all())
            self.assertGreaterEqual(quality_frame[name].min(), 0.0)
            self.assertAlmostEqual(
                results[f"max_validation_{name}"], quality_frame[name].max()
            )
        np.testing.assert_allclose(
            quality_frame["dae_residual_rms"] ** 2,
            (
                12.0 * quality_frame["ode_residual_rms"] ** 2
                + 25.0 * quality_frame["algebraic_residual_rms"] ** 2
            )
            / 37.0,
            rtol=1e-12,
            atol=1e-16,
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


if __name__ == "__main__":
    unittest.main()
