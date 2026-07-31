from __future__ import annotations

import dataclasses
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from extra.trade_growth.kernel import (
    DataSettings,
    SolverSettings,
    multicountry_trade_growth_kernel,
)
from extra.trade_growth.model import (
    DYNAMIC_RESIDUAL_NAMES,
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    multicountry_trade_allocations,
    multicountry_trade_path_values,
    static_trade_equilibrium,
)
from extra.trade_growth.scaling import draw_trade_growth_parameters
from multicountry_growth import InitialConditionSettings, draw_initial_conditions

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def small_data(rank_audit=False):
    return DataSettings(
        train_T=15.0,
        train_points=6,
        sampling="front_loaded",
        dense_T=7.5,
        tail_points=2,
        test_T=7.5,
        test_points=13,
        validation_points=25,
        quality_T=15.0,
        rank_audit=rank_audit,
    )


class TradeGrowthEquations(unittest.TestCase):
    def test_heterogeneous_static_equilibrium_and_accounting(self):
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
            q=jnp.full(3, 1.0 / parameters.eta),
            mu=jnp.asarray((1.0, 1.1, 0.9)),
            parameters=parameters,
        )
        self.assertLess(diagnostics["max_residual"], 1e-10)
        self.assertGreater(diagnostics["minimum_jacobian_singular_value"], 1e-3)
        np.testing.assert_allclose(
            values["trade_shares"].sum(axis=1),
            1.0,
            atol=1e-13,
        )
        np.testing.assert_allclose(values["budget_trade_gap"], 0.0, atol=1e-13)
        np.testing.assert_allclose(values["world_trade_balance"], 0.0, atol=1e-13)
        self.assertFalse(np.allclose(values["trade_shares"], values["trade_shares"].T))

    def test_dynamic_residual_signs_match_the_laws_of_motion(self):
        parameters = draw_trade_growth_parameters(3, seed=11)
        k = jnp.asarray([[0.9, 1.1, 1.3]])
        q = jnp.asarray([[0.8, 0.9, 1.0]])
        mu = jnp.asarray([[1.1, 0.95, 1.05]])
        interest_rate = jnp.asarray([0.04])
        a = jnp.asarray([[-0.15, 0.02, 0.1]])
        static, diagnostics = static_trade_equilibrium(
            k[0],
            q[0],
            mu[0],
            parameters,
        )
        self.assertLess(diagnostics["max_residual"], 1e-10)
        log_static = static["log_static"][None, :, :]
        allocations = multicountry_trade_allocations(
            k,
            q,
            mu,
            interest_rate,
            log_static,
            parameters,
        )
        budget_drift = (
            interest_rate[:, None] * a
            + allocations["factor_income"]
            - allocations["final_expenditure"]
        )
        values = multicountry_trade_path_values(
            k=k,
            a=a,
            q=q,
            mu=mu,
            interest_rate=interest_rate,
            log_static=log_static,
            dlog_k_dt=allocations["capital_drift"] / k,
            da_dt=budget_drift,
            dlog_q_dt=allocations["tobin_q_drift"] / q,
            dlog_mu_dt=(parameters.r - interest_rate[:, None]),
            parameters=parameters,
        )
        for name in DYNAMIC_RESIDUAL_NAMES:
            self.assertLess(float(jnp.max(jnp.abs(values[name]))), 2e-15)
        self.assertLess(
            float(jnp.max(jnp.abs(values["static_residuals"]))),
            1e-10,
        )


class FullDAEKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for num_countries in (2, 3):
            parameters = draw_trade_growth_parameters(
                num_countries,
                seed=0,
            )
            k_0, a_0 = draw_initial_conditions(
                InitialConditionSettings(seed=0),
                parameters,
                dtype=jnp.float64,
            )
            dense_results, dense_extra = multicountry_trade_growth_kernel(
                k_0=k_0,
                a_0=a_0,
                parameters=parameters,
                data_set=small_data(rank_audit=True),
                lm_set=SolverSettings(linear_solver="auto", max_steps=100),
            )
            cg_results, cg_extra = multicountry_trade_growth_kernel(
                k_0=k_0,
                a_0=a_0,
                parameters=parameters,
                data_set=small_data(),
                lm_set=SolverSettings(linear_solver="cg", max_steps=100),
            )
            cls.cases[num_countries] = {
                "parameters": parameters,
                "k_0": k_0,
                "a_0": a_0,
                "dense": (dense_results, dense_extra),
                "cg": (cg_results, cg_extra),
            }

    def test_two_and_three_country_dae_counts_rank_and_random_initials(self):
        for num_countries, case in self.cases.items():
            with self.subTest(num_countries=num_countries):
                results, extra = case["dense"]
                n_train = results["num_train_points"]
                residual_count = (12 * num_countries + 1) * n_train
                free_count = 10 * num_countries + 1
                theta_flat, _ = ravel_pytree(extra["theta"])
                self.assertTrue(
                    results["solution_accepted"],
                    results["rejection_reason"],
                )
                self.assertFalse(results["asset_homotopy_used"])
                self.assertEqual(results["linear_solver_resolved"], "cholesky")
                self.assertEqual(results["dae_residual_count"], residual_count)
                self.assertEqual(results["free_initial_value_count"], free_count)
                self.assertEqual(results["unknown_count"], residual_count + free_count)
                self.assertEqual(extra["metric"].size, residual_count)
                self.assertEqual(theta_flat.size - extra["metric"].size, free_count)
                self.assertEqual(results["dae_jacobian_rank"], residual_count)
                self.assertLess(results["max_train_residual"], 1e-8)
                self.assertLess(results["max_validation_residual"], 2e-3)
                self.assertGreater(float(jnp.std(extra["p"]["k_0"])), 1e-3)
                self.assertGreater(float(jnp.max(jnp.abs(extra["p"]["a_0"]))), 1e-3)

    def test_initial_levels_are_actual_free_unpenalized_unknowns(self):
        for num_countries, case in self.cases.items():
            with self.subTest(num_countries=num_countries):
                _, extra = case["dense"]
                initial = extra["theta"][-1]
                values = extra["evaluate_parametric"](
                    jnp.asarray([0.0]),
                    extra["theta"],
                    extra["p"],
                )
                np.testing.assert_allclose(
                    values["q"][0],
                    jnp.exp(initial[:num_countries]),
                    rtol=2e-13,
                )
                np.testing.assert_allclose(
                    values["marginal_wealth"][0],
                    jnp.exp(initial[num_countries : 2 * num_countries]),
                    rtol=2e-13,
                )
                log_static_0 = initial[2 * num_countries : -1].reshape(
                    len(STATIC_BLOCK_NAMES),
                    num_countries,
                )
                for index, name in enumerate(STATIC_BLOCK_NAMES):
                    np.testing.assert_allclose(
                        values[name][0],
                        jnp.exp(log_static_0[index]),
                        rtol=2e-13,
                    )
                np.testing.assert_allclose(
                    values["interest_rate"][0],
                    jnp.exp(initial[-1]),
                    rtol=2e-13,
                )
                theta_flat, _ = ravel_pytree(extra["theta"])
                metric_size = extra["metric"].size
                penalty = jnp.vdot(
                    extra["metric"].factor_apply(
                        theta_flat[:metric_size],
                        None,
                    ),
                    extra["metric"].factor_apply(
                        theta_flat[:metric_size],
                        None,
                    ),
                )
                shifted = theta_flat.at[metric_size:].add(
                    jnp.linspace(0.1, 0.2, theta_flat.size - metric_size)
                )
                shifted_penalty = jnp.vdot(
                    extra["metric"].factor_apply(shifted[:metric_size], None),
                    extra["metric"].factor_apply(shifted[:metric_size], None),
                )
                self.assertEqual(float(penalty), float(shifted_penalty))

    def test_initial_conditions_and_all_economic_inputs_are_in_p(self):
        for num_countries, case in self.cases.items():
            with self.subTest(num_countries=num_countries):
                _, extra = case["dense"]
                p = extra["p"]
                self.assertEqual(set(p), {"parameters", "k_0", "a_0"})
                np.testing.assert_allclose(p["k_0"], case["k_0"])
                np.testing.assert_allclose(p["a_0"], case["a_0"])
                self.assertEqual(
                    p["parameters"].trade_costs.shape,
                    (num_countries, num_countries),
                )
                self.assertTrue(
                    all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(p))
                )

    def test_balanced_cg_matches_dense_for_two_and_three_countries(self):
        names = (
            "k",
            "a",
            "q",
            "marginal_wealth",
            "consumption",
            "investment",
            "interest_rate",
            "trade_shares",
        )
        points = jnp.linspace(0.0, 10.0, 21)
        for num_countries, case in self.cases.items():
            with self.subTest(num_countries=num_countries):
                dense_results, dense_extra = case["dense"]
                cg_results, cg_extra = case["cg"]
                self.assertTrue(
                    cg_results["solution_accepted"],
                    cg_results["rejection_reason"],
                )
                self.assertEqual(cg_results["linear_solver_resolved"], "cg")
                self.assertEqual(
                    cg_results["lm_set"]["preconditioner"],
                    "balanced_modes",
                )
                dense_values = dense_extra["model"](points)
                cg_values = cg_extra["model"](points)
                for name in names:
                    scale = max(
                        float(jnp.max(jnp.abs(dense_values[name]))),
                        1.0,
                    )
                    error = float(
                        jnp.max(jnp.abs(cg_values[name] - dense_values[name])) / scale
                    )
                    self.assertLess(error, 2e-7, name)
                self.assertLessEqual(
                    cg_results["max_validation_residual"],
                    1.05 * dense_results["max_validation_residual"],
                )

    def test_dense_implicit_ad_matches_the_gauss_newton_system(self):
        _, extra = self.cases[2]["dense"]
        p = extra["p"]
        zero = jax.tree.map(jnp.zeros_like, p)
        trade_cost_direction = (
            jnp.zeros_like(p["parameters"].trade_costs).at[0, 1].set(1.0)
        )
        direction = {
            **zero,
            "parameters": dataclasses.replace(
                zero["parameters"],
                trade_costs=trade_cost_direction,
            ),
            "k_0": zero["k_0"].at[0].set(0.2),
        }
        points = jnp.asarray([2.5])

        def observable(p_value):
            solved = extra["solve_parametric"](p_value)
            values = extra["evaluate_parametric"](
                points,
                solved.x,
                p_value,
            )
            return jnp.log(values["consumption"][0, 0])

        value, tangent = jax.jvp(observable, (p,), (direction,))
        _, pullback = jax.vjp(observable, p)
        gradient = pullback(jnp.ones_like(value))[0]
        reverse_directional = sum(
            jnp.vdot(gradient_leaf, direction_leaf)
            for gradient_leaf, direction_leaf in zip(
                jax.tree.leaves(gradient),
                jax.tree.leaves(direction),
                strict=True,
            )
        )
        self.assertTrue(bool(jnp.isfinite(tangent)))
        self.assertLess(
            float(jnp.abs(tangent - reverse_directional))
            / max(float(jnp.abs(tangent)), 1e-12),
            1e-4,
        )

        solved = extra["solve_parametric"](p)
        theta_flat, unravel_theta = ravel_pytree(solved.x)

        def residual(flat, p_value):
            return extra["residual_fn"](
                unravel_theta(flat),
                None,
                p_value,
            )

        jacobian = jax.jacrev(residual, argnums=0)(theta_flat, p)
        residual_p_dot = jax.jvp(
            lambda p_value: residual(theta_flat, p_value),
            (p,),
            (direction,),
        )[1]
        metric = extra["metric"]
        metric_size = metric.size
        jacobian_transpose = jacobian.T
        whitened_jacobian_transpose = jnp.concatenate(
            [
                metric.factor_solve_transpose(
                    jacobian_transpose[:metric_size],
                    None,
                ),
                jacobian_transpose[metric_size:],
            ],
            axis=0,
        )
        ridge = solved.lm_state.ridge
        normal = whitened_jacobian_transpose @ whitened_jacobian_transpose.T
        diagonal = jnp.arange(metric_size)
        normal = normal.at[diagonal, diagonal].add(ridge)
        factor = jsp_linalg.cho_factor(normal)
        y_dot = jsp_linalg.cho_solve(
            factor,
            -(whitened_jacobian_transpose @ residual_p_dot),
        )
        theta_dot_flat = jnp.concatenate(
            [
                metric.factor_solve(y_dot[:metric_size], None),
                y_dot[metric_size:],
            ]
        )

        def explicit_observable(flat, p_value):
            values = extra["evaluate_parametric"](
                points,
                unravel_theta(flat),
                p_value,
            )
            return jnp.log(values["consumption"][0, 0])

        _, manual_tangent = jax.jvp(
            explicit_observable,
            (theta_flat, p),
            (theta_dot_flat, direction),
        )
        self.assertLess(
            float(jnp.abs(tangent - manual_tangent))
            / max(float(jnp.abs(tangent)), 1e-12),
            2e-6,
        )

    def test_preconditioned_cg_ad_agrees_with_dense_ad(self):
        dense_extra = self.cases[2]["dense"][1]
        cg_extra = self.cases[2]["cg"][1]
        p = dense_extra["p"]
        zero = jax.tree.map(jnp.zeros_like, p)
        transfer = (
            jnp.zeros_like(p["a_0"])
            .at[0]
            .set(1.0 / p["parameters"].populations[0])
            .at[1]
            .set(-1.0 / p["parameters"].populations[1])
        )
        direction = {**zero, "a_0": transfer}
        points = jnp.asarray([2.5])

        def observable(extra_value, p_value):
            solved = extra_value["solve_parametric"](p_value)
            values = extra_value["evaluate_parametric"](
                points,
                solved.x,
                p_value,
            )
            return jnp.log(values["consumption"][0, 0])

        dense_tangent = jax.jvp(
            lambda p_value: observable(dense_extra, p_value),
            (p,),
            (direction,),
        )[1]
        cg_tangent = jax.jvp(
            lambda p_value: observable(cg_extra, p_value),
            (p,),
            (direction,),
        )[1]
        self.assertLess(
            float(jnp.abs(cg_tangent - dense_tangent))
            / max(float(jnp.abs(dense_tangent)), 1e-10),
            2e-5,
        )


class MetricLMKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = draw_trade_growth_parameters(2, seed=0)
        cls.k_0, cls.a_0 = draw_initial_conditions(
            InitialConditionSettings(seed=0),
            cls.parameters,
            dtype=jnp.float64,
        )
        cls.cases = {}
        common = {
            "optimizer": "metric_lm",
            "metric_eps": 1e-7,
            "init_damping": 1e-4,
            "geodesic_acceleration": False,
            "asset_homotopy_steps": 0,
            "max_steps": 100,
            "gtol": 5e-12,
        }
        for linear_solver, preconditioner in (
            ("auto", "balanced_modes"),
            ("qr", "balanced_modes"),
            ("gram_cg", "hybrid_blocks"),
        ):
            settings = SolverSettings(
                **common,
                linear_solver=linear_solver,
                preconditioner=preconditioner,
                preconditioner_refresh_steps=(1 if linear_solver == "gram_cg" else 0),
                cg_tol=1e-8,
                cg_maxiter=2000,
            )
            cls.cases[linear_solver] = multicountry_trade_growth_kernel(
                k_0=cls.k_0,
                a_0=cls.a_0,
                parameters=cls.parameters,
                data_set=small_data(),
                lm_set=settings,
            )

    def test_shifted_metric_is_exact_and_includes_initial_values(self):
        results, extra = self.cases["auto"]
        metric = extra["metric"]
        epsilon = results["metric_epsilon"]
        np.testing.assert_allclose(
            metric.F.T @ metric.F,
            extra["K"] + epsilon * jnp.eye(extra["K"].shape[0]),
            rtol=2e-13,
            atol=2e-13,
        )
        self.assertEqual(float(metric.free_scale), epsilon)
        self.assertEqual(results["penalized_coefficient_count"], 0)
        self.assertEqual(results["unpenalized_initial_value_count"], 0)
        theta_flat, _ = ravel_pytree(extra["theta"])
        metric_size = metric.size
        coefficient_norm = jnp.vdot(
            metric.factor_apply(theta_flat[:metric_size], None),
            metric.factor_apply(theta_flat[:metric_size], None),
        )
        full_norm = coefficient_norm + metric.free_scale * jnp.vdot(
            theta_flat[metric_size:], theta_flat[metric_size:]
        )
        shifted = theta_flat.at[metric_size:].add(0.1)
        shifted_norm = coefficient_norm + metric.free_scale * jnp.vdot(
            shifted[metric_size:], shifted[metric_size:]
        )
        self.assertNotEqual(float(full_norm), float(shifted_norm))

    def test_dense_qr_and_gram_cg_paths_agree(self):
        dense_results, dense_extra = self.cases["auto"]
        self.assertTrue(
            dense_results["solution_accepted"],
            dense_results["rejection_reason"],
        )
        points = jnp.linspace(0.0, 10.0, 21)
        dense_values = dense_extra["model"](points)
        for linear_solver in ("qr", "gram_cg"):
            with self.subTest(linear_solver=linear_solver):
                results, extra = self.cases[linear_solver]
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                values = extra["model"](points)
                errors = []
                for name in (
                    "k",
                    "a",
                    "q",
                    "marginal_wealth",
                    "consumption",
                    "investment",
                    "interest_rate",
                    "trade_shares",
                ):
                    scale = max(float(jnp.max(jnp.abs(dense_values[name]))), 1.0)
                    errors.append(
                        float(jnp.max(jnp.abs(values[name] - dense_values[name])))
                        / scale
                    )
                self.assertLess(max(errors), 2e-7)

    def test_metric_implicit_ad_is_finite_and_transpose_consistent(self):
        _, extra = self.cases["auto"]
        p = extra["p"]
        zero = jax.tree.map(jnp.zeros_like, p)
        direction = {
            **zero,
            "parameters": dataclasses.replace(
                zero["parameters"],
                trade_costs=zero["parameters"].trade_costs.at[0, 1].set(0.2),
            ),
            "k_0": zero["k_0"].at[0].set(0.1),
        }

        def observable(p_value):
            solved = extra["solve_parametric"](p_value)
            values = extra["evaluate_parametric"](jnp.asarray([2.5]), solved.x, p_value)
            return jnp.log(values["consumption"][0, 0])

        value, tangent = jax.jvp(observable, (p,), (direction,))
        _, pullback = jax.vjp(observable, p)
        gradient = pullback(jnp.ones_like(value))[0]
        reverse_directional = sum(
            jnp.vdot(gradient_leaf, direction_leaf)
            for gradient_leaf, direction_leaf in zip(
                jax.tree.leaves(gradient),
                jax.tree.leaves(direction),
                strict=True,
            )
        )
        self.assertTrue(bool(jnp.isfinite(tangent)))
        self.assertLess(float(jnp.abs(tangent - reverse_directional)), 2e-10)


if __name__ == "__main__":
    unittest.main()
