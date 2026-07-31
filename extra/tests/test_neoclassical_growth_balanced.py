import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from neoclassical_growth_nn_recursive import (
    DataSettings,
    StoppingCriteria,
    StructuralParameters,
    euler_residuals,
    neoclassical_growth_nn_recursive,
    positive_policy_grid,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class BalancedGrowthBenchmarkTests(unittest.TestCase):
    def test_invalid_effective_discount_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            neoclassical_growth_benchmark(
                1 / 3,
                0.1,
                0.01,
                2.0,
                1.0,
                np.linspace(0.0, 80.0, 300),
                g=-0.02,
            )

    def test_legacy_benchmark_is_unchanged(self):
        grid = np.linspace(0.0, 80.0, 300)
        times = np.asarray([0.0, 1.0, 5.0, 20.0, 50.0])
        legacy = neoclassical_growth_benchmark(1 / 3, 0.1, 0.11, 1.0, 1.0, grid)
        k_legacy, c_legacy = legacy(times)
        expected_k = np.asarray(
            [
                1.0,
                1.188575276907517,
                1.6576070888312837,
                1.9875268044182206,
                1.9997966951098445,
            ]
        )
        expected_c = np.asarray(
            [
                0.6933829286240208,
                0.7691993702895003,
                0.9428139396469232,
                1.0558069533242005,
                1.0598952703137345,
            ]
        )

        np.testing.assert_allclose(k_legacy, expected_k, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(c_legacy, expected_c, rtol=2e-13, atol=2e-14)

        scaled = neoclassical_growth_benchmark(
            1 / 3, 0.1, 0.11, 1.0, 2.0, grid, z_0=2.0, g=0.0
        )
        k_scaled, c_scaled = scaled(times)
        np.testing.assert_allclose(k_scaled, 2.0 * k_legacy, rtol=2e-13)
        np.testing.assert_allclose(c_scaled, 2.0 * c_legacy, rtol=2e-13)

    def test_benchmark_balanced_growth_and_homogeneity(self):
        a, delta, rho_hat, sigma_crra, g = 1 / 3, 0.1, 0.11, 1.0, 0.02
        kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1 / (a - 1))
        z_0 = 1.3
        k_0 = z_0 * kappa_ss
        grid = np.linspace(0.0, 80.0, 300)
        times = np.linspace(0.0, 50.0, 51)
        baseline = neoclassical_growth_benchmark(
            a,
            delta,
            rho_hat,
            sigma_crra,
            k_0,
            grid,
            z_0=z_0,
            g=g,
        )
        scaled = neoclassical_growth_benchmark(
            a,
            delta,
            rho_hat,
            sigma_crra,
            2.5 * k_0,
            grid,
            z_0=2.5 * z_0,
            g=g,
        )
        k, c = baseline(times)
        k_scaled, c_scaled = scaled(times)
        z = z_0 * np.exp(g * times)

        np.testing.assert_allclose(k / z, kappa_ss, rtol=2e-13)
        np.testing.assert_allclose(c / z, (c / z)[0], rtol=2e-13)
        np.testing.assert_allclose(k_scaled, 2.5 * k, rtol=2e-13)
        np.testing.assert_allclose(c_scaled, 2.5 * c, rtol=2e-13)


class BalancedGrowthRecursiveNNTests(unittest.TestCase):
    def test_policy_grid_margin_stays_multiplicatively_close_to_support(self):
        grid = positive_policy_grid(0.095, 4.34, 100)

        self.assertGreater(grid[0], 0.05)
        self.assertLess(grid[-1], 6.0)
        self.assertTrue(np.all(np.diff(grid) > 0.0))

    def test_benchmark_horizon_must_cover_training_policy_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "both train_T and test_T"):
            neoclassical_growth_nn_recursive(
                verbose=False,
                data_set=DataSettings(
                    train_T=90.0,
                    train_tvc_T=100.0,
                    test_T=10.0,
                    benchmark_T=80.0,
                ),
            )

    def test_g_zero_transient_residual_matches_legacy_one_state_equation(self):
        a, delta, rho_hat = 1 / 3, 0.1, 0.11
        p = StructuralParameters.create(
            jnp.float64,
            a=a,
            delta=delta,
            rho_hat=rho_hat,
            sigma_crra=1.0,
            g=0.0,
            z_0=jnp.asarray([1.0]),
            k_0=jnp.asarray([1.0]),
            A=1.0,
            b_1=1.0,
            b_2=0.0,
            kappa_thr=jnp.inf,
        )

        def policy(log_state, parameters):
            z, k = jnp.exp(log_state)
            share = jax.nn.sigmoid(0.3 + 0.2 * (log_state[1] - log_state[0]))
            return z * (k / z) ** parameters.a * share

        capital = jnp.asarray([0.8, 1.3, 2.2])
        log_states = jnp.stack([jnp.zeros_like(capital), jnp.log(capital)], axis=1)
        two_state = euler_residuals(policy, log_states, p)

        def legacy_residual(k):
            def log_consumption(state):
                share = jax.nn.sigmoid(0.3 + 0.2 * jnp.log(state))
                return jnp.log(state**a * (1.0 - share))

            _, dlog_c_dk = jax.jvp(log_consumption, (k,), (jnp.ones_like(k),))
            share = jax.nn.sigmoid(0.3 + 0.2 * jnp.log(k))
            investment = k**a * share
            return dlog_c_dk * (investment - delta * k) - (
                a * k ** (a - 1.0) - delta - rho_hat
            )

        one_state = jax.vmap(legacy_residual)(capital)
        np.testing.assert_allclose(two_state, one_state, rtol=2e-14, atol=2e-14)

    def test_two_state_euler_residual_on_balanced_growth(self):
        a, delta, rho_hat, sigma_crra, g = 1 / 3, 0.1, 0.11, 1.0, 0.02
        kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1 / (a - 1))
        p = StructuralParameters.create(
            jnp.float64,
            a=a,
            delta=delta,
            rho_hat=rho_hat,
            sigma_crra=sigma_crra,
            g=g,
            z_0=jnp.asarray([1.0]),
            k_0=jnp.asarray([kappa_ss]),
            A=1.0,
            b_1=1.0,
            b_2=0.0,
            kappa_thr=jnp.inf,
        )

        def balanced_policy(log_state, parameters):
            k = jnp.exp(log_state[1])
            return (parameters.delta + parameters.g) * k

        times = jnp.linspace(0.0, 20.0, 11)
        log_states = jnp.stack([g * times, g * times + jnp.log(kappa_ss)], axis=1)
        residual = euler_residuals(balanced_policy, log_states, p)

        np.testing.assert_allclose(np.asarray(residual), 0.0, atol=2e-14)

    def test_multiple_initial_conditions_are_vmapped_and_stacked(self):
        data_set = DataSettings(
            train_T=20.0,
            train_points=33,
            train_tvc_T=50.0,
            test_T=10.0,
            test_points=11,
            num_test_trajectories=6,
        )
        results, extras = neoclassical_growth_nn_recursive(
            z_0=[1.0, 1.0],
            k_0=[0.9, 1.1],
            g=0.02,
            verbose=False,
            data_set=data_set,
            stop_set=StoppingCriteria(max_steps=200, num_attempts=1),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertTrue(results["transversality_ok"])
        self.assertEqual(results["num_train_trajectories"], 2)
        self.assertEqual(results["num_collocation_states"], 66)
        self.assertEqual(extras["train_df"]["trajectory"].nunique(), 2)
        self.assertEqual(len(extras["train_df"]), 66)
        self.assertEqual(extras["test_df"]["trajectory"].nunique(), 6)
        self.assertEqual(len(extras["test_df"]), 6 * 11)
        self.assertEqual(len(extras["test_summary_df"]), 6)
        self.assertLess(results["test_median_abs_k_rel_error"], 1e-3)
        self.assertLess(results["test_median_abs_c_rel_error"], 1e-3)
        first = extras["test_df"].query("t == 0.0")
        np.testing.assert_allclose(first["z"], first["z_0"], rtol=2e-14)
        np.testing.assert_allclose(first["k"], first["k_0"], rtol=2e-14)


if __name__ == "__main__":
    unittest.main()
