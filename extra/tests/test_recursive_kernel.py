import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neoclassical_growth_kernel_recursive import (
    DataSettings,
    InitialConditionSettings,
    StateKernelSettings,
    StoppingCriteria,
    benchmark_policy_grid,
    build_residual_args,
    fixed_collocation_data,
    neoclassical_growth_kernel_recursive,
    residual_fn,
)
from neoclassical_growth_nn_recursive import (
    DataSettings as NNDataSettings,
)
from neoclassical_growth_nn_recursive import (
    StructuralParameters,
    draw_initial_conditions,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def single_initial_condition(z_0=1.0, k_0=1.0):
    return draw_initial_conditions(
        z_0, k_0, 1, 0.0, 0.0, jax.random.key(0), jnp.float64
    )


class FixedRecursiveKernelTests(unittest.TestCase):
    def test_exact_path_centers_match_recursive_nn_grid(self):
        data_set = DataSettings()
        initial_conditions = single_initial_condition()
        train_t, train_log_states, train_c = fixed_collocation_data(
            1 / 3,
            0.1,
            0.11,
            1.0,
            0.02,
            initial_conditions,
            data_set,
            dtype=jnp.float64,
        )
        states = np.exp(np.asarray(train_log_states))

        self.assertEqual(train_t.shape, (1, 65))
        self.assertEqual(train_log_states.shape, (1, 65, 2))
        np.testing.assert_allclose(np.diff(train_t[0]), 1.5625, atol=1e-15)
        np.testing.assert_allclose(states[0, :, 0], np.exp(0.02 * train_t[0]))
        self.assertTrue(np.all(states > 0.0))
        self.assertTrue(np.all(np.asarray(train_c) > 0.0))
        # The recursive NN collocates on its own rollout knots, so equal
        # train_points puts both methods on exactly the same dates.
        nn_data_set = NNDataSettings(
            train_T=data_set.train_T, train_points=data_set.train_points
        )
        nn_knots = np.linspace(0.0, nn_data_set.train_T, nn_data_set.train_points)
        np.testing.assert_allclose(np.asarray(train_t[0]), nn_knots, atol=1e-12)

    def test_draws_keep_z_and_k_separate(self):
        key = jax.random.key(7)
        first = draw_initial_conditions(1.2, 0.9, 12, 0.0025, 0.0025, key, jnp.float64)
        second = draw_initial_conditions(1.2, 0.9, 12, 0.0025, 0.0025, key, jnp.float64)
        other = draw_initial_conditions(
            1.2, 0.9, 12, 0.0025, 0.0025, jax.random.key(8), jnp.float64
        )
        base = np.asarray([1.2, 0.9])
        log_deviation = np.log(np.asarray(first) / base)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(other)))
        self.assertTrue(np.all(np.asarray(first) > 0.0))
        np.testing.assert_allclose(log_deviation.mean(axis=0), 0.0, atol=1e-15)
        np.testing.assert_allclose(
            np.mean(log_deviation**2, axis=0),
            0.0025,
            rtol=1e-13,
        )

    def test_zero_variance_axis_stays_at_base(self):
        drawn = draw_initial_conditions(
            1.2, 0.9, 6, 0.0, 0.0025, jax.random.key(3), jnp.float64
        )
        np.testing.assert_array_equal(np.asarray(drawn)[:, 0], 1.2)
        self.assertGreater(np.std(np.asarray(drawn)[:, 1]), 0.0)

        degenerate = draw_initial_conditions(
            1.2, 0.9, 4, 0.0, 0.0, jax.random.key(3), jnp.float64
        )
        np.testing.assert_array_equal(
            np.asarray(degenerate), np.tile([1.2, 0.9], (4, 1))
        )
        with self.assertRaises(ValueError):
            draw_initial_conditions(
                1.2, 0.9, 1, 0.0025, 0.0025, jax.random.key(3), jnp.float64
            )

    def test_residual_has_forward_and_reverse_mode_ad(self):
        data_set = DataSettings(
            train_T=8.0,
            train_points=9,
            benchmark_T=40.0,
            benchmark_points=80,
        )
        kernel_set = StateKernelSettings(lengthscale=0.25)
        initial_conditions = single_initial_condition()
        _, train_log_states, _ = fixed_collocation_data(
            1 / 3,
            0.1,
            0.11,
            1.0,
            0.02,
            initial_conditions,
            data_set,
            dtype=jnp.float64,
        )
        p = StructuralParameters.create(
            jnp.float64,
            a=1 / 3,
            delta=0.1,
            rho_hat=0.11,
            sigma_crra=1.0,
            g=0.02,
            z_0=jnp.asarray([1.0]),
            k_0=jnp.asarray([1.0]),
        )
        args = build_residual_args(train_log_states, kernel_set)
        alpha = jnp.linspace(-0.01, 0.01, data_set.train_points)
        theta = (alpha, jnp.asarray(-1.2))
        direction = (
            jnp.linspace(0.02, -0.01, data_set.train_points),
            jnp.asarray(0.04),
        )

        def residual(candidate):
            return residual_fn(candidate, args, p, kernel_set)

        _, tangent = jax.jvp(residual, (theta,), (direction,))
        jacobian = jax.jacrev(residual)(theta)
        reverse_over_forward = jacobian[0] @ direction[0] + jacobian[1] * direction[1]
        epsilon = 1e-5
        theta_plus = (
            theta[0] + epsilon * direction[0],
            theta[1] + epsilon * direction[1],
        )
        theta_minus = (
            theta[0] - epsilon * direction[0],
            theta[1] - epsilon * direction[1],
        )
        finite_difference = (residual(theta_plus) - residual(theta_minus)) / (
            2.0 * epsilon
        )

        self.assertTrue(np.isfinite(np.asarray(tangent)).all())
        self.assertGreater(float(jnp.linalg.norm(tangent)), 1e-5)
        np.testing.assert_allclose(
            np.asarray(tangent),
            np.asarray(reverse_over_forward),
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            np.asarray(tangent),
            np.asarray(finite_difference),
            rtol=2e-6,
            atol=2e-8,
        )

    def test_balanced_growth_solve_uses_fixed_centers_and_held_out_paths(self):
        results, extras = neoclassical_growth_kernel_recursive(
            g=0.02,
            verbose=False,
            data_set=DataSettings(test_T=20.0, test_points=21),
            ic_set=InitialConditionSettings(num_test_trajectories=8),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertEqual(results["num_collocation_states"], 65)
        self.assertEqual(results["num_test_trajectories"], 8)
        self.assertAlmostEqual(results["train_point_dt"], 1.5625)
        self.assertTrue(results["transversality_ok"])
        self.assertTrue(results["test_path_ok"])
        self.assertLess(results["max_train_euler"], 2e-5)
        self.assertLess(results["test_median_abs_k_rel_error"], 1e-4)
        self.assertLess(results["test_median_abs_c_rel_error"], 1e-4)
        self.assertGreater(results["shifted_gram_condition"], 1.0)
        self.assertLess(results["shifted_gram_condition"], 1e7)
        self.assertEqual(results["penalized_parameter_count"], 65)
        self.assertEqual(results["unpenalized_parameter_count"], 1)
        self.assertEqual(extras["theta"][0].shape, (65,))
        self.assertEqual(extras["test_df"]["trajectory"].nunique(), 8)
        self.assertEqual(len(extras["test_df"]), 8 * 21)
        self.assertTrue(extras["train_df"]["fixed_exact_center"].all())
        self.assertIn("i_per_z_kernel", extras["policy_df"].columns)
        self.assertIn("i_over_delta_k_kernel", extras["policy_df"].columns)
        self.assertNotIn("k", extras["policy_df"].columns)
        self.assertNotIn("k_ss", results)
        self.assertNotIn("kappa_ss", results)
        self.assertIn("max_test_euler", results)
        self.assertEqual(
            results["parameter_derivative"],
            "conditional_on_fixed_collocation_design",
        )
        for column in ("i_benchmark", "dk_benchmark", "dkappa_benchmark"):
            self.assertIn(column, extras["test_df"].columns)

    def test_free_constant_can_be_disabled(self):
        results, extras = neoclassical_growth_kernel_recursive(
            verbose=False,
            data_set=DataSettings(test_T=10.0, test_points=11),
            ic_set=InitialConditionSettings(num_test_trajectories=2),
            kernel_set=StateKernelSettings(free_constant=False),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertEqual(results["unpenalized_parameter_count"], 0)
        self.assertEqual(results["h_const"], 0.0)
        self.assertEqual(len(extras["theta"]), 1)

    def test_policy_benchmark_covers_both_sides_of_balanced_growth(self):
        a, delta, rho_hat, sigma_crra, g = 1 / 3, 0.1, 0.11, 1.0, 0.02
        grid = np.linspace(1.6, 2.4, 7)
        data_set = DataSettings()
        interpolated = benchmark_policy_grid(
            a, delta, rho_hat, sigma_crra, g, grid, data_set
        )
        direct = []
        times = np.linspace(0.0, data_set.benchmark_T, data_set.benchmark_points)
        for kappa_0 in grid:
            _, _, c = fixed_collocation_data(
                a,
                delta,
                rho_hat,
                sigma_crra,
                g,
                single_initial_condition(1.0, kappa_0),
                DataSettings(
                    train_T=data_set.train_T,
                    train_points=data_set.benchmark_points,
                    benchmark_T=data_set.benchmark_T,
                    benchmark_points=data_set.benchmark_points,
                ),
                dtype=jnp.float64,
            )
            self.assertEqual(len(times), c.shape[1])
            direct.append(float(c[0, 0]))
        np.testing.assert_allclose(interpolated, direct, rtol=3e-4, atol=2e-6)

    def test_multiple_training_initial_conditions_stack_fixed_centers(self):
        results, extras = neoclassical_growth_kernel_recursive(
            g=0.02,
            verbose=False,
            data_set=DataSettings(test_T=10.0, test_points=11),
            ic_set=InitialConditionSettings(
                num_train_trajectories=2,
                num_test_trajectories=4,
                train_log_k_variance=0.0025,
            ),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertEqual(results["num_train_trajectories"], 2)
        self.assertEqual(results["num_collocation_states"], 130)
        self.assertEqual(extras["train_df"]["trajectory"].nunique(), 2)
        self.assertEqual(extras["theta"][0].shape, (130,))

    def test_short_lengthscale_shows_up_in_reported_diagnostics(self):
        results, _ = neoclassical_growth_kernel_recursive(
            g=0.02,
            verbose=False,
            data_set=DataSettings(test_T=10.0, test_points=11),
            ic_set=InitialConditionSettings(num_test_trajectories=4),
            kernel_set=StateKernelSettings(lengthscale=0.15),
            stop_set=StoppingCriteria(max_steps=60),
        )

        # Acceptance is convergence only, so a policy that fits the fixed
        # centers while generating the wrong transition must be visible in the
        # reported diagnostics rather than hidden by a rejection.
        worst_benchmark_error = max(
            results["test_mean_abs_k_rel_error"],
            results["test_mean_abs_c_rel_error"],
        )
        self.assertTrue(
            not results["solver_converged"] or worst_benchmark_error > 5e-3,
            (results["stopping_reason"], worst_benchmark_error),
        )


if __name__ == "__main__":
    unittest.main()
