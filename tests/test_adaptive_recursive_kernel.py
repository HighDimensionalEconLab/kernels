import unittest

import jax
import jax.numpy as jnp
import numpy as np
from tinydiffeq import SaveAt

from kernels import rbf_state_features, rbf_state_gram
from neoclassical_growth_kernel_recursive_adaptive import (
    DataSettings,
    ResidualArgs,
    StateKernelSettings,
    StoppingCriteria,
    StructuralParameters,
    gram_factor,
    neoclassical_growth_kernel_recursive_adaptive,
    ode_components,
    rollout,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class AdaptiveRecursiveKernelTests(unittest.TestCase):
    def test_policy_kernel_gram(self):
        centers = jnp.asarray([-0.4, -0.1, 0.3, 0.8])
        sigma = jnp.asarray(1.2)
        ell = jnp.asarray(0.7)
        gram = rbf_state_gram(centers, sigma, ell)
        value_rows = jax.vmap(lambda q: rbf_state_features(q, centers, sigma, ell))(
            centers
        )
        np.testing.assert_allclose(
            np.asarray(gram),
            np.asarray(value_rows),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(np.asarray(gram), np.asarray(gram.T), atol=1e-12)

    def test_rollout_jvp_includes_moving_states(self):
        kernel_set = StateKernelSettings()
        data_set = DataSettings(train_T=5.0, adaptive_steps=16)
        parameters = StructuralParameters.create(
            jnp.float64,
            a=1 / 3,
            delta=0.1,
            rho_hat=0.11,
            sigma_crra=1.0,
            g=0.0,
            z_0=1.0,
            k_0=1.0,
        )
        capacity = data_set.adaptive_steps + 1
        centers = jnp.zeros((capacity, 2), dtype=jnp.float64)
        active = jnp.arange(capacity) == 0
        _, cholesky = gram_factor(centers, kernel_set, active)
        args = ResidualArgs(
            centers=centers,
            center_times=jnp.zeros(capacity),
            active=active,
            gram_cholesky=cholesky,
        )
        beta = jnp.zeros(capacity)
        direction = jnp.linspace(-1.0, 1.0, beta.size)
        direction /= jnp.linalg.norm(direction)

        def simulated_states(coefficients):
            return rollout(
                coefficients,
                args,
                parameters,
                kernel_set,
                data_set,
                data_set.train_T,
                data_set.adaptive_steps,
                SaveAt(steps=True),
            ).xs

        primal, tangent = jax.jvp(simulated_states, (beta,), (direction,))
        active = np.asarray(
            rollout(
                beta,
                args,
                parameters,
                kernel_set,
                data_set,
                data_set.train_T,
                data_set.adaptive_steps,
                SaveAt(steps=True),
            ).accepted
        )
        self.assertTrue(np.isfinite(np.asarray(primal)[active]).all())
        self.assertTrue(np.isfinite(np.asarray(tangent)[active]).all())
        self.assertGreater(float(jnp.max(jnp.abs(tangent))), 1e-3)

    def test_fixed_solver_is_not_available(self):
        with self.assertRaisesRegex(ValueError, "unknown ode_solver"):
            ode_components(DataSettings(ode_solver="rk4"))

    def test_short_solve(self):
        results, extras = neoclassical_growth_kernel_recursive_adaptive(
            verbose=False,
            measure_compilation=False,
            data_set=DataSettings(
                train_T=5.0,
                adaptive_steps=32,
                test_T=5.0,
                test_points=11,
                test_steps=20,
                train_tvc_T=8.0,
                tvc_points=9,
                tvc_steps=32,
            ),
            stop_set=StoppingCriteria(max_steps=100, train_loss_threshold=1e-9),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertTrue(results["test_path_ok"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)
        self.assertTrue(results["center_fixed_point_ok"])
        self.assertGreater(results["num_collocation_states"], 1)
        self.assertLess(
            results["num_collocation_states"], results["collocation_capacity"]
        )
        self.assertGreater(results["refresh_count"], 0)
        self.assertIn("i_kernel", extras["policy_df"].columns)


if __name__ == "__main__":
    unittest.main()
