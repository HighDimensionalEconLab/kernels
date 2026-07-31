import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from neoclassical_growth_nn_recursive import (
    DataSettings,
    StructuralParameters,
    collocation_residuals,
    collocation_rollout,
    euler_residuals,
    production,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


@struct.dataclass
class SharePolicy:
    logit: jax.Array
    slope: jax.Array

    def __call__(self, log_state, p):
        share = jax.nn.sigmoid(self.logit + self.slope * (log_state[1] - log_state[0]))
        z, k = jnp.exp(log_state)
        return production(z, k, p) * share


def parameters(*, z_0=(1.0,), k_0=(1.0,), g=0.02, dtype=jnp.float64):
    return StructuralParameters.create(
        dtype,
        a=1 / 3,
        delta=0.1,
        rho_hat=0.11,
        sigma_crra=1.0,
        g=g,
        z_0=jnp.asarray(z_0),
        k_0=jnp.asarray(k_0),
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        kappa_thr=jnp.inf,
    )


class RecursiveGrowthCollocationTests(unittest.TestCase):
    def test_float32_vmap_is_pure_for_one_and_three_trajectories(self):
        data_set = DataSettings(train_T=8.0, train_points=9)
        expected_states = data_set.train_points
        expected_steps = expected_states - 1
        for initial_conditions in (
            ((1.0,), (1.0,)),
            ((1.0, 1.02, 0.98), (1.0, 0.98, 1.02)),
        ):
            p = parameters(
                z_0=initial_conditions[0],
                k_0=initial_conditions[1],
                dtype=jnp.float32,
            )

            def residual_from_logit(logit):
                policy = SharePolicy(logit, jnp.asarray(0.15, jnp.float32))
                return collocation_residuals(policy, p, data_set).reshape(-1)

            logit = jnp.asarray(-1.0, jnp.float32)
            jaxpr = str(jax.make_jaxpr(residual_from_logit)(logit))
            residual, pullback = jax.vjp(residual_from_logit, logit)
            cotangent = pullback(jnp.ones_like(residual))[0]
            policy = SharePolicy(logit, jnp.asarray(0.15, jnp.float32))
            solution = collocation_rollout(policy, p, data_set)
            expected_times = jnp.linspace(
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(data_set.train_T, dtype=jnp.float32),
                expected_states,
                dtype=jnp.float32,
            )

            self.assertNotIn("f64", jaxpr)
            # Collocation is the integrator's own states, so no interpolation
            # search can appear in the differentiated residual.
            self.assertNotIn("searchsorted", jaxpr)
            self.assertEqual(residual.dtype, jnp.float32)
            self.assertEqual(cotangent.dtype, jnp.float32)
            self.assertTrue(np.all(np.isfinite(np.asarray(residual))))
            self.assertTrue(bool(jnp.isfinite(cotangent)))
            self.assertEqual(
                solution.ts.shape,
                (len(initial_conditions[0]), expected_states),
            )
            self.assertEqual(
                solution.xs.shape,
                (len(initial_conditions[0]), expected_states, 2),
            )
            self.assertTrue(bool(jnp.all(solution.ok)))
            self.assertTrue(bool(jnp.all(solution.accepted)))
            self.assertTrue(bool(jnp.all(solution.num_steps == expected_steps)))
            np.testing.assert_array_equal(
                np.asarray(solution.ts),
                np.broadcast_to(
                    np.asarray(expected_times),
                    (len(initial_conditions[0]), expected_states),
                ),
            )

    def test_residual_is_the_euler_equation_at_every_rollout_state(self):
        data_set = DataSettings(train_T=8.0, train_points=9)
        p = parameters(z_0=(1.0, 1.05), k_0=(1.0, 0.95))
        policy = SharePolicy(jnp.asarray(-1.0), jnp.asarray(0.15))

        solution = collocation_rollout(policy, p, data_set)
        residual = collocation_residuals(policy, p, data_set)
        expected = euler_residuals(policy, solution.xs.reshape(-1, 2), p).reshape(
            residual.shape
        )

        self.assertEqual(residual.shape, (2, data_set.train_points))
        # Normalized by every residual entry, not by the states of one
        # trajectory, so the loss threshold is trajectory-count invariant.
        np.testing.assert_allclose(
            np.asarray(residual),
            np.asarray(expected) / np.sqrt(2 * data_set.train_points),
            rtol=1e-15,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
