import unittest

import jax
import jax.numpy as jnp
import numpy as np
from tinydiffeq import SaveAt

from extra.neoclassical_growth_nn_recursive_adaptive import (
    DataSettings,
    StructuralParameters,
    adaptive_collocation,
    adaptive_residuals,
    adaptive_rollout,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class AdaptiveRecursiveNNTests(unittest.TestCase):
    def test_training_collocation_keeps_a_leading_vmap_axis(self):
        parameters = StructuralParameters.create(
            jnp.float32,
            a=1 / 3,
            delta=0.1,
            rho_hat=0.11,
            k_0=1.0,
            A=1.0,
            b_1=1.0,
            b_2=0.0,
            k_thr=jnp.inf,
        )
        data_set = DataSettings(
            train_T=5.0,
            ode_solver="tsit5",
            adaptive_steps=16,
            dt=0.25,
        )
        states, times, active, ok = adaptive_collocation(
            lambda k: jnp.asarray(0.12, k.dtype) * k,
            parameters,
            data_set,
        )

        self.assertEqual(states.shape, (1, 17))
        self.assertEqual(times.shape, (1, 17))
        self.assertEqual(active.shape, (1, 17))
        self.assertEqual(ok.shape, (1,))
        self.assertEqual(states.dtype, jnp.float32)
        self.assertEqual(times.dtype, jnp.float32)
        self.assertTrue(bool(ok[0]))

        def residual(scale):
            return adaptive_residuals(lambda k: scale * k, parameters, data_set)

        scale = jnp.asarray(0.12, jnp.float32)
        jaxpr = str(jax.make_jaxpr(residual)(scale))
        values, pullback = jax.vjp(residual, scale)
        cotangent = pullback(jnp.ones_like(values))[0]
        self.assertNotIn("f64", jaxpr)
        self.assertEqual(values.dtype, jnp.float32)
        self.assertEqual(cotangent.dtype, jnp.float32)
        self.assertTrue(bool(jnp.isfinite(cotangent)))

    def test_accepted_states_move_with_policy_parameters(self):
        parameters = StructuralParameters.create(
            jnp.float64,
            a=1 / 3,
            delta=0.1,
            rho_hat=0.11,
            k_0=1.0,
            A=1.0,
            b_1=1.0,
            b_2=0.0,
            k_thr=jnp.inf,
        )

        for ode_solver in ("tsit5", "rodas5p"):
            with self.subTest(ode_solver=ode_solver):
                data_set = DataSettings(
                    train_T=5.0,
                    ode_solver=ode_solver,
                    adaptive_steps=16,
                    dt=0.25,
                    step_growth_max=3.0,
                )

                def solution(scale):
                    return adaptive_rollout(
                        lambda k: scale * k,
                        data_set.train_T,
                        data_set.adaptive_steps,
                        parameters,
                        data_set,
                        SaveAt(steps=True),
                    )

                scale = jnp.asarray(0.12, dtype=jnp.float64)
                primal = solution(scale)
                _, tangent = jax.jvp(
                    lambda value: solution(value).xs,
                    (scale,),
                    (jnp.ones(()),),
                )
                active = np.asarray(primal.accepted)

                self.assertTrue(bool(primal.ok))
                self.assertGreater(int(active.sum()), 2)
                self.assertLess(int(active.sum()), data_set.adaptive_steps + 1)
                np.testing.assert_allclose(
                    np.asarray(tangent)[active],
                    np.asarray(primal.ts)[active],
                    rtol=1e-11,
                    atol=1e-11,
                )


if __name__ == "__main__":
    unittest.main()
