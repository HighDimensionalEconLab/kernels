import math
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import open_economy_growth_nn_recursive as recursive
from open_economy_growth import ModelParameters

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class ThresholdCaptured(Exception):
    pass


class OpenEconomyRecursiveDtypeTests(unittest.TestCase):
    def test_float32_exact_collocation_and_vjp_are_dtype_pure(self):
        dtype = jnp.float32
        data_set = recursive.DataSettings(
            train_T=2.0,
            train_points=3,
            rollout_steps=4,
            test_T=2.0,
            test_points=3,
            diagnostic_T=2.0,
            diagnostic_points=3,
            diagnostic_steps=2,
            benchmark_T=2.0,
            benchmark_points=3,
            policy_grid_size=3,
        )
        parameters = recursive.DynamicModelParameters.from_model(
            ModelParameters(), dtype
        )
        model = recursive.SequentialPolicyNN(
            recursive.SequentialPolicySettings(width=4, depth=1),
            dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        graphdef, theta, fixed_state = nnx.split(model, nnx.Param, ...)
        train_t = recursive.make_train_t_grid(data_set, dtype)
        args = recursive.ResidualArgs(
            graphdef=graphdef,
            fixed_state=fixed_state,
            train_t=train_t,
            data_set=data_set,
        )
        p = recursive.ResidualParameters(
            k_0=jnp.asarray(1.0, dtype=dtype),
            a_0=jnp.asarray(0.0, dtype=dtype),
            model=parameters,
        )
        policy = nnx.merge(graphdef, theta, fixed_state)
        initial_k = jnp.reshape(p.k_0, (1,))
        solution = recursive.collocation_rollout(
            policy,
            train_t,
            initial_k,
            parameters,
            data_set,
        )

        self.assertEqual(solution.ts.shape, (1, 3))
        self.assertEqual(solution.xs.shape, (1, 3, 2))
        self.assertEqual(solution.ok.shape, (1,))
        self.assertTrue(bool(solution.ok[0]))
        self.assertEqual(int(solution.num_steps[0]), data_set.rollout_steps)
        self.assertEqual(int(solution.num_accepted[0]), data_set.rollout_steps)
        np.testing.assert_array_equal(solution.ts[0], train_t)

        residual = recursive.residual_fn(theta, args, p)
        _, pullback = jax.vjp(
            lambda value: recursive.residual_fn(value, args, p), theta
        )
        gradient = pullback(jnp.ones_like(residual))[0]
        self.assertEqual(residual.shape, (2 * data_set.train_points,))
        self.assertEqual(residual.dtype, dtype)
        self.assertEqual(
            {leaf.dtype for leaf in jax.tree.leaves(gradient)},
            {jnp.dtype(dtype)},
        )

        residual_jaxpr = str(
            jax.make_jaxpr(lambda value: recursive.residual_fn(value, args, p))(theta)
        )
        vjp_jaxpr = str(
            jax.make_jaxpr(
                lambda value, cotangent: jax.vjp(
                    lambda candidate: recursive.residual_fn(candidate, args, p),
                    value,
                )[1](cotangent)[0]
            )(theta, jnp.ones_like(residual))
        )
        rollout_jaxpr = str(
            jax.make_jaxpr(
                lambda value, times, capital: recursive.collocation_rollout(
                    value,
                    times,
                    capital,
                    parameters,
                    data_set,
                )
            )(policy, train_t, initial_k)
        )
        self.assertNotIn("f64", residual_jaxpr)
        self.assertNotIn("f64", vjp_jaxpr)
        self.assertNotIn("f64", rollout_jaxpr)
        self.assertNotIn("searchsorted", rollout_jaxpr)

    def test_float32_full_driver_keeps_benchmark_gate_on_device(self):
        results, _ = recursive.open_economy_growth_nn_recursive(
            use_float64=False,
            data_set=recursive.DataSettings(
                diagnostic_T=50.0,
                diagnostic_points=51,
                diagnostic_steps=50,
            ),
            verbose=False,
        )

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertTrue(results["benchmark_match"])
        self.assertEqual(results["residual_dtype"], "float32")
        self.assertEqual(results["benchmark_error_dtype"], "float32")
        self.assertEqual(results["benchmark_error_device"], jax.default_backend())

    def test_default_train_loss_threshold_depends_on_residual_dtype(self):
        captured_atol = []

        class SolverProbe:
            def __init__(self, *args, **kwargs):
                pass

            def solve(self, *args, **kwargs):
                captured_atol.append(kwargs["atol"])
                raise ThresholdCaptured

        for use_float64, expected_loss in ((True, 1e-14), (False, 1e-12)):
            with self.subTest(use_float64=use_float64):
                captured_atol.clear()
                with mock.patch.object(recursive, "LevenbergMarquardt", SolverProbe):
                    with self.assertRaises(ThresholdCaptured):
                        recursive.open_economy_growth_nn_recursive(
                            use_float64=use_float64,
                            stop_set=recursive.StoppingCriteria(),
                            verbose=False,
                        )
                self.assertEqual(len(captured_atol), 1)
                self.assertEqual(captured_atol[0], math.sqrt(expected_loss))

    def test_adaptive_dt_min_and_train_loss_threshold_validation(self):
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(adaptive_dt_min=value):
                with self.assertRaisesRegex(
                    ValueError, "adaptive_dt_min must be positive or None"
                ):
                    recursive.open_economy_growth_nn_recursive(
                        use_float64=False,
                        data_set=recursive.DataSettings(adaptive_dt_min=value),
                        verbose=False,
                    )

        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(train_loss_threshold=value):
                with self.assertRaisesRegex(
                    ValueError, "train_loss_threshold must be positive"
                ):
                    recursive.open_economy_growth_nn_recursive(
                        use_float64=False,
                        stop_set=recursive.StoppingCriteria(train_loss_threshold=value),
                        verbose=False,
                    )


if __name__ == "__main__":
    unittest.main()
