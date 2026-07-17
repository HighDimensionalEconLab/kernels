import unittest

import jax
import jax.numpy as jnp
import numpy as np
from nlls_gram import (
    Metric,
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
)

from kernel_metrics import metric_with_compute_dtype, repeated_blockdiag_metric

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class RepeatedBlockdiagMetricTests(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.repeats = 3
        self.additional_size = 2
        block_matrix = jnp.array(
            [
                [2.0, 0.2, 0.1, 0.0],
                [0.2, 1.8, 0.0, 0.1],
                [0.1, 0.0, 1.5, 0.2],
                [0.0, 0.1, 0.2, 1.2],
            ]
        )
        self.block_metric = metric_from_cholesky(jnp.linalg.cholesky(block_matrix))
        self.additional_metric = metric_from_diagonal(jnp.array([0.3, 0.7]))
        self.repeated = repeated_blockdiag_metric(
            self.block_metric,
            self.block_size,
            self.repeats,
            additional=(self.additional_metric, self.additional_size),
        )
        self.reference = blockdiag_metric(
            [(self.block_metric, self.block_size)] * self.repeats
            + [(self.additional_metric, self.additional_size)]
        )

    def test_callbacks_match_general_blockdiag_metric(self):
        total_size = self.block_size * self.repeats + self.additional_size
        vector = jax.random.normal(jax.random.key(0), (total_size,))
        matrix = jax.random.normal(jax.random.key(1), (total_size, 5))

        for name in ("solve", "inv_sqrt", "inv_sqrt_transpose"):
            with self.subTest(callback=name):
                actual = jax.jit(getattr(self.repeated, name))(matrix)
                expected = jax.jit(getattr(self.reference, name))(matrix)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            self.repeated.norm(vector),
            self.reference.norm(vector),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_repeated_callback_receives_one_combined_rhs(self):
        received_shapes = []

        def solve(x):
            received_shapes.append(x.shape)
            return x

        metric = repeated_blockdiag_metric(
            Metric(solve=solve),
            self.block_size,
            self.repeats,
        )
        rhs_columns = 5
        metric.solve(jnp.ones((self.block_size * self.repeats, rhs_columns)))

        self.assertEqual(
            received_shapes,
            [(self.block_size, self.repeats * rhs_columns)],
        )

    def test_compute_dtype_preserves_caller_dtype(self):
        factor = jnp.linalg.cholesky(
            jnp.array([[2.0, 0.2], [0.2, 1.5]], dtype=jnp.float64)
        )
        metric = metric_with_compute_dtype(metric_from_cholesky(factor), jnp.float64)
        value = jnp.array([1.0, -0.5], dtype=jnp.float32)

        solved = metric.solve(value)

        self.assertEqual(solved.dtype, jnp.float32)
        np.testing.assert_allclose(
            solved,
            metric_from_cholesky(factor).solve(value.astype(jnp.float64)),
            rtol=1e-6,
        )

    def test_partial_callback_and_shape_contracts(self):
        metric = repeated_blockdiag_metric(
            Metric(solve=lambda x: x),
            self.block_size,
            self.repeats,
        )
        self.assertIsNotNone(metric.solve)
        self.assertIsNone(metric.norm)
        self.assertIsNone(metric.inv_sqrt)
        self.assertIsNone(metric.inv_sqrt_transpose)

        with self.assertRaisesRegex(ValueError, "leading size"):
            self.repeated.solve(jnp.ones(self.block_size))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            repeated_blockdiag_metric(self.block_metric, self.block_size, 0)


if __name__ == "__main__":
    unittest.main()
