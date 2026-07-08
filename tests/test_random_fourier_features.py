import unittest

import jax.numpy as jnp
from flax import nnx

from random_fourier_features import (
    RandomFourierFeatures,
    RFFFeatures,
    RFFSettings,
    exact_kernel_matrix,
    normalize_kernel_name,
)


class RandomFourierFeatureTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_kernel_name("gaussian"), "rbf")
        self.assertEqual(normalize_kernel_name("matern_0.5"), "matern12")
        self.assertEqual(normalize_kernel_name(1.5), "matern32")

    def test_feature_and_model_shapes(self):
        x = jnp.ones((7, 3), dtype=jnp.float64)
        features = RFFFeatures(
            input_dim=3,
            num_features=32,
            kernel="rbf",
            rngs=nnx.Rngs(0),
        )
        model = RandomFourierFeatures(
            RFFSettings(
                input_dim=3,
                num_features=32,
                output_dim=5,
                kernel="matern32",
            ),
            rngs=nnx.Rngs(1),
        )

        self.assertEqual(features(x).shape, (7, 32))
        self.assertEqual(features.gram(x, x).shape, (7, 7))
        self.assertEqual(model(x).shape, (7, 5))

    def test_deterministic_initialization(self):
        lhs = RFFFeatures(input_dim=2, num_features=64, rngs=nnx.Rngs(123))
        rhs = RFFFeatures(input_dim=2, num_features=64, rngs=nnx.Rngs(123))

        self.assertTrue(jnp.array_equal(lhs.frequencies[...], rhs.frequencies[...]))
        self.assertTrue(jnp.array_equal(lhs.phases[...], rhs.phases[...]))

    def test_only_head_is_trainable(self):
        model = RandomFourierFeatures(
            RFFSettings(input_dim=2, num_features=16, output_dim=3),
            rngs=nnx.Rngs(0),
        )
        param_state = nnx.state(model, nnx.Param)

        self.assertIn("head", param_state)
        self.assertNotIn("features", param_state)
        self.assertEqual(param_state["head"]["kernel"][...].shape, (16, 3))

    def test_kernel_approximations(self):
        x = jnp.array(
            [
                [-1.0, -0.5],
                [-0.4, 0.25],
                [0.0, 0.0],
                [0.5, -0.25],
                [1.0, 0.75],
            ],
            dtype=jnp.float64,
        )
        for kernel in ["rbf", "matern12", "matern32"]:
            with self.subTest(kernel=kernel):
                features = RFFFeatures(
                    input_dim=2,
                    num_features=8192,
                    kernel=kernel,
                    radius=1.3,
                    sigma=1.2,
                    rngs=nnx.Rngs(123),
                )
                approximate = features.gram(x, x)
                exact = exact_kernel_matrix(
                    x,
                    x,
                    kernel=kernel,
                    radius=1.3,
                    sigma=1.2,
                )
                max_abs_error = float(jnp.max(jnp.abs(approximate - exact)))

                self.assertLess(max_abs_error, 5e-2)


if __name__ == "__main__":
    unittest.main()
