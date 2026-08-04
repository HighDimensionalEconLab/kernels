import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from kernels import (
    additive_rbf_state_features,
    additive_rbf_state_gram,
    ard_rbf_state_features,
    ard_rbf_state_gram,
    rbf_state_features,
)
from neoclassical_growth_kernel_recursive import (
    DataSettings,
    InitialConditionSettings,
    StateKernelSettings,
    build_residual_args,
    investment,
    neoclassical_growth_kernel_recursive,
    validate_settings,
)
from neoclassical_growth_nn_recursive import (
    DataSettings as NNDataSettings,
)
from neoclassical_growth_nn_recursive import (
    InvestmentPolicyNN,
    InvestmentPolicySettings,
    StructuralParameters,
    neoclassical_growth_nn_recursive,
)
from neoclassical_growth_nn_recursive import (
    StoppingCriteria as NNStoppingCriteria,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


CENTERS_2D = jnp.asarray(
    [[-0.4, 0.2], [-0.1, -0.3], [0.3, 0.7], [0.8, -0.6], [0.0, 0.0]]
)
POINTS_2D = jnp.asarray([[0.15, -0.25], [-0.6, 0.4], [0.0, 0.9]])
LOG_STATES = jnp.asarray(
    [[0.0, -0.3], [0.0, 0.0], [0.2, 0.6], [-0.4, 0.5], [0.1, -0.2]]
)


def structural(delta=0.1, a=1 / 3, rho_hat=0.11, sigma_crra=1.0, g=0.02):
    return StructuralParameters.create(
        jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        g=g,
        z_0=jnp.asarray([1.0]),
        k_0=jnp.asarray([1.0]),
    )


class VectorStateKernelTests(unittest.TestCase):
    def test_ard_features_are_products_of_per_dimension_rbfs(self):
        sigma = jnp.asarray(1.2)
        lengthscales = jnp.asarray([0.7, 0.35])
        features = ard_rbf_state_features(POINTS_2D, CENTERS_2D, sigma, lengthscales)
        per_dimension = [
            rbf_state_features(
                POINTS_2D[:, d], CENTERS_2D[:, d], jnp.asarray(1.0), lengthscales[d]
            )
            for d in range(2)
        ]
        np.testing.assert_allclose(
            np.asarray(features),
            np.asarray(sigma**2 * per_dimension[0] * per_dimension[1]),
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertEqual(features.shape, (3, 5))

    def test_additive_features_are_sums_of_per_dimension_rbfs(self):
        sigmas = jnp.asarray([1.2, 0.8])
        lengthscales = jnp.asarray([0.7, 0.35])
        features = additive_rbf_state_features(
            POINTS_2D, CENTERS_2D, sigmas, lengthscales
        )
        per_dimension = sum(
            rbf_state_features(
                POINTS_2D[:, d], CENTERS_2D[:, d], sigmas[d], lengthscales[d]
            )
            for d in range(2)
        )
        np.testing.assert_allclose(
            np.asarray(features), np.asarray(per_dimension), rtol=1e-13, atol=1e-13
        )

    def test_grams_match_features_and_are_symmetric(self):
        sigma = jnp.asarray(1.2)
        sigmas = jnp.asarray([1.2, 0.8])
        lengthscales = jnp.asarray([0.7, 0.35])
        for gram, feature_rows in (
            (
                ard_rbf_state_gram(CENTERS_2D, sigma, lengthscales),
                jax.vmap(
                    lambda x: ard_rbf_state_features(x, CENTERS_2D, sigma, lengthscales)
                )(CENTERS_2D),
            ),
            (
                additive_rbf_state_gram(CENTERS_2D, sigmas, lengthscales),
                jax.vmap(
                    lambda x: additive_rbf_state_features(
                        x, CENTERS_2D, sigmas, lengthscales
                    )
                )(CENTERS_2D),
            ),
        ):
            np.testing.assert_allclose(
                np.asarray(gram), np.asarray(feature_rows), rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(np.asarray(gram), np.asarray(gram.T), atol=1e-12)
            eigenvalues = np.linalg.eigvalsh(np.asarray(gram))
            self.assertGreater(eigenvalues.min(), -1e-10)


class NonHomotheticPolicyMapTests(unittest.TestCase):
    def test_zero_parameters_invest_delta_k0_everywhere(self):
        p = structural()
        nn = InvestmentPolicyNN(
            InvestmentPolicySettings(
                head_init_scale=0.0, homothetic_representation=False
            ),
            z_scale=1.3,
            k_scale=0.8,
            dtype=jnp.float64,
            rngs=nnx.Rngs(0),
        )
        for state_kernel in ("ard_rbf", "additive_rbf"):
            kernel_set = StateKernelSettings(
                homothetic_representation=False, state_kernel=state_kernel
            )
            args = build_residual_args(
                LOG_STATES.reshape(1, 5, 2), kernel_set, 1.3, 0.8
            )
            theta = (jnp.zeros(5), jnp.zeros(()))
            for log_state in LOG_STATES:
                self.assertEqual(
                    float(investment(theta, args, log_state, p, kernel_set)),
                    float(p.delta) * 0.8,
                )
        for log_state in LOG_STATES:
            self.assertEqual(float(nn(log_state, p)), float(p.delta) * 0.8)

    def test_nonhomothetic_policies_are_not_homogeneous(self):
        p = structural()
        nn = InvestmentPolicyNN(
            InvestmentPolicySettings(homothetic_representation=False),
            dtype=jnp.float64,
            rngs=nnx.Rngs(3),
        )
        kernel_set = StateKernelSettings(homothetic_representation=False)
        args = build_residual_args(LOG_STATES.reshape(1, 5, 2), kernel_set)
        theta = (0.4 * jnp.linspace(-1.0, 1.0, 5), jnp.asarray(-0.2))
        scale = jnp.log(2.5)
        for log_state in LOG_STATES:
            scaled_state = log_state + scale
            for i_scaled, i_base in (
                (float(nn(scaled_state, p)), float(nn(log_state, p))),
                (
                    float(investment(theta, args, scaled_state, p, kernel_set)),
                    float(investment(theta, args, log_state, p, kernel_set)),
                ),
            ):
                self.assertNotAlmostEqual(i_scaled, 2.5 * i_base, places=2)

    def test_linear_mean_enters_the_raw_policy_and_needs_the_toggle(self):
        kernel_set = StateKernelSettings(
            homothetic_representation=False, linear_mean=True
        )
        args = build_residual_args(LOG_STATES.reshape(1, 5, 2), kernel_set, 1.0, 2.0)
        p = structural()
        theta = (jnp.zeros(5), jnp.zeros(()), jnp.asarray([0.3, -0.7]))
        log_state = LOG_STATES[2]
        x = np.asarray(log_state) - np.log([1.0, 2.0])
        expected_raw = float(x @ np.asarray([0.3, -0.7]))
        i = float(investment(theta, args, log_state, p, kernel_set))
        # investment = delta * k_0 * U(x @ mean_coefficients) with everything
        # else zero, so the linear mean is visible through a nonunit U.
        self.assertNotAlmostEqual(i, float(p.delta) * 2.0, places=6)
        self.assertGreater(abs(expected_raw), 0.1)
        with self.assertRaisesRegex(ValueError, "linear_mean"):
            validate_settings(
                DataSettings(),
                StateKernelSettings(linear_mean=True),
                1e-12,
            )

    def test_homothetic_default_keeps_prior_gram_and_centers(self):
        kernel_set = StateKernelSettings()
        args = build_residual_args(LOG_STATES.reshape(1, 5, 2), kernel_set)
        self.assertEqual(args.centers.shape, (5,))
        np.testing.assert_array_equal(
            np.asarray(args.centers),
            np.asarray(LOG_STATES[:, 1] - LOG_STATES[:, 0]),
        )


class NonHomotheticSolveTests(unittest.TestCase):
    def test_kernel_solve_records_the_toggle_and_solves(self):
        results, extras = neoclassical_growth_kernel_recursive(
            g=0.02,
            verbose=False,
            data_set=DataSettings(test_T=10.0, test_points=11),
            ic_set=InitialConditionSettings(num_test_trajectories=2),
            kernel_set=StateKernelSettings(
                homothetic_representation=False, linear_mean=True
            ),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertFalse(results["homothetic_representation"])
        self.assertEqual(results["policy"], "delta_k0_unit_softplus")
        self.assertEqual(results["state_coordinate"], "(log z, log k)")
        self.assertEqual(results["kernel"], "ard_rbf")
        self.assertEqual(results["state_kernel"], "ard_rbf")
        self.assertEqual(results["log_z_lengthscale"], 0.4)
        self.assertEqual(results["log_k_lengthscale"], 0.4)
        self.assertEqual(results["unpenalized_parameter_count"], 3)
        self.assertEqual(len(extras["theta"]), 3)
        self.assertEqual(extras["theta"][0].shape, (65,))
        self.assertEqual(extras["theta"][2].shape, (2,))
        self.assertEqual(extras["residual_args"].centers.shape, (65, 2))
        self.assertLess(results["max_train_euler"], 1e-4)

    def test_nn_solve_records_the_toggle(self):
        results, _ = neoclassical_growth_nn_recursive(
            g=0.02,
            verbose=False,
            data_set=NNDataSettings(
                train_T=20.0, train_tvc_T=50.0, test_T=10.0, test_points=11
            ),
            ic_set=InitialConditionSettings(num_test_trajectories=2),
            hc_set=InvestmentPolicySettings(homothetic_representation=False),
            stop_set=NNStoppingCriteria(max_steps=2, num_attempts=1),
        )

        self.assertFalse(results["homothetic_representation"])
        self.assertEqual(results["policy"], "delta_k0_unit_softplus")
        self.assertEqual(results["state_coordinate"], "(log z, log k)")
        self.assertTrue(np.isfinite(results["train_loss"]))


if __name__ == "__main__":
    unittest.main()
