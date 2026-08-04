import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from neoclassical_growth_kernel_recursive import (
    DataSettings,
    InitialConditionSettings,
    StateKernelSettings,
    build_residual_args,
    investment,
    neoclassical_growth_kernel_recursive,
)
from neoclassical_growth_kernel_recursive import (
    residual_fn as kernel_residual_fn,
)
from neoclassical_growth_nn_recursive import (
    DataSettings as NNDataSettings,
)
from neoclassical_growth_nn_recursive import (
    InvestmentPolicyNN,
    InvestmentPolicySettings,
    StructuralParameters,
    log_state_drift,
    neoclassical_growth_nn_recursive,
)
from neoclassical_growth_nn_recursive import (
    StoppingCriteria as NNStoppingCriteria,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


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


def kernel_theta_and_args(alpha_scale=0.0, h_const=0.0):
    log_states = jnp.stack([jnp.zeros(5), jnp.linspace(-0.4, 0.9, 5)], axis=1).reshape(
        1, 5, 2
    )
    args = build_residual_args(log_states, StateKernelSettings())
    alpha = alpha_scale * jnp.linspace(-1.0, 1.0, 5)
    theta = (alpha, jnp.asarray(h_const))
    return theta, args


LOG_STATES = jnp.asarray(
    [[0.0, -0.3], [0.0, 0.0], [0.2, 0.6], [-0.4, 0.5], [0.1, -0.2]]
)


class SharedOutputMapTests(unittest.TestCase):
    def test_zero_parameters_hold_capital_at_rest(self):
        p = structural()
        nn = InvestmentPolicyNN(
            InvestmentPolicySettings(head_init_scale=0.0),
            dtype=jnp.float64,
            rngs=nnx.Rngs(0),
        )
        theta, args = kernel_theta_and_args()
        kernel_set = StateKernelSettings()
        for log_state in LOG_STATES:
            k = float(jnp.exp(log_state[1]))
            for i in (
                float(nn(log_state, p)),
                float(investment(theta, args, log_state, p, kernel_set)),
            ):
                self.assertAlmostEqual(i, float(p.delta) * k, places=13)
            drift = log_state_drift(nn, log_state, p)
            self.assertAlmostEqual(float(drift[1]), 0.0, places=13)

    def test_output_map_is_homogeneous_of_degree_one(self):
        p = structural()
        nn = InvestmentPolicyNN(
            InvestmentPolicySettings(), dtype=jnp.float64, rngs=nnx.Rngs(3)
        )
        theta, args = kernel_theta_and_args(alpha_scale=0.4, h_const=-0.2)
        kernel_set = StateKernelSettings()
        scale = jnp.log(2.5)
        for log_state in LOG_STATES:
            scaled_state = log_state + scale
            self.assertAlmostEqual(
                float(nn(scaled_state, p)), 2.5 * float(nn(log_state, p)), places=12
            )
            self.assertAlmostEqual(
                float(investment(theta, args, scaled_state, p, kernel_set)),
                2.5 * float(investment(theta, args, log_state, p, kernel_set)),
                places=12,
            )

    def test_output_map_uses_only_delta_and_the_state(self):
        nn = InvestmentPolicyNN(
            InvestmentPolicySettings(), dtype=jnp.float64, rngs=nnx.Rngs(3)
        )
        base = structural()
        perturbed = structural(a=0.5, rho_hat=0.2, sigma_crra=2.0, g=0.0)
        double_delta = structural(delta=0.2)
        for log_state in LOG_STATES:
            base_i = float(nn(log_state, base))
            self.assertEqual(base_i, float(nn(log_state, perturbed)))
            self.assertAlmostEqual(
                2.0 * base_i, float(nn(log_state, double_delta)), places=13
            )

    def test_infeasible_consumption_is_a_loud_nan(self):
        # unit_softplus(10) makes i vastly exceed output, so c < 0. The
        # residual must surface NaN rather than clamp to a feasible value.
        p = structural()
        theta, args = kernel_theta_and_args(h_const=10.0)
        residual = kernel_residual_fn(theta, args, p, StateKernelSettings())
        self.assertTrue(bool(jnp.isnan(residual).any()))

    def test_nn_and_kernel_share_initial_conditions(self):
        ic_set = InitialConditionSettings(
            num_train_trajectories=2,
            num_test_trajectories=3,
            train_log_k_variance=0.0025,
            data_seed=11,
        )
        _, nn_extras = neoclassical_growth_nn_recursive(
            g=0.02,
            verbose=False,
            data_set=NNDataSettings(
                train_T=20.0, train_tvc_T=50.0, test_T=10.0, test_points=11
            ),
            ic_set=ic_set,
            stop_set=NNStoppingCriteria(max_steps=2),
        )
        _, kernel_extras = neoclassical_growth_kernel_recursive(
            g=0.02,
            verbose=False,
            data_set=DataSettings(
                train_T=20.0, train_tvc_T=50.0, test_T=10.0, test_points=11
            ),
            ic_set=ic_set,
        )

        np.testing.assert_array_equal(
            np.asarray(nn_extras["train_initial_conditions"]),
            np.asarray(kernel_extras["train_initial_conditions"]),
        )
        np.testing.assert_array_equal(
            np.asarray(nn_extras["test_initial_conditions"]),
            np.asarray(kernel_extras["test_initial_conditions"]),
        )

    def test_test_center_recenters_only_held_out_draws(self):
        results, extras = neoclassical_growth_kernel_recursive(
            verbose=False,
            data_set=DataSettings(test_T=10.0, test_points=11),
            ic_set=InitialConditionSettings(
                num_test_trajectories=4,
                test_log_z_variance=0.0,
                test_log_k_variance=0.0,
                test_k_0=1.5,
            ),
        )

        self.assertEqual(results["test_k_0"], 1.5)
        self.assertEqual(results["k_0"], 1.0)
        train = np.asarray(extras["train_initial_conditions"])
        test = np.asarray(extras["test_initial_conditions"])
        np.testing.assert_array_equal(train, [[1.0, 1.0]])
        np.testing.assert_array_equal(test, np.tile([1.0, 1.5], (4, 1)))


if __name__ == "__main__":
    unittest.main()
