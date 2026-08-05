import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import stochastic_open_economy_growth_kernel_recursive as kernel
import stochastic_open_economy_growth_nn_recursive as nn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# Every problem-half input (and ic_set) must be identical for both solvers;
# a mismatch is a loud FileNotFoundError.
DATA_SET = dict(
    train_points=17,
    test_points=9,
    train_T=20.0,
    test_T=20.0,
)
NN_DATA_SET = nn.DataSettings(**DATA_SET, tvc_points=33, tvc_T=40.0)
KERNEL_DATA_SET = nn.DataSettings(**DATA_SET)


class StochasticOpenEconomyKernelRecursiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.nn_results, cls.nn_extras = nn.stochastic_open_economy_growth_nn_recursive(
            data_set=NN_DATA_SET,
            nn_set=nn.SequentialPolicySettings(width=8, depth=2),
            stop_set=nn.StoppingCriteria(
                max_steps=30,
                num_epochs=2,
                num_attempts=2,
                train_loss_threshold=1e-6,
            ),
            policy_cache=cls.tmp.name,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_missing_reference_fails_loudly(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as raised:
                kernel.stochastic_open_economy_growth_kernel_recursive(
                    policy_dir=empty,
                    data_set=KERNEL_DATA_SET,
                    verbose=False,
                )
        self.assertIn("--policy_cache", str(raised.exception))

    def test_frozen_design_solve_matches_reference(self):
        self.assertTrue(self.nn_results["solution_accepted"])
        results, extras = kernel.stochastic_open_economy_growth_kernel_recursive(
            policy_dir=self.tmp.name,
            data_set=KERNEL_DATA_SET,
            # The driver default asks for eight training trajectories; the
            # cached NN reference above ran at the NN default, and the
            # problem hash must match.
            ic_set=nn.InitialConditionSettings(),
            tolerance_set=nn.ToleranceSettings(
                test_euler_mean_abs=5e-2,
                test_euler_max=5e-1,
            ),
            verbose=False,
        )

        self.assertTrue(results["solver_converged"])
        self.assertEqual(results["collocation_design"], "frozen_reference_nn_rollout")
        self.assertEqual(
            results["parameter_derivative"],
            "conditional_on_fixed_collocation_design",
        )
        self.assertEqual(
            results["variant"], "fixed_recursive_kernel_stochastic_open_economy"
        )
        self.assertEqual(results["rkhs_coefficients"], 2 * 4 * 17)
        theta = extras["theta"]
        self.assertEqual(len(theta), 4)
        self.assertEqual(theta[0].shape, (4 * 17,))
        self.assertEqual(theta[1].shape, (4 * 17,))
        self.assertEqual(theta[2].shape, (2,))
        self.assertEqual(theta[3].shape, (4, 2))
        self.assertTrue(extras["train_df"]["fixed_reference_center"].all())

    def test_zero_theta_policy_is_the_anchored_replacement_policy(self):
        reference_policy = self.nn_extras["model"]
        kernel_set = kernel.StateKernelSettings()
        train_states = jnp.asarray(
            [
                [0.0, 0.0, -0.8, 0.0],
                [0.02, 0.4, -1.5, 0.01],
                [-0.03, 0.9, -3.0, -0.01],
            ],
            dtype=jnp.float64,
        )
        args = kernel.build_residual_args(train_states, kernel_set, reference_policy)
        theta = (
            jnp.zeros(3, dtype=jnp.float64),
            jnp.zeros(3, dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.zeros((4, 2), dtype=jnp.float64),
        )
        policy = kernel.make_policy(theta, args, kernel_set)
        outputs = jax.vmap(policy)(train_states)
        expected = jnp.stack(
            [
                jnp.full(3, reference_policy.investment_anchor[...]),
                jnp.full(3, reference_policy.current_account_anchor[...]),
            ],
            axis=-1,
        )

        np.testing.assert_array_equal(outputs, expected)


if __name__ == "__main__":
    unittest.main()
