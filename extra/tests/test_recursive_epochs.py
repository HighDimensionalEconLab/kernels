import unittest

import jax
import numpy as np

from neoclassical_growth_nn_recursive import (
    DataSettings,
    InitialConditionSettings,
    InvestmentPolicySettings,
    StoppingCriteria,
    neoclassical_growth_nn_recursive,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

SMALL_DATA = DataSettings(
    train_T=20.0, train_points=17, train_tvc_T=50.0, test_T=10.0, test_points=11
)
SMALL_IC = InitialConditionSettings(
    data_seed=0,
    num_train_trajectories=3,
    num_test_trajectories=2,
    train_log_z_variance=0.01,
    train_log_k_variance=0.01,
    test_log_z_variance=0.0025,
    test_log_k_variance=0.0025,
)
SMALL_NET = InvestmentPolicySettings(mlp_width=8, mlp_depth=2)


def solve(stop_set):
    return neoclassical_growth_nn_recursive(
        g=0.0,
        verbose=False,
        data_set=SMALL_DATA,
        ic_set=SMALL_IC,
        hc_set=SMALL_NET,
        stop_set=stop_set,
    )


class EpochModeTests(unittest.TestCase):
    def test_single_epoch_solve_matches_pinned_values(self):
        # Pinned same-seed reference values.
        results, _ = solve(
            StoppingCriteria(max_steps=60, num_attempts=2, train_loss_threshold=1e-12)
        )
        self.assertEqual(results["steps"], 43)
        self.assertEqual(results["stopping_reason"], "converged")
        self.assertEqual(results["attempt"], 1)
        self.assertTrue(results["solution_accepted"])
        self.assertEqual(results["num_epochs"], 1)
        self.assertEqual(results["epochs_run"], 1)
        np.testing.assert_allclose(
            results["train_loss"], 9.870304777681247e-13, rtol=1e-6
        )
        np.testing.assert_allclose(
            results["max_train_euler"], 2.186724533559392e-06, rtol=1e-6
        )

    def test_fixed_budget_epochs_resample_and_run_to_completion(self):
        results, extras = solve(
            StoppingCriteria(
                max_steps=4,
                num_attempts=1,
                num_epochs=3,
                epoch_early_stopping=False,
            )
        )
        self.assertEqual(results["steps"], 12)
        self.assertEqual(results["epochs_run"], 3)
        self.assertEqual(results["num_epochs"], 3)
        self.assertEqual(results["stopping_reason"], "max_steps")
        self.assertTrue(results["solution_accepted"])
        self.assertTrue(np.isfinite(results["train_loss"]))
        # The carried batch was resampled away from the initial draw.
        initial = np.asarray(extras["train_initial_conditions"])
        carried = (
            extras["train_df"].groupby("trajectory")[["z_0", "k_0"]].first().to_numpy()
        )
        self.assertGreater(np.abs(carried - initial).max(), 1e-4)

    def test_epoch_early_stopping_fires_at_a_boundary(self):
        results, _ = solve(
            StoppingCriteria(
                max_steps=5,
                num_attempts=1,
                num_epochs=6,
                train_loss_threshold=1e-2,
            )
        )
        self.assertEqual(results["stopping_reason"], "converged")
        self.assertEqual(results["steps"] % 5, 0)
        self.assertLess(results["steps"], 30)
        self.assertTrue(results["solution_accepted"])

    def test_epoch_mode_requires_a_positive_training_variance(self):
        with self.assertRaisesRegex(ValueError, "positive training log variance"):
            neoclassical_growth_nn_recursive(
                verbose=False,
                data_set=SMALL_DATA,
                hc_set=SMALL_NET,
                stop_set=StoppingCriteria(max_steps=4, num_epochs=2),
            )


if __name__ == "__main__":
    unittest.main()
