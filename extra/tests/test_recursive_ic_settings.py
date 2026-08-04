import unittest

import jax
import numpy as np

from neoclassical_growth_kernel_recursive import (
    DataSettings,
    neoclassical_growth_kernel_recursive,
)
from neoclassical_growth_nn_recursive import InitialConditionSettings

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

FAST_DATA = DataSettings(test_T=5.0, test_points=6)


class TestSettingsInheritanceTests(unittest.TestCase):
    def test_none_test_variances_inherit_the_train_variances(self):
        results, extras = neoclassical_growth_kernel_recursive(
            verbose=False,
            data_set=FAST_DATA,
            ic_set=InitialConditionSettings(
                num_train_trajectories=5,
                num_test_trajectories=4,
                train_log_z_variance=0.01,
                train_log_k_variance=0.01,
            ),
        )
        self.assertEqual(results["test_log_z_variance"], 0.01)
        self.assertEqual(results["test_log_k_variance"], 0.01)
        test_ics = np.asarray(extras["test_initial_conditions"])
        log_test = np.log(test_ics)
        for axis in range(2):
            self.assertAlmostEqual(np.mean(log_test[:, axis] ** 2), 0.01, places=12)

    def test_explicit_test_variances_are_honored_per_axis(self):
        results, extras = neoclassical_growth_kernel_recursive(
            verbose=False,
            data_set=FAST_DATA,
            ic_set=InitialConditionSettings(
                num_train_trajectories=5,
                num_test_trajectories=4,
                train_log_z_variance=0.01,
                train_log_k_variance=0.01,
                test_log_z_variance=0.0,
                test_log_k_variance=0.04,
            ),
        )
        self.assertEqual(results["test_log_z_variance"], 0.0)
        self.assertEqual(results["test_log_k_variance"], 0.04)
        test_ics = np.asarray(extras["test_initial_conditions"])
        np.testing.assert_array_equal(test_ics[:, 0], 1.0)
        self.assertAlmostEqual(np.mean(np.log(test_ics[:, 1]) ** 2), 0.04, places=12)

    def test_shifted_test_centers_recenter_the_same_standardized_cloud(self):
        base_kwargs = dict(
            verbose=False,
            data_set=FAST_DATA,
            ic_set=InitialConditionSettings(
                num_train_trajectories=5,
                num_test_trajectories=4,
                train_log_z_variance=0.01,
                train_log_k_variance=0.01,
                test_log_z_variance=0.0025,
                test_log_k_variance=0.0025,
            ),
        )
        results, extras = neoclassical_growth_kernel_recursive(**base_kwargs)
        shift = float(np.exp(0.25))
        shifted_ic = InitialConditionSettings(
            num_train_trajectories=5,
            num_test_trajectories=4,
            train_log_z_variance=0.01,
            train_log_k_variance=0.01,
            test_log_z_variance=0.0025,
            test_log_k_variance=0.0025,
            test_k_0=shift,
        )
        shifted_results, shifted_extras = neoclassical_growth_kernel_recursive(
            verbose=False, data_set=FAST_DATA, ic_set=shifted_ic
        )
        self.assertEqual(shifted_results["test_k_0"], shift)
        base_ics = np.asarray(extras["test_initial_conditions"])
        shifted_ics = np.asarray(shifted_extras["test_initial_conditions"])
        np.testing.assert_allclose(shifted_ics[:, 0], base_ics[:, 0], rtol=1e-13)
        np.testing.assert_allclose(
            np.log(shifted_ics[:, 1]), np.log(base_ics[:, 1]) + 0.25, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
