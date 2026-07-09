import math
import unittest

import jax.numpy as jnp
import numpy as np

from asset_pricing_kernel import asset_pricing_kernel
from asset_pricing_kernel_nlls import NLLSGramSettings, asset_pricing_kernel_nlls
from human_capital_kernel import human_capital_initial_conditions, human_capital_kernel
from kernels import (
    integrated_gaussian_kernel,
    integrated_gaussian_kernel_quad,
    matern_kernel_0p5_matvec_pair,
)
from neoclassical_growth_concave_convex_kernel import (
    neoclassical_growth_concave_convex_kernel,
)
from neoclassical_growth_kernel import (
    DataSettings as GrowthDataSettings,
)
from neoclassical_growth_kernel import (
    KernelSettings,
    neoclassical_growth_kernel,
)
from neoclassical_growth_nn import (
    DataSettings as GrowthNNDataSettings,
)
from neoclassical_growth_nn import (
    GrowthPolicySettings as GrowthNNSettings,
)
from neoclassical_growth_nn import (
    StoppingCriteria as GrowthNNStoppingCriteria,
)
from neoclassical_growth_nn import (
    neoclassical_growth_nn,
)
from neoclassical_growth_nn_integrated import (
    DataSettings as IntegratedNNDataSettings,
)
from neoclassical_growth_nn_integrated import (
    GrowthDerivativeSettings as IntegratedNNSettings,
)
from neoclassical_growth_nn_integrated import (
    StoppingCriteria as IntegratedNNStoppingCriteria,
)
from neoclassical_growth_nn_integrated import (
    neoclassical_growth_nn_integrated,
)
from neoclassical_growth_nn_recursive_i import (
    DataSettings as RecursiveIDataSettings,
)
from neoclassical_growth_nn_recursive_i import (
    StoppingCriteria as RecursiveIStoppingCriteria,
)
from neoclassical_growth_nn_recursive_i import (
    neoclassical_growth_nn_recursive_i,
)
from optimal_advertising_kernel import optimal_advertising_kernel

FAST_GROWTH_DATA = GrowthDataSettings(train_T=10.0, train_points=11, test_T=10.0)


class KernelSmokeTests(unittest.TestCase):
    def test_integrated_gaussian_kernel_matches_quadrature(self):
        for ell in (2.0, 10.0):
            for t_i, t_j in ((0.0, 3.0), (5.0, 1.0), (10.0, 8.0)):
                analytic = float(integrated_gaussian_kernel(t_i, t_j, 1.0, ell))
                quad = float(integrated_gaussian_kernel_quad(t_i, t_j, 1.0, ell))
                self.assertAlmostEqual(analytic, quad, places=8)

    def test_neoclassical_growth_smoke(self):
        results, extras = neoclassical_growth_kernel(data_set=FAST_GROWTH_DATA)

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["k_rel_error_mean"], 2e-2)
        self.assertLess(results["c_rel_error_mean"], 2e-2)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_neoclassical_growth_rbf_smoke(self):
        # Gaussian/RBF limit with a short length scale (well conditioned).
        results, _ = neoclassical_growth_kernel(
            kernel_set=KernelSettings(nu=math.inf, ell=2.0),
            data_set=FAST_GROWTH_DATA,
        )

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["k_rel_error_mean"], 2e-2)

    def test_concave_convex_smoke(self):
        results, extras = neoclassical_growth_concave_convex_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-5)
        self.assertLess(results["max_validation_residual"], 5e-3)
        self.assertLess(results["solve_time"], 0.5)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

    def test_asset_pricing_smoke(self):
        results, extras = asset_pricing_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-10)
        self.assertLess(results["p_rel_error_mean"], 5e-3)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

    def test_matern_matvec_parallel_matches_sequential(self):
        rng = np.random.default_rng(7)
        t = jnp.asarray(np.sort(rng.uniform(0.0, 50.0, 1001)))
        v = jnp.asarray(rng.standard_normal(1001))
        k_seq, kt_seq = matern_kernel_0p5_matvec_pair(t, v, 1.3, 10.0, parallel=False)
        k_par, kt_par = matern_kernel_0p5_matvec_pair(t, v, 1.3, 10.0, parallel=True)

        self.assertLess(float(jnp.max(jnp.abs(k_seq - k_par))), 1e-11)
        self.assertLess(float(jnp.max(jnp.abs(kt_seq - kt_par))), 1e-11)

    def test_asset_pricing_nlls_smoke(self):
        reference, extras = asset_pricing_kernel_nlls()

        self.assertTrue(reference["solution_accepted"], reference["rejection_reason"])
        self.assertLess(reference["max_train_residual"], 1e-10)
        self.assertLess(reference["p_rel_error_mean"], 5e-3)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

        # The structured CG methods must recover the same min-norm selection
        # as the dense default, not just interpolate.
        for method in ["preconditioned_cg", "matrix_free"]:
            with self.subTest(method=method):
                lm_set = NLLSGramSettings(method=method)
                results, extras = asset_pricing_kernel_nlls(lm_set=lm_set)

                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["max_train_residual"], 1e-10)
                self.assertLess(results["p_rel_error_mean"], 5e-3)
                self.assertLess(abs(results["p_0"] - reference["p_0"]), 1e-9)
                self.assertLess(
                    abs(results["rkhs_norm_p"] - reference["rkhs_norm_p"]), 1e-9
                )
                self.assertTrue((extras["test_df"]["p"] > 0.0).all())

    def test_optimal_advertising_smoke(self):
        results, extras = optimal_advertising_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertAlmostEqual(results["gamma"], 0.32)
        self.assertTrue((extras["test_df"]["u"] > 0.0).all())

    def test_human_capital_smoke(self):
        h_0, c_0, residual, _ = human_capital_initial_conditions()
        self.assertAlmostEqual(float(h_0), 1.3745155888757778, places=10)
        self.assertLess(float(jnp.linalg.norm(residual, ord=jnp.inf)), 1e-12)

        results, extras = human_capital_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertEqual(results["num_train_points"], 81)
        self.assertAlmostEqual(results["h_0"], 1.3745155888757778, places=10)
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["max_test_feasibility"], 1e-3)
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())


class NNSmokeTests(unittest.TestCase):
    def test_neoclassical_growth_nn_smoke(self):
        results, frames = neoclassical_growth_nn(
            verbose=False,
            measure_compilation=False,
            data_set=GrowthNNDataSettings(
                train_T=10.0,
                train_points=11,
                test_T=10.0,
                test_points=21,
                test_tvc_T=50.0,
            ),
            nn_set=GrowthNNSettings(width=16, depth=2),
            stop_set=GrowthNNStoppingCriteria(
                max_steps=40,
                train_loss_threshold=1e-10,
                num_attempts=1,
            ),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)
        self.assertTrue((frames["test_df"]["k"] > 0.0).all())
        self.assertIn("nn_set", frames["hyperparameters"])

    def test_neoclassical_growth_nn_integrated_rff_smoke(self):
        results, frames = neoclassical_growth_nn_integrated(
            verbose=False,
            measure_compilation=False,
            data_set=IntegratedNNDataSettings(
                train_T=10.0,
                train_points=11,
                test_T=10.0,
                test_points=21,
                test_tvc_T=50.0,
                integration_substeps=4,
            ),
            nn_set=IntegratedNNSettings(
                approximation="rff",
                width=1024,
                rff_kernel="matern12",
                rff_radius=10.0,
                rff_sigma=1.0,
            ),
            stop_set=IntegratedNNStoppingCriteria(
                max_steps=120,
                train_loss_threshold=1e-8,
                num_attempts=1,
            ),
        )

        self.assertEqual(results["approximation"], "rff")
        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)

    def test_recursive_i_smoke(self):
        results, extras = neoclassical_growth_nn_recursive_i(
            verbose=False,
            measure_compilation=False,
            data_set=RecursiveIDataSettings(
                dt=2.0, train_T=20.0, train_tvc_T=100.0, test_T=20.0
            ),
            stop_set=RecursiveIStoppingCriteria(max_steps=200, num_attempts=1),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertTrue(results["transversality_ok"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)
        self.assertGreater(results["min_test_c"], 0.0)
        self.assertIn("i_nn", extras["policy_df"].columns)


if __name__ == "__main__":
    unittest.main()
