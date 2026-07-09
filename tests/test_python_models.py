import math
import unittest

import jax.numpy as jnp
import numpy as np
import scipy.integrate

from asset_pricing_kernel import asset_pricing_kernel
from asset_pricing_kernel_nlls import DataSettings as APDataSettings
from asset_pricing_kernel_nlls import KernelSettings as APKernelSettings
from asset_pricing_kernel_nlls import (
    NLLSGramSettings,
    asset_pricing_kernel_nlls,
)
from human_capital_kernel import human_capital_initial_conditions, human_capital_kernel
from kernels import (
    gaussian_kernel,
    integrated_kernel_matrices,
    matern_kernel_0p5,
    matern_kernel_1p5,
    matern_kernel_2p5,
    matern_kernel_matvec_pair,
    matern_qsm_solve,
    normalize_kernel,
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
from neoclassical_growth_kernel_nlls import (
    DataSettings as GrowthNLLSDataSettings,
)
from neoclassical_growth_kernel_nlls import (
    KernelSettings as GrowthNLLSKernelSettings,
)
from neoclassical_growth_kernel_nlls import (
    NLLSGramSettings as GrowthNLLSSettings,
)
from neoclassical_growth_kernel_nlls import (
    neoclassical_growth_kernel_nlls,
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
FAST_GROWTH_NLLS = GrowthNLLSDataSettings(train_T=10.0, train_points=11, test_T=10.0)


class KernelSmokeTests(unittest.TestCase):
    def test_integrated_kernels_match_quadrature(self):
        # (32.161, 5.829) is where the removed adaptive-quadrature versions
        # silently missed their tolerance by ~2.6e-4.
        kernel_fns = {
            "matern12": matern_kernel_0p5,
            "matern32": matern_kernel_1p5,
            "matern52": matern_kernel_2p5,
            "rbf": gaussian_kernel,
        }
        geometries = ((0.0, 3.0), (5.0, 1.0), (10.0, 8.0), (32.161, 5.829))
        for name, kernel_fn in kernel_fns.items():
            for ell in (2.0, 10.0):
                for t_i, t_j in geometries:
                    with self.subTest(kernel=name, ell=ell, t_i=t_i, t_j=t_j):
                        _, K_tilde = integrated_kernel_matrices(
                            jnp.array([t_i]), jnp.array([t_j]), name, 1.3, ell
                        )
                        reference = scipy.integrate.quad(
                            lambda s: float(kernel_fn(s, t_j, 1.3, ell)),
                            0.0,
                            t_i,
                            points=[t_j] if 0.0 < t_j < t_i else None,
                            epsabs=1e-13,
                            epsrel=1e-13,
                            limit=200,
                        )[0]
                        self.assertLess(abs(float(K_tilde[0, 0]) - reference), 1e-11)

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

    def test_neoclassical_growth_nlls_smoke(self):
        results, extras = neoclassical_growth_kernel_nlls(data_set=FAST_GROWTH_NLLS)

        self.assertEqual(results["method_resolved"], "dense")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-8)
        self.assertLess(results["k_rel_error_mean"], 2e-2)
        self.assertLess(results["c_rel_error_mean"], 2e-2)
        self.assertTrue((extras["test_df"]["k"] > 0.0).all())
        self.assertTrue((extras["test_df"]["c"] > 0.0).all())

        # matrix_free recovers the same selection as dense at scale.
        data_set = GrowthNLLSDataSettings(train_points=501)
        dense, _ = neoclassical_growth_kernel_nlls(
            data_set=data_set, lm_set=GrowthNLLSSettings(method="dense")
        )
        structured, _ = neoclassical_growth_kernel_nlls(
            data_set=data_set, lm_set=GrowthNLLSSettings(method="matrix_free")
        )
        self.assertEqual(structured["method_resolved"], "matrix_free")
        self.assertTrue(structured["solution_accepted"])
        self.assertLess(abs(structured["c_0"] - dense["c_0"]), 1e-4)

    def test_neoclassical_growth_nlls_matern32_auto_stays_dense(self):
        # The matern32 iterative window from asset pricing fails the growth
        # gates at scale (measured at n >= 501): auto must not pick it.
        results, _ = neoclassical_growth_kernel_nlls(
            kernel_set=GrowthNLLSKernelSettings(kernel="matern32"),
            data_set=GrowthNLLSDataSettings(train_points=501),
        )
        self.assertEqual(results["method_resolved"], "dense")
        self.assertTrue(results["solution_accepted"], results["rejection_reason"])

    def test_neoclassical_growth_nlls_kernels_smoke(self):
        for kernel_set, error_bar in [
            (GrowthNLLSKernelSettings(kernel="matern32"), 2e-2),
            (GrowthNLLSKernelSettings(kernel="matern52"), 2e-2),
            (GrowthNLLSKernelSettings(kernel="rbf", ell=2.0), 2e-2),
        ]:
            with self.subTest(kernel=kernel_set.kernel):
                results, _ = neoclassical_growth_kernel_nlls(
                    kernel_set=kernel_set, data_set=FAST_GROWTH_NLLS
                )
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["k_rel_error_mean"], error_bar)

    def test_asset_pricing_smoke(self):
        results, extras = asset_pricing_kernel()

        self.assertTrue(results["solution_accepted"], results["rejection_reason"])
        self.assertLess(results["max_train_residual"], 1e-10)
        self.assertLess(results["p_rel_error_mean"], 5e-3)
        self.assertTrue((extras["test_df"]["p"] > 0.0).all())

    def test_matern_matvecs_match_dense(self):
        rng = np.random.default_rng(7)
        t = jnp.asarray(np.sort(rng.uniform(0.0, 50.0, 501)))
        v = jnp.asarray(rng.standard_normal(501))
        for name in ("matern12", "matern32", "matern52"):
            K, K_tilde = integrated_kernel_matrices(t, t, name, 1.3, 10.0)
            K = (K + K.T) / 2
            for parallel in (False, True):
                with self.subTest(kernel=name, parallel=parallel):
                    k_v, k_tilde_v = matern_kernel_matvec_pair(
                        name, t, v, 1.3, 10.0, parallel=parallel
                    )
                    self.assertLess(float(jnp.max(jnp.abs(K @ v - k_v))), 1e-10)
                    self.assertLess(
                        float(jnp.max(jnp.abs(K_tilde @ v - k_tilde_v))), 1e-10
                    )

    def test_matern_qsm_solve_matches_dense(self):
        rng = np.random.default_rng(8)
        t = jnp.linspace(0.0, 50.0, 300)
        v = jnp.asarray(rng.standard_normal(300))
        for name in ("matern32", "matern52"):
            with self.subTest(kernel=name):
                jitter = 1e-8 * 1.3**2
                K, _ = integrated_kernel_matrices(t, t, name, 1.3, 10.0)
                K_jittered = np.asarray((K + K.T) / 2) + jitter * np.eye(300)
                solution = matern_qsm_solve(name, t, 1.3, 10.0, jitter)(v)
                reference = np.linalg.solve(K_jittered, np.asarray(v))
                self.assertLess(
                    float(np.max(np.abs(np.asarray(solution) - reference)))
                    / float(np.max(np.abs(reference))),
                    1e-6,
                )

    def test_asset_pricing_nlls_smoke(self):
        reference, extras = asset_pricing_kernel_nlls()

        self.assertEqual(reference["method_resolved"], "dense")
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

    def test_asset_pricing_nlls_kernels_smoke(self):
        # Every named kernel solves on the dense path; the smoother kernels
        # need their documented working configurations (rbf: short length
        # scale -- its Gram at ell = 10 is numerically singular for every
        # solver, IPOPT included).
        cases = {
            "matern32": (APKernelSettings(kernel="matern32"), 2e-2),
            "matern52": (APKernelSettings(kernel="matern52"), 2e-2),
            "rbf": (APKernelSettings(kernel="rbf", ell=2.0), 1e-1),
        }
        for name, (kernel_set, error_bar) in cases.items():
            with self.subTest(kernel=name):
                results, _ = asset_pricing_kernel_nlls(kernel_set=kernel_set)

                self.assertEqual(results["method_resolved"], "dense")
                self.assertTrue(
                    results["solution_accepted"], results["rejection_reason"]
                )
                self.assertLess(results["max_train_residual"], 1e-8)
                self.assertLess(results["p_rel_error_mean"], error_bar)

        # matern32 matrix_free agrees with its dense selection.
        kernel_set = APKernelSettings(kernel="matern32")
        data_set = APDataSettings(train_T=50.0, train_points=1001)
        dense, _ = asset_pricing_kernel_nlls(
            kernel_set=kernel_set,
            data_set=data_set,
            lm_set=NLLSGramSettings(method="dense"),
        )
        structured, _ = asset_pricing_kernel_nlls(
            kernel_set=kernel_set,
            data_set=data_set,
            lm_set=NLLSGramSettings(method="matrix_free"),
        )
        self.assertTrue(structured["solution_accepted"])
        self.assertLess(abs(structured["p_0"] - dense["p_0"]), 1e-8)

    def test_normalize_kernel_aliases(self):
        cases = [
            (0.5, "matern12"),
            (1.5, "matern32"),
            (2.5, "matern52"),
            (math.inf, "rbf"),
            ("gaussian", "rbf"),
            ("se", "rbf"),
            ("matern_1_2", "matern12"),
            ("matern2.5", "matern52"),
        ]
        for alias, expected in cases:
            self.assertEqual(normalize_kernel(alias), expected)
        with self.assertRaises(ValueError):
            normalize_kernel("matern72")

    def test_asset_pricing_nlls_jitter_contract(self):
        # A nonzero nugget applies identically across methods, so their
        # min-norm selections agree.
        kernel_set = APKernelSettings(kernel="matern32")
        data_set = APDataSettings(train_T=50.0, train_points=501)
        jitter = 1e-6
        dense, _ = asset_pricing_kernel_nlls(
            kernel_set=kernel_set,
            data_set=data_set,
            lm_set=NLLSGramSettings(method="dense", metric_jitter=jitter),
        )
        structured, _ = asset_pricing_kernel_nlls(
            kernel_set=kernel_set,
            data_set=data_set,
            lm_set=NLLSGramSettings(method="matrix_free", metric_jitter=jitter),
        )
        self.assertTrue(structured["solution_accepted"])
        self.assertLess(abs(structured["p_0"] - dense["p_0"]), 1e-8)

        # matern12 matrix_free is nugget-free by construction.
        with self.assertRaises(ValueError):
            asset_pricing_kernel_nlls(
                lm_set=NLLSGramSettings(method="matrix_free", metric_jitter=1e-8)
            )

    def test_asset_pricing_nlls_auto_dispatch(self):
        small, _ = asset_pricing_kernel_nlls()
        self.assertEqual(small["method_resolved"], "dense")

        large, _ = asset_pricing_kernel_nlls(
            data_set=APDataSettings(train_T=50.0, train_points=501)
        )
        self.assertEqual(large["method_resolved"], "matrix_free")
        self.assertTrue(large["solution_accepted"])

        rbf, _ = asset_pricing_kernel_nlls(
            kernel_set=APKernelSettings(kernel="rbf", ell=2.0),
            data_set=APDataSettings(train_T=50.0, train_points=501),
        )
        self.assertEqual(rbf["method_resolved"], "dense")

        with self.assertRaises(ValueError):
            asset_pricing_kernel_nlls(
                kernel_set=APKernelSettings(kernel="matern52"),
                lm_set=NLLSGramSettings(method="matrix_free"),
            )

        # An unsorted grid at scale sends auto to preconditioned_cg (its
        # niche) and correctly refuses matrix_free.
        points = np.linspace(0.0, 50.0, 1600)
        points[10], points[11] = points[11], points[10]
        unsorted, _ = asset_pricing_kernel_nlls(
            data_set=APDataSettings(train_points_list=[float(p) for p in points])
        )
        self.assertEqual(unsorted["method_resolved"], "preconditioned_cg")
        self.assertTrue(unsorted["solution_accepted"], unsorted["rejection_reason"])
        with self.assertRaises(ValueError):
            asset_pricing_kernel_nlls(
                data_set=APDataSettings(train_points_list=[float(p) for p in points]),
                lm_set=NLLSGramSettings(method="matrix_free"),
            )

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
