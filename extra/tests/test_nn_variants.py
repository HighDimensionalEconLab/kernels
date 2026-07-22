import unittest

import jax

from extra.neoclassical_growth_nn_integrated import (
    DataSettings as IntegratedDataSettings,
)
from extra.neoclassical_growth_nn_integrated import (
    StoppingCriteria as IntegratedStoppingCriteria,
)
from extra.neoclassical_growth_nn_integrated import neoclassical_growth_nn_integrated
from extra.neoclassical_growth_nn_rff import DataSettings as RFFDataSettings
from extra.neoclassical_growth_nn_rff import StoppingCriteria as RFFStoppingCriteria
from extra.neoclassical_growth_nn_rff import neoclassical_growth_nn_rff

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class NNVariantSmokeTests(unittest.TestCase):
    def test_integrated_mlp_smoke(self):
        results, _ = neoclassical_growth_nn_integrated(
            verbose=False,
            data_set=IntegratedDataSettings(
                train_T=10.0,
                train_points=11,
                test_T=10.0,
                test_points=21,
                test_tvc_T=50.0,
                integration_substeps=4,
            ),
            stop_set=IntegratedStoppingCriteria(
                max_steps=120,
                train_loss_threshold=1e-8,
                num_attempts=1,
            ),
        )

        self.assertEqual(results["approximation"], "mlp")
        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)

    def test_rff_smoke(self):
        results, _ = neoclassical_growth_nn_rff(
            verbose=False,
            data_set=RFFDataSettings(
                train_T=10.0,
                train_points=11,
                test_T=10.0,
                test_points=21,
                test_tvc_T=50.0,
                integration_substeps=4,
            ),
            stop_set=RFFStoppingCriteria(
                max_steps=120,
                train_loss_threshold=1e-8,
                num_attempts=1,
            ),
        )

        self.assertEqual(results["approximation"], "rff")
        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertLess(results["test_mean_abs_k_rel_error"], 5e-2)


if __name__ == "__main__":
    unittest.main()
