import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


# Per-model shared constants: every solver variant of a model family defaults
# to the same instance so they solve the identical model on the same windows,
# and other families can define their own instances alongside.
#
# The calibration is mapped exactly from quarterly anchors:
# e^(-rho_hat/4) = beta, e^(-eta/4) = rho, and the OU innovation scale
# matches the AR(1) stationary variance, sigma_z^2/(2 eta) =
# sigma^2/(1 - rho^2). Recursive drivers roll held-out initial conditions
# for test_T, while sequential drivers evaluate their path solution on a
# grid extending past train_T to test extrapolation.
@dataclass(frozen=True)
class GrowthConstants:
    a: float
    delta: float
    sigma_crra: float
    quarterly_beta: float
    quarterly_tfp_rho: float
    quarterly_tfp_sigma_benchmark: float
    quarterly_tfp_sigma_high: float
    train_T: float
    test_T: float
    sequence_test_T: float
    transversality_T: float
    benchmark_T: float
    benchmark_points: int

    @property
    def rho_hat(self):
        return -4.0 * math.log(self.quarterly_beta)

    @property
    def eta(self):
        return -4.0 * math.log(self.quarterly_tfp_rho)

    @property
    def z_ergodic_variance_benchmark(self):
        return self.quarterly_tfp_sigma_benchmark**2 / (1.0 - self.quarterly_tfp_rho**2)

    @property
    def z_ergodic_variance_high(self):
        return self.quarterly_tfp_sigma_high**2 / (1.0 - self.quarterly_tfp_rho**2)

    @property
    def sigma_z_benchmark(self):
        return math.sqrt(2.0 * self.eta * self.z_ergodic_variance_benchmark)

    @property
    def sigma_z_high(self):
        return math.sqrt(2.0 * self.eta * self.z_ergodic_variance_high)


# Neoclassical growth, from Aruoba, Fernandez-Villaverde & Rubio-Ramirez
# (2006 JEDC, Tables 1-2): beta = 0.9896 (a 4% annual interest rate), log-TFP
# AR(1) rho = 0.95 with innovation sd 0.007 (their high-variance sensitivity
# column is 0.035), capital share 1/3, annual depreciation 0.1 (King-Rebelo
# 1999's 0.025 quarterly). sigma_crra = 2 is their tau and is a free choice
# rather than a mapped value. train_T = 100 covers ~10 e-foldings of the
# calibrated saddle path (|lambda_stable| ~ 0.105); the transversality
# certificate and the benchmark BVP extend to twice that.
NEOCLASSICAL_GROWTH_CONSTANTS = GrowthConstants(
    a=1 / 3,
    delta=0.1,
    sigma_crra=2.0,
    quarterly_beta=0.9896,
    quarterly_tfp_rho=0.95,
    quarterly_tfp_sigma_benchmark=0.007,
    quarterly_tfp_sigma_high=0.035,
    train_T=100.0,
    test_T=40.0,
    sequence_test_T=110.0,
    transversality_T=200.0,
    benchmark_T=200.0,
    benchmark_points=801,
)


# Gram-form Levenberg-Marquardt settings shared across solvers; each solver
# pins its tuned values inline in its opt_set signature default.
@dataclass(frozen=True)
class OptimizerSettings:
    init_damping: float = 1e-5
    damping_decrease: float = 0.35
    damping_increase: float = 8.0
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.8


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale
# without shrinking the local Jacobian by softplus'(0).
def unit_softplus(raw):
    log_two = jnp.log(jnp.asarray(2.0, dtype=raw.dtype))
    return jax.nn.softplus(2.0 * log_two * raw) / log_two


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "softplus": jax.nn.softplus,
    "unit_softplus": unit_softplus,
    "identity": lambda x: x,
}
