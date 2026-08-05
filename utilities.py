import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from nlls_gram import LMStatus

STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


# solution_accepted / rejection_reason results-dict fields from the collected
# gate failures; an empty list is acceptance.
def acceptance_fields(rejection_reasons):
    accepted = not rejection_reasons
    return {
        "solution_accepted": accepted,
        "rejection_reason": "accepted" if accepted else ",".join(rejection_reasons),
    }


# perf_counter anchored at construction; elapsed(x) blocks on x first so
# asynchronous dispatch cannot leak out of the timed span.
class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self, value=None):
        if value is not None:
            jax.block_until_ready(value)
        return time.perf_counter() - self.start


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


def make_train_t_grid(data_set, dtype):
    if data_set.train_points_list is None:
        values = np.linspace(0.0, data_set.train_T, data_set.train_points).tolist()
    else:
        values = list(data_set.train_points_list)
    values = sorted(set(float(value) for value in values))
    return jnp.asarray(values, dtype=dtype)


# uniform | front_loaded sampling (dense head, sparse tail); an explicit
# train_points_list overrides the sampling family.
def make_sampled_train_grid(data_set, dtype):
    if data_set.train_points_list is not None:
        train_data = jnp.sort(jnp.asarray(data_set.train_points_list, dtype=dtype))
    elif data_set.sampling == "uniform":
        train_data = jnp.linspace(
            0.0, data_set.train_T, data_set.train_points, dtype=dtype
        )
    elif data_set.sampling == "front_loaded":
        dense_points = data_set.train_points - data_set.tail_points
        if (
            dense_points < 2
            or data_set.tail_points < 1
            or not 0.0 < data_set.dense_T < data_set.train_T
        ):
            raise ValueError(
                "front_loaded sampling requires at least 2 dense points, "
                "1 tail point, and 0 < dense_T < train_T"
            )
        dense_data = jnp.linspace(0.0, data_set.dense_T, dense_points, dtype=dtype)
        tail_data = jnp.linspace(
            data_set.dense_T, data_set.train_T, data_set.tail_points + 1, dtype=dtype
        )[1:]
        train_data = jnp.concatenate([dense_data, tail_data])
    else:
        raise ValueError("sampling must be 'front_loaded' or 'uniform'")
    if len(train_data) < 3 or not bool(jnp.all(jnp.diff(train_data) > 0.0)):
        raise ValueError("training points must be strictly increasing")
    if float(train_data[0]) != 0.0:
        raise ValueError("training points must start at zero")
    return train_data


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
