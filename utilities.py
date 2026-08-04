from dataclasses import dataclass

import jax
import jax.numpy as jnp


# Gram-form Levenberg-Marquardt settings shared across solvers (override via
# opt_set); drivers pin their tuned values inline in the signature default.
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
