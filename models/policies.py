from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from nlls_gram import DrawNNXModule

from models.kernels import rbf_state_features, rbf_state_gram, state_features
from models.utilities import ACTIVATIONS, load_model, save_model, unit_softplus

# The growth investment policy on (log z, log k), in two representations: an
# anchored unit-softplus MLP and a state-space RKHS expansion. Both serve the
# deterministic recursive and the stochastic growth solvers.


@dataclass(frozen=True)
class InvestmentPolicySettings:
    mlp_width: int = 32
    mlp_depth: int = 3
    hidden_activation: str = "gelu"
    head_init_scale: float = 1e-3
    homothetic_representation: bool = True


# i = anchor * U(h), U the unit softplus with near-zero head: homothetic is
# h(log kappa) with kappa = k / z^{1/(1-a)} the detrended capital, anchored at
# delta * k; non-homothetic h(centered log z, log k) anchored at the constant
# delta * k_scale; consumption Y - i stays unclamped.
class InvestmentPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: InvestmentPolicySettings,
        *,
        z_scale=1.0,
        k_scale=1.0,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.homothetic = nnx.static(settings.homothetic_representation)
        self.log_z_scale = nnx.Variable(jnp.asarray(math.log(z_scale), dtype=dtype))
        self.log_k_scale = nnx.Variable(jnp.asarray(math.log(k_scale), dtype=dtype))
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        in_features = 1 if self.homothetic else 2
        self.layers = nnx.List(
            [
                nnx.Linear(
                    in_features,
                    settings.mlp_width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(settings.mlp_depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.mlp_width,
                    settings.mlp_width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.head = nnx.Linear(
            settings.mlp_width,
            1,
            kernel_init=nnx.initializers.normal(settings.head_init_scale),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, log_state, p):
        log_z, log_k = log_state
        if self.homothetic:
            h = jnp.atleast_1d(log_k - log_z / (1.0 - p.a))
            anchor = p.delta * jnp.exp(log_k)
        else:
            h = jnp.stack(
                [log_z - self.log_z_scale[...], log_k - self.log_k_scale[...]]
            )
            anchor = p.delta * self.k_scale[...]
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return anchor * unit_softplus(self.head(h)[0])


# DrawNNXModule redraws theta but passes args through unchanged; multi-start
# chains each attempt's initial args from the previous draw, so re-chaining
# restart_key/epoch_key here (and resetting epoch to 0) is what keeps each
# retry's epoch-boundary resampling independent of the others.
class DrawInvestmentPolicy:
    def __init__(self, settings, *, z_scale, k_scale, dtype):
        self.draw_theta = DrawNNXModule(
            InvestmentPolicyNN, settings, z_scale=z_scale, k_scale=k_scale, dtype=dtype
        )

    def __call__(self, key, _, args):
        theta, _ = self.draw_theta(key, _, args)
        restart_key, epoch_key = jax.random.split(args.restart_key)
        return theta, args.replace(
            epoch_key=epoch_key,
            restart_key=restart_key,
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def __hash__(self):
        return hash((DrawInvestmentPolicy, self.draw_theta))

    def __eq__(self, other):
        return (
            isinstance(other, DrawInvestmentPolicy)
            and self.draw_theta == other.draw_theta
        )


# The load template only needs the architecture from meta: the scale
# Variables ride in the artifact's nondiff block and overwrite the defaults.
def make_investment_policy(meta, dtype):
    return InvestmentPolicyNN(
        InvestmentPolicySettings(**meta["settings"]), dtype=dtype, rngs=nnx.Rngs(0)
    )


def save_growth_policy(path, policy, settings, *, structural=None, extra=None):
    meta = {"settings": asdict(settings)}
    if structural is not None:
        meta["structural"] = {
            name: float(np.asarray(value))
            for name, value in structural.items()
            if np.asarray(value).ndim == 0
        }
    if extra:
        meta.update(extra)
    save_model(path, policy, meta)


def load_growth_policy(path):
    return load_model(path, make_investment_policy)


@jax.tree_util.register_static
@dataclass(frozen=True)
class StateKernelSettings:
    amplitude: float = 1.0
    lengthscale: float = 0.6
    metric_epsilon: float = 3e-5
    # With the free constant, theta is (alpha, h_const) and only alpha is
    # penalized; without it the policy is the pure RKHS expansion.
    free_constant: bool = True
    # Homothetic: 1-D RBF on log(k/z). Non-homothetic: a 2-D expansion on the
    # centered (log z, log k) with product (ard) or sum (additive) composition
    # of RBF or Matern 1-D kernels; lengthscales default per axis to
    # `lengthscale`, and linear_mean appends two unpenalized coefficients.
    homothetic_representation: bool = True
    state_kernel: Literal[
        "ard_rbf",
        "additive_rbf",
        "ard_matern32",
        "ard_matern52",
        "additive_matern32",
        "additive_matern52",
    ] = "ard_rbf"
    log_z_lengthscale: float | None = None
    log_k_lengthscale: float | None = None
    linear_mean: bool = False


@struct.dataclass
class KernelResidualArgs:
    centers: jax.Array
    train_log_states: jax.Array
    gram: jax.Array
    state_offset: jax.Array
    k_scale: jax.Array
    # 1/(1-a): log kappa = log k - z_detrend * log z for Hicks-neutral z.
    z_detrend: jax.Array


def state_lengthscales(kernel_set):
    ell_z = kernel_set.log_z_lengthscale
    ell_k = kernel_set.log_k_lengthscale
    return (
        kernel_set.lengthscale if ell_z is None else ell_z,
        kernel_set.lengthscale if ell_k is None else ell_k,
    )


def build_residual_args(train_log_states, kernel_set, a, z_0=1.0, k_0=1.0):
    flat_states = train_log_states.reshape(-1, 2)
    dtype = flat_states.dtype
    state_offset = jnp.log(jnp.asarray([z_0, k_0], dtype))
    z_detrend = jnp.asarray(1.0 / (1.0 - a), dtype)
    if kernel_set.homothetic_representation:
        centers = flat_states[:, 1] - z_detrend * flat_states[:, 0]
        gram = rbf_state_gram(
            centers,
            jnp.asarray(kernel_set.amplitude, dtype),
            jnp.asarray(kernel_set.lengthscale, dtype),
        )
    else:
        centers = flat_states - state_offset
        gram = state_features(
            kernel_set.state_kernel,
            centers,
            centers,
            kernel_set.amplitude,
            state_lengthscales(kernel_set),
        )
    gram = (gram + gram.T) / 2.0
    return KernelResidualArgs(
        centers=centers,
        train_log_states=flat_states,
        gram=gram,
        state_offset=state_offset,
        k_scale=jnp.asarray(k_0, dtype),
        z_detrend=z_detrend,
    )


def raw_policy(theta, args, log_state, kernel_set):
    alpha = theta[0]
    if kernel_set.homothetic_representation:
        q = log_state[1] - args.z_detrend * log_state[0]
        features = rbf_state_features(
            q,
            args.centers,
            jnp.asarray(kernel_set.amplitude, q.dtype),
            jnp.asarray(kernel_set.lengthscale, q.dtype),
        )
        h = features @ alpha
    else:
        x = log_state - args.state_offset
        features = state_features(
            kernel_set.state_kernel,
            x,
            args.centers,
            kernel_set.amplitude,
            state_lengthscales(kernel_set),
        )
        h = features @ alpha
        # The linear-mean coefficients are the last theta leaf, after the free
        # constant when both are present; like h_const they are unpenalized.
        if kernel_set.linear_mean:
            h = h + x @ theta[-1]
    if kernel_set.free_constant and len(theta) > 1:
        h = h + theta[1]
    return h


# Same output maps as the recursive NN: unit softplus of the raw policy
# anchored at delta * k (homothetic) or the constant delta * k_0.
def investment(theta, args, log_state, p, kernel_set):
    raw = raw_policy(theta, args, log_state, kernel_set)
    if kernel_set.homothetic_representation:
        k = jnp.exp(log_state[1])
        return p.delta * k * unit_softplus(raw)
    return p.delta * args.k_scale * unit_softplus(raw)


@struct.dataclass
class KernelInvestmentPolicy:
    theta: tuple
    args: KernelResidualArgs
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)

    def __call__(self, log_state, p):
        return investment(self.theta, self.args, log_state, p, self.kernel_set)
