import hashlib
import json
import math
import pathlib
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from nlls_gram import LMStatus
from tinydiffeq import SRA1, EulerMaruyama, Milstein

DOMAIN_EPS = 1e-8

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


# Directional derivative d/dtau f(x + tau v) at tau = 0 by central
# differences, for double-checking implicit-AD sensitivities. Host-side and
# jit-free: f maps a concrete pytree to a pytree of concrete arrays. eps_f
# is the measured noise floor of f -- an iterative inner solver's stopping
# tolerance usually sets it, far above machine eps -- and the step balances
# that noise against the stencil's truncation order: h ~ eps_f^(1/5) for
# the five-point order-4 stencil, h ~ (3 eps_f)^(1/3) for the two-point
# order-2 one, scaled by the magnitude of x along the direction when that
# exceeds one. Steps are taken along the unit direction, so h is the actual
# perturbation size in x space. Returns (derivative, h).
def finite_difference_directional(f, x, v, eps_f, order=4):
    x_leaves = jax.tree_util.tree_leaves(x)
    v_leaves = jax.tree_util.tree_leaves(v)
    v_norm = math.sqrt(sum(float(jnp.sum(jnp.square(leaf))) for leaf in v_leaves))
    if v_norm == 0.0:
        raise ValueError("the direction v must be nonzero")
    x_along_v = (
        sum(
            float(jnp.sum(x_leaf * v_leaf))
            for x_leaf, v_leaf in zip(x_leaves, v_leaves, strict=True)
        )
        / v_norm
    )
    scale = max(1.0, abs(x_along_v))
    if order == 4:
        h = eps_f ** (1.0 / 5.0) * scale
        stencil = ((-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0))
    elif order == 2:
        h = (3.0 * eps_f) ** (1.0 / 3.0) * scale
        stencil = ((-1.0, -1.0), (1.0, 1.0))
    else:
        raise ValueError("order must be 2 or 4")
    outputs = [
        f(
            jax.tree_util.tree_map(
                lambda x_leaf, v_leaf: x_leaf + (offset * h / v_norm) * v_leaf, x, v
            )
        )
        for offset, _ in stencil
    ]
    # The stencil runs in tau with step h / v_norm, so the weighted sum over
    # a denominator of (12 or 2) * (h / v_norm) is d f / d tau directly.
    denominator = (12.0 if order == 4 else 2.0) * h / v_norm
    weights = [weight / denominator for _, weight in stencil]
    derivative = jax.tree_util.tree_map(
        lambda *values: sum(w * value for w, value in zip(weights, values)),
        *outputs,
    )
    return derivative, h


# Long-format DataFrame from a dict of same-shaped arrays: one row per
# element, index columns from np.indices (one per index name), one data
# column per key. `scalars` maps column name -> python scalar, or array
# indexed by the FIRST index (e.g. one value per trajectory), broadcast
# across that index's rows.
def long_df(tree, index_names, scalars=None):
    named = jax.device_get(dict(tree))
    shape = next(iter(named.values())).shape
    idx = np.indices(shape).reshape(len(index_names), -1)
    data = dict(zip(index_names, idx))
    for name, value in (scalars or {}).items():
        value = np.asarray(value)
        data[name] = (
            value[idx[0]] if value.ndim else np.broadcast_to(value, idx[0].shape)
        )
    data.update({name: arr.reshape(-1) for name, arr in named.items()})
    return pd.DataFrame(data)


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
# certificate extends to twice that.
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
)


@dataclass(frozen=True)
class InitialConditionSettings:
    # One data_seed fixes independent train and test draws around (z_0, k_0),
    # separate from the NN seed; a zero log variance pins its axis at the base.
    data_seed: int = 0
    num_train_trajectories: int = 1
    num_test_trajectories: int = 50
    train_log_z_variance: float = 0.0
    train_log_k_variance: float = 0.0
    # None inherits the training variances; results record the resolved values.
    test_log_z_variance: float | None = None
    test_log_k_variance: float | None = None
    # Recenter the held-out draws when set (out-of-distribution testing).
    test_z_0: float | None = None
    test_k_0: float | None = None


def draw_initial_conditions(
    z_0, k_0, num_trajectories, log_z_variance, log_k_variance, key, dtype
):
    base = jnp.asarray([z_0, k_0], dtype=dtype)
    if log_z_variance == 0.0 and log_k_variance == 0.0:
        return jnp.tile(base, (num_trajectories, 1))
    if num_trajectories < 2:
        raise ValueError("positive log variances require at least two trajectories")
    draws = jax.random.normal(key, (num_trajectories, 2), dtype=dtype)
    # Center and standardize each column so every axis carries exactly the
    # requested population variance; a zero variance pins its axis at the base.
    centered = draws - jnp.mean(draws, axis=0, keepdims=True)
    standardized = centered / jnp.sqrt(jnp.mean(centered**2, axis=0, keepdims=True))
    scales = jnp.sqrt(jnp.asarray([log_z_variance, log_k_variance], dtype=dtype))
    return jnp.exp(jnp.log(base)[None, :] + scales[None, :] * standardized)


def standardized_initial_states(key, center, scales, num_trajectories):
    # Additive in the caller's coordinate, unlike the multiplicative
    # draw_initial_conditions: additive spreads keep a level axis
    # non-degenerate at a zero center. One trajectory pins the batch at the
    # center.
    if num_trajectories == 1:
        return center[None, :]
    draws = jax.random.normal(key, (num_trajectories, center.size), dtype=center.dtype)
    centered = draws - jnp.mean(draws, axis=0, keepdims=True)
    standardized = centered / jnp.sqrt(jnp.mean(centered**2, axis=0, keepdims=True))
    return center[None, :] + scales[None, :] * standardized


# Train and test draws from one data_seed; the resolved test design rides
# along for the results dicts.
@dataclass(frozen=True)
class InitialConditionDraw:
    train_ic_key: jax.Array
    test_ic_key: jax.Array
    train_initial_conditions: jax.Array
    test_initial_conditions: jax.Array
    test_z_center: float
    test_k_center: float
    test_log_z_variance: float
    test_log_k_variance: float


def growth_initial_conditions(z_0, k_0, ic_set, dtype):
    train_ic_key, test_ic_key = jax.random.split(jax.random.key(ic_set.data_seed))
    train_initial_conditions = draw_initial_conditions(
        z_0,
        k_0,
        ic_set.num_train_trajectories,
        ic_set.train_log_z_variance,
        ic_set.train_log_k_variance,
        train_ic_key,
        dtype,
    )
    test_z_center = z_0 if ic_set.test_z_0 is None else ic_set.test_z_0
    test_k_center = k_0 if ic_set.test_k_0 is None else ic_set.test_k_0
    test_log_z_variance = (
        ic_set.train_log_z_variance
        if ic_set.test_log_z_variance is None
        else ic_set.test_log_z_variance
    )
    test_log_k_variance = (
        ic_set.train_log_k_variance
        if ic_set.test_log_k_variance is None
        else ic_set.test_log_k_variance
    )
    test_initial_conditions = draw_initial_conditions(
        test_z_center,
        test_k_center,
        ic_set.num_test_trajectories,
        test_log_z_variance,
        test_log_k_variance,
        test_ic_key,
        dtype,
    )
    return InitialConditionDraw(
        train_ic_key=train_ic_key,
        test_ic_key=test_ic_key,
        train_initial_conditions=train_initial_conditions,
        test_initial_conditions=test_initial_conditions,
        test_z_center=test_z_center,
        test_k_center=test_k_center,
        test_log_z_variance=test_log_z_variance,
        test_log_k_variance=test_log_k_variance,
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
    "identity": nnx.identity,
}

SDE_SOLVERS = {
    "sra1": SRA1(),
    "milstein": Milstein(),
    "euler_maruyama": EulerMaruyama(),
}


def sample_trajectory_noise(
    solver, key, num_trajectories, n_steps, dt, dtype, state_size
):
    reference_state = jnp.zeros(state_size, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    return jax.vmap(
        lambda trajectory_key: solver.sample_noise(
            reference_state, trajectory_key, n_steps, dt, dtype
        )
    )(jax.random.split(key, num_trajectories))


# Content-addressed policy filenames: the key hashes every input that
# determines the trained artifact, so existence of
# <dir>/<model>_policy_<key>.npz is a cache hit for an identical run.
def cache_key(config):
    canonical = json.dumps(config, sort_keys=True, default=float)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# One .npz per trained module: the nnx.Param block and the non-differentiated
# Variable block as "params/..." and "nondiff/..." flat keys (nnx's supported
# State.to_pure_dict / replace_by_pure_dict conversion, list indices numeric),
# plus a JSON __meta__ block. RNG state is never written: a reload rebuilds
# the structural template from meta and takes every array from the file.
def save_model(path, module, meta):
    def flatten(namespace, node, out):
        if isinstance(node, dict):
            for key, value in node.items():
                flatten((*namespace, str(key)), value, out)
        else:
            out["/".join(namespace)] = np.asarray(node)
        return out

    _, params, _, nondiff = nnx.split(module, nnx.Param, nnx.RngState, ...)
    arrays = flatten(("params",), params.to_pure_dict(), {})
    flatten(("nondiff",), nondiff.to_pure_dict(), arrays)
    meta = {**meta, "dtype": str(next(iter(arrays.values())).dtype)}
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __meta__=np.asarray(json.dumps(meta)), **arrays)


def load_model(path, make_module):
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    dtype = jnp.dtype(meta["dtype"])
    blocks = {"params": {}, "nondiff": {}}
    for name in data.files:
        if name == "__meta__":
            continue
        namespace, _, rest = name.partition("/")
        node = blocks[namespace]
        *parents, leaf = rest.split("/")
        for part in parents:
            node = node.setdefault(int(part) if part.isdigit() else part, {})
        array = data[name]
        node[int(leaf) if leaf.isdigit() else leaf] = jnp.asarray(
            array, dtype=dtype if np.issubdtype(array.dtype, np.inexact) else None
        )
    # An empty replace_by_pure_dict is a silent no-op that would return the
    # template's random initialization as if it were the trained module.
    if not blocks["params"]:
        raise ValueError(f"{path} carries no params/ arrays")
    graphdef, params, rngs, nondiff = nnx.split(
        make_module(meta, dtype), nnx.Param, nnx.RngState, ...
    )
    params.replace_by_pure_dict(blocks["params"])
    if blocks["nondiff"]:
        nondiff.replace_by_pure_dict(blocks["nondiff"])
    return nnx.merge(graphdef, params, rngs, nondiff), meta
