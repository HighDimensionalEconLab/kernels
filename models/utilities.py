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
# the structural template from meta, takes the Param and Variable arrays from
# the file, and keeps the template's RNG state.
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
        if namespace not in blocks:
            raise ValueError(
                f"{path}: unknown npz key namespace {namespace!r}; expected "
                "params/ or nondiff/ (an artifact from the flat pre-namespace "
                "format must be regenerated)"
            )
        node = blocks[namespace]
        *parents, leaf = rest.split("/")
        for part in parents:
            node = node.setdefault(int(part) if part.isdigit() else part, {})
        array = data[name]
        node[int(leaf) if leaf.isdigit() else leaf] = jnp.asarray(
            array, dtype=dtype if np.issubdtype(array.dtype, np.inexact) else None
        )
    graphdef, params, rngs, nondiff = nnx.split(
        make_module(meta, dtype), nnx.Param, nnx.RngState, ...
    )

    def leaf_paths(node, prefix=()):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaf_paths(value, (*prefix, str(key)))
        else:
            yield "/".join(prefix)

    # A missing Param subtree in replace_by_pure_dict is a silent no-op that
    # would keep the template's random initialization for those leaves.
    missing = set(leaf_paths(params.to_pure_dict())) - set(leaf_paths(blocks["params"]))
    if missing:
        raise ValueError(f"{path} is missing params arrays: {sorted(missing)}")
    params.replace_by_pure_dict(blocks["params"])
    if blocks["nondiff"]:
        nondiff.replace_by_pure_dict(blocks["nondiff"])
    return nnx.merge(graphdef, params, rngs, nondiff), meta
