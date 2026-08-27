import dataclasses
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.tree_util import Partial
from nlls_gram import DrawNNXModule, LMStatus
from tinydiffeq import SRA1, EulerMaruyama, Milstein


# Base for the structural-parameter pytrees: create casts every field to one
# dtype; to_dict returns plain Python scalars/lists for the results record.
class ParameterPyTree(struct.PyTreeNode):
    @classmethod
    def create(cls, *, dtype, **kwargs):
        return cls(
            **{
                key: jnp.asarray(value, dtype=dtype)
                for key, value in kwargs.items()
                if value is not None
            }
        )

    def to_dict(self):
        values = jax.device_get(
            {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if getattr(self, f.name) is not None
            }
        )
        return {k: v.item() if v.ndim == 0 else v.tolist() for k, v in values.items()}


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


# Levenberg-Marquardt settings shared across solvers -- exactly the
# constructor and solve() kwargs; each solver pins its tuned values inline in
# its lm_set signature default and threads only the fields it uses.
@dataclass(frozen=True)
class OptimizerSettings:
    init_damping: float = 1e-5
    damping_decrease: float = 0.35
    damping_increase: float = 8.0
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.8
    max_steps: int = 100
    atol: float = 1e-9
    gtol: float = 1e-9
    num_attempts: int = 5  # MultiStart num_starts; multi-start drivers only


# RidgeLevenbergMarquardt extras: the ridge weight, optional ridge
# continuation (ridge_init=None solves at the fixed ridge), and the linear
# solver for the whitened subproblem.
@dataclass(frozen=True)
class RidgeSettings:
    ridge: float = 1e-10
    ridge_init: float | None = None
    ridge_decrease: float = 0.01
    ridge_grad_rtol: float = 1e-2
    linear_solver: str = "auto"  # "auto" | "cholesky" | "qr" | "cg"


# Matrix-free CG configuration, read only when linear_solver == "cg".
@dataclass(frozen=True)
class CGSettings:
    tol: float = 1e-10
    maxiter: int = 40
    damping_floor: float = 1e-10


# Uno/IPOPT options, applied via solver.set_option gated on `is not None`.
@dataclass(frozen=True)
class UnoOptimizerSettings:
    preset: str = "ipopt"
    max_iterations: int | None = None
    time_limit: float | None = None
    primal_tolerance: float | None = None
    dual_tolerance: float | None = None


ACCEPTED_UNO_SOLUTION_STATUSES = frozenset(
    {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
)


# Sequence-space solver args; the recursive drivers keep bespoke ones.
@struct.dataclass
class KernelResidualArgs:
    K: jax.Array
    K_tilde: jax.Array
    batch: dict | None = None
    unravel: Partial | None = None


@struct.dataclass
class NNXResidualArgs:
    batch: dict
    graphdef: nnx.GraphDef
    nondiff: nnx.State


# Multi-start draw splitting redrawn modules with the solver's diff filter
# rather than bare nnx.Param: a Param the filter freezes out of theta must
# not reappear in the drawn state. The filter joins the value-hash spec.
class DrawNNXModuleFiltered(DrawNNXModule):
    def __init__(self, diff_filter, module_cls, *args, **kwargs):
        super().__init__(module_cls, *args, **kwargs)
        self.diff_filter = diff_filter

    def __call__(self, key, x_old, args_old):
        module = self.module_cls(*self.args, rngs=nnx.Rngs(key), **dict(self.kwargs))
        _, theta, _ = nnx.split(module, self.diff_filter, ...)
        return theta, args_old

    def __hash__(self):
        return hash((type(self), self.diff_filter, super().__hash__()))

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and self.diff_filter == other.diff_filter
            and super().__eq__(other)
        )


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
    return train_data


# Forward-mode counterpart of jax.value_and_grad for scalar-input functions:
# one jvp pass with a unit tangent returns (f(x), df/dx), for any output
# pytree (reverse-mode value_and_grad requires a scalar output).
def value_and_grad_fwd(f):
    def value_and_grad(x):
        return jax.jvp(f, (x,), (jnp.ones_like(x),))

    return value_and_grad


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
