import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import struct
from jax.tree_util import Partial
from nlls_gram import LMStatus
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
