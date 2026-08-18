from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import DrawNNXModule, LevenbergMarquardt, LMAction, LMStatus, MultiStart
from tinydiffeq import RK4, ConstantStepSize, IController, SaveAt, Tsit5, solve_ode

from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.utilities import (
    ACTIVATIONS,
    STATUS_TO_REASON,
    OptimizerSettings,
    SolveSpec,
    unit_softplus,
)
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)

DOMAIN_EPS = 1e-8


@struct.dataclass
class StructuralParameters:
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    g: jax.Array
    z_0: jax.Array
    k_0: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class DataSettings:
    # Training collocation on the integrator's own states: n points from
    # n - 1 steps including the initial condition, so nothing is
    # interpolated. ode_solver "rk4" spends the budget on a fixed grid;
    # "tsit5" spends the same budget adaptively and enforces residuals on
    # the accepted steps only, concentrating points where the transition is
    # fast. rtol/atol are the adaptive error thresholds and initial_dt the
    # adaptive starting step (None: the fixed grid's train_T / (n - 1));
    # all three are ignored under rk4. Test/tvc rollouts stay fixed-step.
    # Thresholds much below 1e-6 demand more steps than a ~32-step budget
    # allows while the interim policy is still wrong, so training dies on
    # inf residuals instead of adapting.
    train_points: int = 65
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = CONSTANTS.train_T
    train_tvc_T: float = CONSTANTS.transversality_T
    test_T: float = CONSTANTS.test_T
    policy_grid_size: int = 100
    benchmark_T: float = CONSTANTS.benchmark_T
    benchmark_points: int = CONSTANTS.benchmark_points
    ode_solver: str = "rk4"
    rtol: float = 1e-6
    atol: float = 1e-8
    initial_dt: float | None = None


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


# Training ICs and keys live in args (the epoch callback may rewrite args
# only), so the implicit derivative is conditional on the carried batch.
@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    initial_log_states: jax.Array
    epoch_key: jax.Array
    restart_key: jax.Array
    epoch: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    data_set: DataSettings = struct.field(pytree_node=False)


@dataclass(frozen=True)
class StoppingCriteria:
    # max_steps is the LM-update budget per epoch; training ICs are redrawn at
    # each interior epoch boundary, so num_epochs = 1 never resamples.
    max_steps: int = 250
    train_loss_threshold: float = 1e-14
    # Multi-start restarts with fresh parameter draws; later attempts run only
    # when an earlier one fails, so converging solves pay nothing.
    num_attempts: int = 5
    log_tvc_threshold: float = 0.0
    num_epochs: int = 1
    epoch_early_stopping: bool = True


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


# One .npz per trained policy: the nnx.Param state through nnx's supported
# State.to_pure_dict / replace_by_pure_dict conversion (keys joined with "/",
# list indices numeric), plus a JSON block with the architecture settings,
# scales, dtype, and the structural scalars, so a load is rollout-ready
# without the training driver (e.g. as the trajectory generator for a kernel
# design).
def save_investment_policy(path, policy, settings, *, structural=None, extra=None):
    def flatten(prefix, node, out):
        if isinstance(node, dict):
            for key, value in node.items():
                flatten((*prefix, str(key)), value, out)
        else:
            out["/".join(prefix)] = np.asarray(node)
        return out

    arrays = flatten((), nnx.state(policy, nnx.Param).to_pure_dict(), {})
    meta = {
        "settings": asdict(settings),
        "log_z_scale": float(policy.log_z_scale[...]),
        "log_k_scale": float(policy.log_k_scale[...]),
        "k_scale": float(policy.k_scale[...]),
        "dtype": str(next(iter(arrays.values())).dtype),
    }
    if structural is not None:
        meta["structural"] = {
            name: float(np.asarray(value))
            for name, value in structural.items()
            if np.asarray(value).ndim == 0
        }
    if extra:
        meta.update(extra)
    np.savez(path, __meta__=np.asarray(json.dumps(meta)), **arrays)


def load_investment_policy(path):
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    settings = InvestmentPolicySettings(**meta["settings"])
    dtype = jnp.dtype(meta["dtype"])
    module = InvestmentPolicyNN(
        settings,
        z_scale=math.exp(meta["log_z_scale"]),
        k_scale=meta["k_scale"],
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )
    graphdef, theta, nondiff = nnx.split(module, nnx.Param, ...)
    nested = {}
    for name in data.files:
        if name == "__meta__":
            continue
        node = nested
        *parents, leaf = name.split("/")
        for part in parents:
            node = node.setdefault(int(part) if part.isdigit() else part, {})
        node[int(leaf) if leaf.isdigit() else leaf] = jnp.asarray(
            data[name], dtype=dtype
        )
    theta.replace_by_pure_dict(nested)
    return nnx.merge(graphdef, theta, nondiff), meta


# Content-addressed policy filenames: the key hashes every input that
# determines the trained artifact, so existence of
# <dir>/<model>_policy_<key>.npz is a cache hit for an identical run.
def policy_cache_key(config):
    canonical = json.dumps(config, sort_keys=True, default=float)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def meta_lookup(meta, dotted):
    node = meta
    for part in dotted.split("."):
        node = node[part]
    return node


# Scan saved policies and keep those whose meta matches every criterion;
# criteria keys are dotted paths into the meta, e.g.
# {"structural.sigma_crra": 2.0, "dtype": "float64"}.
def find_saved_policies(directory, criteria=None):
    matches = []
    for path in sorted(pathlib.Path(directory).glob("*_policy_*.npz")):
        meta = json.loads(str(np.load(path)["__meta__"]))
        try:
            matched = all(
                meta_lookup(meta, key) == value
                for key, value in (criteria or {}).items()
            )
        except KeyError:
            continue
        if matched:
            matches.append((path, meta))
    return matches


# Hicks-neutral technology: z is the TFP level, so Y = z k^a. Balanced growth
# at per-capita rate g needs the TFP trend (1 - a) g; the labor-augmenting
# index z^{1/(1-a)} grows at g and is the detrending scale for quantities.
def production(z, k, p):
    return z * k**p.a


def mpk(z, k, p):
    return p.a * z * k ** (p.a - 1.0)


def labor_augmenting_scale(z, p):
    return z ** (1.0 / (1.0 - p.a))


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return production(z, k, p) - policy(log_state, p)


def log_state_drift(policy, log_state, p):
    _, log_k = log_state
    k = jnp.exp(log_k)
    investment = policy(log_state, p)
    return jnp.stack([(1.0 - p.a) * p.g, investment / k - p.delta])


def euler_residuals(policy, log_states, p):
    def residual_scalar(log_state):
        state_drift = log_state_drift(policy, log_state, p)
        log_c, dlog_c_dt = jax.jvp(
            lambda state: jnp.log(consumption(policy, state, p)),
            (log_state,),
            (state_drift,),
        )
        z, k = jnp.exp(log_state)
        # The tangent c_dot / c alone is finite for negative consumption, so
        # the zero-scaled primal log c keeps the residual NaN outside the
        # model's domain (c <= 0) instead of quietly extending the Euler
        # equation there. It contributes exactly zero on the feasible domain.
        domain_guard = 0.0 * log_c
        return (
            dlog_c_dt
            + domain_guard
            - (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
        )

    return jax.vmap(residual_scalar)(log_states)


def draw_initial_conditions(
    z_0, k_0, num_trajectories, log_z_variance, log_k_variance, key, dtype
):
    if z_0 <= 0.0 or k_0 <= 0.0:
        raise ValueError("z_0 and k_0 must be positive")
    if num_trajectories < 1:
        raise ValueError("trajectory counts must be positive")
    if log_z_variance < 0.0 or log_k_variance < 0.0:
        raise ValueError("initial-condition log variances must be nonnegative")
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
    # Trace-safe standardized draws, ADDITIVE in the caller's coordinate:
    # center + scales * standardized normals, dimension-generic, so a zero
    # scale pins its axis at the center whatever the center's value. Its
    # neighbour draw_initial_conditions is the multiplicative (log-scale)
    # variant -- do not merge them: additive spreads are what keep a level
    # axis (e.g. net foreign assets) non-degenerate at a zero center.
    if num_trajectories == 1:
        return center[None, :]
    draws = jax.random.normal(key, (num_trajectories, center.size), dtype=center.dtype)
    centered = draws - jnp.mean(draws, axis=0, keepdims=True)
    standardized = centered / jnp.sqrt(jnp.mean(centered**2, axis=0, keepdims=True))
    return center[None, :] + scales[None, :] * standardized


# At each interior epoch boundary: redraw the training ICs, stop on a
# nonfinite or converged fresh-batch loss, else continue with damping reset.
@dataclass(frozen=True)
class EpochCallback:
    num_train_trajectories: int
    log_z_0: float
    log_k_0: float
    log_z_scale: float
    log_k_scale: float
    epoch_early_stopping: bool

    def next_epoch(self, ctx):
        epoch_key, draw_key = jax.random.split(ctx.args.epoch_key)
        dtype = ctx.args.initial_log_states.dtype
        fresh = ctx.args.replace(
            initial_log_states=standardized_initial_states(
                draw_key,
                jnp.asarray([self.log_z_0, self.log_k_0], dtype),
                jnp.asarray([self.log_z_scale, self.log_k_scale], dtype),
                self.num_train_trajectories,
            ),
            epoch_key=epoch_key,
            epoch=ctx.args.epoch + 1,
        )
        r = residual_fn(ctx.x, fresh, ctx.p)
        loss = jnp.sum(r * r)
        nonfinite = ~jnp.isfinite(loss)
        if self.epoch_early_stopping:
            converged = loss < fresh.train_loss_threshold
        else:
            converged = jnp.asarray(False)
        status = jnp.where(
            nonfinite,
            jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
                jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
            ),
        )
        return nonfinite | converged, status, fresh, ctx.initial_lm_state.damping

    def mid_epoch(self, ctx):
        return (
            jnp.asarray(False),
            jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
            ctx.args,
            ctx.lm_state.damping,
        )

    def __call__(self, ctx):
        boundary = (ctx.step % ctx.args.epoch_steps == 0) & (
            ctx.step < ctx.args.total_steps
        )
        stop, status, args, damping = jax.lax.cond(
            boundary, self.next_epoch, self.mid_epoch, ctx
        )
        return LMAction(
            stop=stop,
            status=status,
            args=args,
            lm_state=replace(ctx.lm_state, damping=damping),
        )


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


def positive_policy_grid(grid_lo, grid_hi, size):
    if grid_lo <= 0.0 or grid_hi < grid_lo:
        raise ValueError("policy-grid bounds must satisfy 0 < grid_lo <= grid_hi")
    log_lo = math.log(grid_lo)
    log_hi = math.log(grid_hi)
    log_margin = 0.05 * (log_hi - log_lo) + 1e-6
    return np.exp(np.linspace(log_lo - log_margin, log_hi + log_margin, size))


def log_state_drift_field(log_state, _, args, p):
    policy, structural = p
    return log_state_drift(policy, log_state, structural)


def batched_rollout(policy, initial_log_states, T, budget, dt_0, p):
    def solve_one(initial_log_state):
        return solve_ode(
            log_state_drift_field,
            RK4(),
            0.0,
            T,
            initial_log_state,
            dt_0=dt_0,
            controller=ConstantStepSize(),
            max_steps=budget,
            save_at=SaveAt(steps=True),
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_log_states)


@partial(jax.jit, static_argnames=("data_set",))
def collocation_rollout(policy, initial_log_states, p, data_set):
    budget = data_set.train_points - 1
    if data_set.ode_solver == "rk4":
        return batched_rollout(
            policy,
            initial_log_states,
            data_set.train_T,
            budget,
            data_set.train_T / budget,
            p,
        )
    dt_0 = (
        data_set.train_T / budget
        if data_set.initial_dt is None
        else data_set.initial_dt
    )

    def solve_one(initial_log_state):
        return solve_ode(
            log_state_drift_field,
            Tsit5(),
            0.0,
            data_set.train_T,
            initial_log_state,
            dt_0=dt_0,
            controller=IController(rtol=data_set.rtol, atol=data_set.atol),
            max_steps=budget,
            save_at=SaveAt(steps=True),
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_log_states)


@partial(jax.jit, static_argnames=("T", "budget", "dt_0"))
def rollout_steps(policy, initial_log_states, T, budget, dt_0, p):
    return batched_rollout(policy, initial_log_states, T, budget, dt_0, p)


def collocation_residuals(policy, initial_log_states, p, data_set):
    solution = collocation_rollout(policy, initial_log_states, p, data_set)
    log_states = solution.xs
    euler = euler_residuals(policy, log_states.reshape(-1, 2), p).reshape(
        log_states.shape[:2]
    )
    # Mean squared residual, so the loss threshold means the same thing at any
    # number of trajectories or collocation points. An adaptive rollout only
    # fills the accepted prefix of the step buffer, so both the residuals and
    # the count are restricted to that prefix.
    if data_set.ode_solver == "rk4":
        count = jnp.asarray(euler.size, dtype=euler.dtype)
        normalized = euler / jnp.sqrt(count)
    else:
        active = solution.accepted
        count = jnp.maximum(jnp.sum(active), 1).astype(euler.dtype)
        normalized = jnp.where(active, euler, 0.0) / jnp.sqrt(count)
    # A nonfinite trajectory did not cover the requested economic horizon. Make
    # that a loud optimizer failure instead of fitting a prefix.
    return jnp.where(
        solution.ok[:, None],
        normalized,
        jnp.full_like(euler, jnp.inf),
    )


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.nondiff)
    return collocation_residuals(
        policy, args.initial_log_states, p, args.data_set
    ).reshape(-1)


def log_tvc_residual(policy, initial_log_states, T_start, T_end, p, data_set):
    solution = rollout_steps(
        policy,
        initial_log_states,
        T_end,
        data_set.tvc_steps,
        T_end / data_set.tvc_steps,
        p,
    )
    valid = (solution.ts >= T_start) & solution.ok[:, None]
    flat_states = solution.xs.reshape(-1, 2)
    c = jax.vmap(lambda state: consumption(policy, state, p))(flat_states).reshape(
        solution.ts.shape
    )
    _, k = jnp.moveaxis(jnp.exp(solution.xs), -1, 0)
    log_tvc = -p.rho_hat * solution.ts - p.sigma_crra * jnp.log(c) + jnp.log(k)
    c_0 = jax.vmap(lambda state: consumption(policy, state, p))(initial_log_states)
    k_0 = jnp.exp(initial_log_states[:, 1])
    initial_log_tvc = -p.sigma_crra * jnp.log(c_0) + jnp.log(k_0)
    normalized_log_tvc = log_tvc - initial_log_tvc[:, None]
    # Alongside the windowed max, the per-trajectory certificate at the single
    # terminal date T_end; a failed rollout is a loud +inf.
    terminal_log_tvc = jnp.where(solution.ok, normalized_log_tvc[:, -1], jnp.inf)
    return (
        jnp.max(jnp.where(valid, normalized_log_tvc, -jnp.inf)),
        terminal_log_tvc,
        solution.ok,
    )


@lru_cache(maxsize=None)
def cached_benchmark_trajectory(
    a,
    delta,
    rho_hat,
    sigma_crra,
    g,
    z_initial,
    k_initial,
    times,
    benchmark_T,
    benchmark_points,
):
    benchmark_grid = np.linspace(0.0, benchmark_T, benchmark_points)
    benchmark = neoclassical_growth_benchmark(
        a,
        delta,
        rho_hat,
        sigma_crra,
        k_initial,
        benchmark_grid,
        z_0=z_initial,
        g=g,
    )
    return benchmark(np.asarray(times))


def benchmark_trajectories(
    a, delta, rho_hat, sigma_crra, g, initial_conditions, times, data_set
):
    k_paths = []
    c_paths = []
    for z_initial, k_initial in np.asarray(initial_conditions):
        k_path, c_path = cached_benchmark_trajectory(
            a,
            delta,
            rho_hat,
            sigma_crra,
            g,
            float(z_initial),
            float(k_initial),
            tuple(float(t) for t in np.asarray(times)),
            data_set.benchmark_T,
            data_set.benchmark_points,
        )
        k_paths.append(k_path)
        c_paths.append(c_path)
    times = np.asarray(times)
    z_paths = np.asarray(initial_conditions)[:, :1] * np.exp(
        (1.0 - a) * g * times[None, :]
    )
    return z_paths, np.stack(k_paths), np.stack(c_paths)


def benchmark_policy_grid(a, delta, rho_hat, sigma_crra, g, kappa_grid, data_set):
    # Interpolate c/z from benchmark transitions started at the low, middle,
    # and high grid points. The long transitions overlap in the interior and
    # asymptote toward balanced growth from both sides, so the interpolation
    # support is dense without ever computing a steady state. Sampling stops
    # at the training horizon to stay out of the terminal boundary layer of
    # the truncated BVP.
    policy_sample_T = max(data_set.train_T, data_set.test_T)
    benchmark_times = np.linspace(0.0, policy_sample_T, data_set.benchmark_points)
    kappa_samples = []
    c_tilde_samples = []
    kappa_starts = [kappa_grid[0], kappa_grid[len(kappa_grid) // 2], kappa_grid[-1]]
    for kappa_initial in np.unique(kappa_starts):
        z_path, k_path, c_path = benchmark_trajectories(
            a,
            delta,
            rho_hat,
            sigma_crra,
            g,
            np.asarray([[1.0, kappa_initial]]),
            benchmark_times,
            data_set,
        )
        scale_path = z_path ** (1.0 / (1.0 - a))
        kappa_samples.append((k_path / scale_path).reshape(-1))
        c_tilde_samples.append((c_path / scale_path).reshape(-1))

    kappa_path = np.concatenate(kappa_samples)
    c_tilde_path = np.concatenate(c_tilde_samples)
    order = np.argsort(kappa_path)
    kappa_sorted = kappa_path[order]
    c_tilde_sorted = c_tilde_path[order]
    kappa_unique, unique_indices = np.unique(kappa_sorted, return_index=True)
    return np.interp(kappa_grid, kappa_unique, c_tilde_sorted[unique_indices])


def add_benchmark_path_diagnostics(
    test_df,
    test_values,
    test_initial_conditions,
    z_benchmark,
    k_benchmark,
    c_benchmark,
    *,
    a,
    delta,
    g,
):
    k_error = (np.asarray(test_values["k"]) - k_benchmark) / np.abs(k_benchmark).clip(
        min=DOMAIN_EPS
    )
    c_error = (np.asarray(test_values["c"]) - c_benchmark) / np.abs(c_benchmark).clip(
        min=DOMAIN_EPS
    )
    scale_benchmark = z_benchmark ** (1.0 / (1.0 - a))
    kappa_benchmark = k_benchmark / scale_benchmark
    c_tilde_benchmark = c_benchmark / scale_benchmark
    i_benchmark = z_benchmark * k_benchmark**a - c_benchmark
    dk_benchmark = i_benchmark - delta * k_benchmark
    dkappa_benchmark = i_benchmark / scale_benchmark - (delta + g) * kappa_benchmark

    columns = {
        "z_benchmark": z_benchmark,
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "kappa_benchmark": kappa_benchmark,
        "c_tilde_benchmark": c_tilde_benchmark,
        "i_benchmark": i_benchmark,
        "dk_benchmark": dk_benchmark,
        "dkappa_benchmark": dkappa_benchmark,
        "k_rel_error": k_error,
        "c_rel_error": c_error,
        "k_abs_rel_error": np.abs(k_error),
        "c_abs_rel_error": np.abs(c_error),
    }
    for name, value in columns.items():
        test_df[name] = np.asarray(value).reshape(-1)

    initial = np.asarray(test_initial_conditions)
    test_summary_df = pd.DataFrame(
        {
            "trajectory": np.arange(initial.shape[0]),
            "z_0": initial[:, 0],
            "k_0": initial[:, 1],
            "kappa_0": initial[:, 1] / initial[:, 0] ** (1.0 / (1.0 - a)),
            "mean_k_abs_rel_error": np.mean(np.abs(k_error), axis=1),
            "median_k_abs_rel_error": np.median(np.abs(k_error), axis=1),
            "mean_c_abs_rel_error": np.mean(np.abs(c_error), axis=1),
            "median_c_abs_rel_error": np.median(np.abs(c_error), axis=1),
        }
    )
    return k_error, c_error, test_summary_df


# nnx.jit so one compiled evaluation serves both the nnx policy and the
# kernel policy pytree.
@nnx.jit
def trajectory_values(policy, log_states, p):
    shape = log_states.shape[:-1]
    flat_states = log_states.reshape(-1, 2)
    investment = jax.vmap(lambda state: policy(state, p))(flat_states).reshape(shape)
    c = jax.vmap(lambda state: consumption(policy, state, p))(flat_states).reshape(
        shape
    )
    euler = euler_residuals(policy, flat_states, p).reshape(shape)
    z, k = jnp.moveaxis(jnp.exp(log_states), -1, 0)
    scale = labor_augmenting_scale(z, p)
    kappa = k / scale
    dk = investment - p.delta * k
    dz = (1.0 - p.a) * p.g * z
    dkappa = investment / scale - (p.delta + p.g) * kappa
    return {
        "z": z,
        "k": k,
        "kappa": kappa,
        "i": investment,
        "c": c,
        "dz": dz,
        "dk": dk,
        "dkappa": dkappa,
        "euler_residual": euler,
    }


SCALE_INVARIANCE_LAMBDAS = (math.exp(-0.25), 1.0, math.exp(0.25))


# i/k at matched kappa under common scales (z, k) -> (lam^{1-a} z, lam k),
# which hold kappa = k / z^{1/(1-a)} fixed; the dispersion across columns is
# zero iff the policy is homothetic.
def scale_invariance_columns(policy, kappa_grid, p, dtype, suffix):
    log_lambdas = jnp.log(jnp.asarray(SCALE_INVARIANCE_LAMBDAS, dtype=dtype))
    log_kappa = jnp.log(jnp.asarray(kappa_grid, dtype=dtype))
    states = jnp.stack(
        [
            jnp.broadcast_to(
                (1.0 - p.a) * log_lambdas[:, None],
                (len(log_lambdas), log_kappa.size),
            ),
            log_lambdas[:, None] + log_kappa[None, :],
        ],
        axis=-1,
    )
    values = trajectory_values(policy, states, p)
    i_over_k = np.asarray(values["i"] / values["k"])
    return {
        f"i_over_k_scale_{tag}_{suffix}": i_over_k[row]
        for row, tag in enumerate(("lo", "mid", "hi"))
    }


def trajectory_frame(values, times, initial_conditions):
    num_trajectories, num_times = values["z"].shape
    initial = np.asarray(initial_conditions)
    trajectory = np.repeat(np.arange(num_trajectories), num_times)
    frame = pd.DataFrame(
        {
            "trajectory": trajectory,
            "t": np.asarray(times).reshape(-1),
            "z_0": initial[trajectory, 0],
            "k_0": initial[trajectory, 1],
            "kappa_0": np.asarray(values["kappa"])[trajectory, 0],
        }
    )
    for name, value in values.items():
        frame[name] = np.asarray(value).reshape(-1)
    return frame


def collocation_step_diagnostics(times, accepted=None):
    # accepted (adaptive rollouts) masks the valid prefix of each step
    # buffer; None means every saved step is a collocation state.
    times = np.asarray(times)
    if accepted is None:
        accepted = np.ones(times.shape, dtype=bool)
    else:
        accepted = np.asarray(accepted)
    step_sizes = np.concatenate(
        [np.diff(row_times[row_mask]) for row_times, row_mask in zip(times, accepted)]
    )
    return {
        "num_collocation_states": int(accepted.sum()),
        "collocation_states_per_trajectory": int(accepted.sum(axis=1).max()),
        "collocation_dt": float(step_sizes.max()) if step_sizes.size else np.nan,
        "collocation_dt_min": float(step_sizes.min()) if step_sizes.size else np.nan,
    }


# The shared NEOCLASSICAL_GROWTH_CONSTANTS (utilities.py) maps the Aruoba,
# Fernandez-Villaverde & Rubio-Ramirez (2006 JEDC) benchmark values; the
# stochastic variant solves this same model with stochastic TFP.
def neoclassical_growth_nn_recursive(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    hc_set: InvestmentPolicySettings = InvestmentPolicySettings(),
    opt_set: OptimizerSettings = OptimizerSettings(
        init_damping=3.26e-5,
        damping_decrease=0.482,
        damping_increase=2.71,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.541,
    ),
    stop_set: StoppingCriteria = StoppingCriteria(),
    # Directory for content-addressed policy artifacts (e.g. ".results"): an
    # existing artifact for this exact configuration is loaded instead of
    # training, and a fresh solve is saved there. save_policy writes one
    # explicit extra path.
    policy_cache: str | None = None,
    save_policy: str | None = None,
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    if not 0.0 < a < 1.0:
        raise ValueError("a must lie strictly between zero and one")
    if sigma_crra <= 0.0:
        raise ValueError("sigma_crra must be positive")
    # Well-posedness of the planner objective, not steady-state structure:
    # finiteness of discounted utility needs rho_hat + (sigma - 1) g > 0, the
    # discounted-capital certificate needs g < rho_hat, and detrended capital
    # needs positive effective depreciation.
    if g >= rho_hat:
        raise ValueError("g must be less than rho_hat")
    if rho_hat + (sigma_crra - 1.0) * g <= 0.0:
        raise ValueError("rho_hat + (sigma_crra - 1) * g must be positive")
    if delta + g <= 0.0:
        raise ValueError("delta + g must be positive")
    if data_set.train_T <= 0.0:
        raise ValueError("train_T must be positive")
    if data_set.train_points < 2:
        raise ValueError("train_points must be at least two")
    if data_set.ode_solver not in {"rk4", "tsit5"}:
        raise ValueError("ode_solver must be 'rk4' or 'tsit5'")
    if not (math.isfinite(data_set.rtol) and data_set.rtol > 0.0):
        raise ValueError("rtol must be positive")
    if not (math.isfinite(data_set.atol) and data_set.atol > 0.0):
        raise ValueError("atol must be positive")
    if data_set.initial_dt is not None and not (
        math.isfinite(data_set.initial_dt) and data_set.initial_dt > 0.0
    ):
        raise ValueError("initial_dt must be positive when set")
    if not (
        math.isfinite(stop_set.train_loss_threshold)
        and stop_set.train_loss_threshold > 0.0
    ):
        raise ValueError("train_loss_threshold must be positive")
    if data_set.tvc_steps < 1:
        raise ValueError("tvc_steps must be positive")
    if data_set.test_points < 2:
        raise ValueError("test_points must be at least two")
    if data_set.train_tvc_T <= data_set.train_T:
        raise ValueError("train_tvc_T must exceed train_T")
    n_test_steps = data_set.test_points - 1
    test_T = data_set.test_T
    if data_set.benchmark_T < max(data_set.train_T, test_T):
        raise ValueError("benchmark_T must cover both train_T and test_T")
    # Without a positive variance every epoch redraw returns the same batch.
    if stop_set.num_epochs > 1 and (
        ic_set.train_log_z_variance <= 0.0 and ic_set.train_log_k_variance <= 0.0
    ):
        raise ValueError("epoch resampling requires a positive training log variance")

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
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        g=g,
        z_0=train_initial_conditions[:, 0],
        k_0=train_initial_conditions[:, 1],
    )

    fit_start = time.perf_counter()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    graphdef, _, nondiff = nnx.split(
        InvestmentPolicyNN(
            hc_set, z_scale=z_0, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(0)
        ),
        nnx.Param,
        ...,
    )
    # The initial training draw consumes train_ic_key itself; the epoch and
    # restart streams chain from independent fold-ins of the same key.
    residual_args = ResidualArgs(
        graphdef=graphdef,
        nondiff=nondiff,
        initial_log_states=jnp.log(train_initial_conditions),
        epoch_key=jax.random.fold_in(train_ic_key, 1),
        restart_key=jax.random.fold_in(train_ic_key, 2),
        epoch=jnp.asarray(0, dtype=jnp.int32),
        train_loss_threshold=jnp.asarray(stop_set.train_loss_threshold, dtype=dtype),
        epoch_steps=jnp.asarray(stop_set.max_steps, dtype=jnp.int32),
        total_steps=jnp.asarray(
            stop_set.num_epochs * stop_set.max_steps, dtype=jnp.int32
        ),
        data_set=data_set,
    )
    # Two-part artifact key <problem>-<run>: the problem hash covers the deep
    # structural parameters, dtype, and the training initial-condition design
    # shared by any solver of the same problem, so a consumer can glob
    # *_policy_<problem>-*.npz for a reference solution; the run hash covers
    # the training minutiae. Test/benchmark settings enter neither, so
    # changing them still reuses a cached policy.
    problem_config = {
        "technology": "hicks",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "use_float64": use_float64,
        "train_ic": {
            "num_train_trajectories": ic_set.num_train_trajectories,
            "train_log_z_variance": ic_set.train_log_z_variance,
            "train_log_k_variance": ic_set.train_log_k_variance,
        },
    }
    run_config = {
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
        "ode_solver": data_set.ode_solver,
        "rtol": data_set.rtol,
        "atol": data_set.atol,
        "initial_dt": data_set.initial_dt,
        "hc_set": asdict(hc_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
    }
    problem_key = policy_cache_key(problem_config)
    run_key = policy_cache_key(run_config)
    cache_key = f"{problem_key}-{run_key}"
    cache_path = (
        None
        if policy_cache is None
        else pathlib.Path(policy_cache)
        / f"neoclassical_growth_nn_recursive_policy_{cache_key}.npz"
    )
    cache_hit = cache_path is not None and cache_path.exists()
    if cache_hit:
        policy, cached_meta = load_investment_policy(cache_path)
        theta_final, solve_args = nnx.split(policy, nnx.Param, ...)[1], residual_args
        steps = 0
        epochs_run = 0
        attempt = 0
        stopping_reason = "cached"
        final_damping = float("nan")
        solver_accepted = bool(cached_meta["solution_accepted"])
        train_time = 0.0
        total_fit_time = time.perf_counter() - fit_start
        setup_time = total_fit_time
        r = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(r * r))
        if verbose:
            print(f"loaded {cache_path}, train_loss={train_loss:.3e}")
    else:
        solver = LevenbergMarquardt(
            residual_fn,
            init_damping=opt_set.init_damping,
            damping_decrease=opt_set.damping_decrease,
            damping_increase=opt_set.damping_increase,
            geodesic_acceleration=opt_set.geodesic_acceleration,
            geodesic_acceptance_ratio=opt_set.geodesic_acceptance_ratio,
        )
        _, theta_0, _ = nnx.split(
            InvestmentPolicyNN(
                hc_set, z_scale=z_0, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(init_key)
            ),
            nnx.Param,
            ...,
        )
        callback = EpochCallback(
            num_train_trajectories=ic_set.num_train_trajectories,
            log_z_0=math.log(z_0),
            log_k_0=math.log(k_0),
            log_z_scale=math.sqrt(ic_set.train_log_z_variance),
            log_k_scale=math.sqrt(ic_set.train_log_k_variance),
            epoch_early_stopping=stop_set.epoch_early_stopping,
        )
        draw = DrawInvestmentPolicy(hc_set, z_scale=z_0, k_scale=k_0, dtype=dtype)
        # The per-step tolerance is sound only while the training batch never
        # changes; across epochs, convergence is judged at the boundaries
        # instead.
        atol = (
            math.sqrt(stop_set.train_loss_threshold)
            if stop_set.num_epochs == 1
            else 0.0
        )
        multi_start = MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=draw,
        )
        solve_options = dict(
            max_steps=stop_set.num_epochs * stop_set.max_steps,
            atol=atol,
            callback=callback,
            max_steps_is_success=not stop_set.epoch_early_stopping,
        )

        start_time = time.perf_counter()
        result = solver.solve(
            theta_0,
            residual_args,
            p=p,
            multi_start=multi_start,
            **solve_options,
        )
        jax.block_until_ready(result)
        train_time = time.perf_counter() - start_time
        total_fit_time = time.perf_counter() - fit_start
        setup_time = start_time - fit_start
        steps = int(result.steps)
        # info.loss can describe the pre-swap batch when an epoch boundary
        # stopped the solve, so the loss is recomputed at the returned
        # (x, args).
        theta_final, solve_args = result.x, result.args
        r = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(r * r))
        epochs_run = -(-steps // stop_set.max_steps)
        stopping_reason = STATUS_TO_REASON.get(int(result.status), "max_steps")
        attempt = int(result.multi_start.attempt) + 1
        final_damping = float(result.lm_state.damping)
        solver_accepted = bool(result.multi_start.accepted)
        if verbose:
            print(
                f"attempt {attempt}: {stopping_reason} after {steps} steps "
                f"(epoch {epochs_run}/{stop_set.num_epochs}), "
                f"train_loss={train_loss:.3e} ({train_time:.2f}s)"
            )
        policy = nnx.merge(graphdef, theta_final, nondiff)

    # Measured, not requested: a float64 leak anywhere upstream shows up here.
    residual_dtype = str(r.dtype)

    # Training diagnostics use the batch the solution was returned on.
    train_batch_log_states = solve_args.initial_log_states
    train_batch_initial_conditions = jnp.exp(train_batch_log_states)
    train_solution = collocation_rollout(policy, train_batch_log_states, p, data_set)
    train_log_states = train_solution.xs
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_solution.ts,
        train_batch_initial_conditions,
    )
    # Adaptive rollouts pad the step buffer past the accepted prefix; those
    # rows are not collocation states and leave the frame here.
    train_accepted = (
        None if data_set.ode_solver == "rk4" else np.asarray(train_solution.accepted)
    )
    if train_accepted is not None:
        train_df = train_df[train_accepted.reshape(-1)].reset_index(drop=True)
    train_df["collocation_step_index"] = train_df.groupby("trajectory").cumcount()
    collocation_diagnostics = collocation_step_diagnostics(
        train_solution.ts, train_accepted
    )

    test_initial_log_states = jnp.log(test_initial_conditions)
    test_solution = rollout_steps(
        policy,
        test_initial_log_states,
        test_T,
        n_test_steps,
        test_T / n_test_steps,
        p,
    )
    jax.block_until_ready(test_solution)
    test_log_states = jnp.where(
        test_solution.ok[:, None, None], test_solution.xs, jnp.nan
    )
    test_values = trajectory_values(policy, test_log_states, p)
    test_path_ok = bool(
        jnp.all(test_solution.ok)
        and jnp.all(jnp.isfinite(test_log_states))
        and jnp.all(jnp.isfinite(test_values["c"]))
        and jnp.min(test_values["c"]) > 0.0
    )
    test_df = trajectory_frame(test_values, test_solution.ts, test_initial_conditions)

    max_log_tvc, train_terminal_log_tvc, tvc_ok = log_tvc_residual(
        policy,
        train_batch_log_states,
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    max_log_tvc = float(max_log_tvc)
    transversality_ok = bool(
        jnp.all(tvc_ok) and max_log_tvc < stop_set.log_tvc_threshold
    )
    _, test_terminal_log_tvc, test_tvc_ok = log_tvc_residual(
        policy,
        test_initial_log_states,
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    test_max_log_tvc = float(jnp.max(test_terminal_log_tvc))
    test_transversality_ok = bool(
        jnp.all(test_tvc_ok) and test_max_log_tvc < stop_set.log_tvc_threshold
    )
    max_train_euler = float(train_df["euler_residual"].abs().max())
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    train_kappa_0 = np.asarray(
        train_batch_initial_conditions[:, 1]
        / train_batch_initial_conditions[:, 0] ** (1.0 / (1.0 - a))
    )
    test_kappa_0 = np.asarray(
        test_initial_conditions[:, 1]
        / test_initial_conditions[:, 0] ** (1.0 / (1.0 - a))
    )
    train_kappa = train_df["kappa"].to_numpy()
    test_kappa = test_df["kappa"].to_numpy()
    collocation_kappa_spread = float(np.max(train_kappa) / np.min(train_kappa))
    results = {
        "variant": "recursive_nn",
        "homothetic_representation": hc_set.homothetic_representation,
        "policy": (
            "delta_k_unit_softplus"
            if hc_set.homothetic_representation
            else "delta_k0_unit_softplus"
        ),
        "state_coordinate": (
            "log(k / z^(1/(1-a)))"
            if hc_set.homothetic_representation
            else "(log z, log k)"
        ),
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "test_z_0": test_z_center,
        "test_k_0": test_k_center,
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "residual_dtype": residual_dtype,
        "metric": "identity_parameter_metric",
        "selection": "minimum_identity_metric_parameter_step",
        "train_loss": train_loss,
        "max_train_euler": max_train_euler,
        "max_test_euler": max_test_euler,
        "final_damping": final_damping,
        "policy_cache_key": cache_key,
        "policy_cache_hit": cache_hit,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_values["euler_residual"]))),
        "test_trajectory_euler_mean_abs_p50": float(
            jnp.quantile(test_trajectory_euler, 0.5)
        ),
        "test_trajectory_euler_mean_abs_p90": float(
            jnp.quantile(test_trajectory_euler, 0.9)
        ),
        "test_trajectory_euler_mean_abs_max": float(jnp.max(test_trajectory_euler)),
        "max_log_tvc_residual": max_log_tvc,
        "mean_log_tvc_residual": float(jnp.mean(train_terminal_log_tvc)),
        "test_mean_log_tvc_residual": float(jnp.mean(test_terminal_log_tvc)),
        "test_max_log_tvc_residual": test_max_log_tvc,
        "test_transversality_ok": test_transversality_ok,
        "tvc_normalization": "relative_to_initial_log_certificate",
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_values["c"])),
        "test_path_ok": test_path_ok,
        "steps": steps,
        "num_epochs": stop_set.num_epochs,
        "epochs_run": epochs_run,
        "parameter_derivative": "conditional_on_carried_training_batch",
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "solve_time": train_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "train_time_success": train_time if solver_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        "solution_accepted": solver_accepted,
        "solver_converged": solver_accepted,
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        **collocation_diagnostics,
        "collocation_path_ok": bool(jnp.all(train_solution.ok)),
        "collocation_kappa_spread": collocation_kappa_spread,
        "collocation_log_kappa_min": float(np.log(np.min(train_kappa))),
        "collocation_log_kappa_max": float(np.log(np.max(train_kappa))),
        "test_log_kappa_min": float(np.log(np.nanmin(test_kappa))),
        "test_log_kappa_max": float(np.log(np.nanmax(test_kappa))),
        "train_kappa_0_min": float(train_kappa_0.min()),
        "train_kappa_0_max": float(train_kappa_0.max()),
        "test_kappa_0_min": float(test_kappa_0.min()),
        "test_kappa_0_max": float(test_kappa_0.max()),
        "train_log_z_variance": ic_set.train_log_z_variance,
        "train_log_k_variance": ic_set.train_log_k_variance,
        "test_log_z_variance": test_log_z_variance,
        "test_log_k_variance": test_log_k_variance,
        "ode_solver": data_set.ode_solver,
        "rtol": data_set.rtol,
        "atol": data_set.atol,
        "initial_dt": data_set.initial_dt,
        "train_points": data_set.train_points,
        "train_steps": data_set.train_points - 1,
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    z_benchmark, k_benchmark, c_benchmark = benchmark_trajectories(
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        test_initial_conditions,
        np.asarray(test_solution.ts[0]),
        data_set,
    )
    k_error, c_error, test_summary_df = add_benchmark_path_diagnostics(
        test_df,
        test_values,
        test_initial_conditions,
        z_benchmark,
        k_benchmark,
        c_benchmark,
        a=a,
        delta=delta,
        g=g,
    )
    results.update(
        test_mean_abs_k_rel_error=float(np.mean(np.abs(k_error))),
        test_mean_abs_c_rel_error=float(np.mean(np.abs(c_error))),
        test_median_abs_k_rel_error=float(np.median(np.abs(k_error))),
        test_median_abs_c_rel_error=float(np.median(np.abs(c_error))),
        test_pooled_median_abs_k_rel_error=float(np.median(np.abs(k_error))),
        test_pooled_median_abs_c_rel_error=float(np.median(np.abs(c_error))),
        test_median_trajectory_mean_abs_k_rel_error=float(
            test_summary_df["mean_k_abs_rel_error"].median()
        ),
        test_median_trajectory_mean_abs_c_rel_error=float(
            test_summary_df["mean_c_abs_rel_error"].median()
        ),
        test_max_abs_k_rel_error=float(np.max(np.abs(k_error))),
        test_max_abs_c_rel_error=float(np.max(np.abs(c_error))),
    )

    # Diagnostic grid bounds come from the realized states, never a steady
    # state. Snapping to a coarse log lattice keeps the cached benchmark
    # starts shared across seeds and data draws.
    observed_kappa = np.concatenate(
        [
            train_kappa,
            test_df["kappa"].to_numpy(),
            train_kappa_0,
            test_kappa_0,
        ]
    )
    observed_kappa = observed_kappa[np.isfinite(observed_kappa) & (observed_kappa > 0)]
    if observed_kappa.size == 0:
        observed_kappa = np.concatenate([train_kappa_0, test_kappa_0])
    grid_lo = math.exp(math.floor(math.log(observed_kappa.min()) / 0.05) * 0.05)
    grid_hi = math.exp(math.ceil(math.log(observed_kappa.max()) / 0.05) * 0.05)
    kappa_grid = positive_policy_grid(grid_lo, grid_hi, data_set.policy_grid_size)
    grid_log_states = jnp.stack(
        [
            jnp.zeros(data_set.policy_grid_size, dtype=dtype),
            jnp.log(jnp.asarray(kappa_grid, dtype=dtype)),
        ],
        axis=1,
    )
    grid_values = trajectory_values(policy, grid_log_states[None, :, :], p)
    grid_output_per_z = np.asarray(
        production(
            jnp.ones(data_set.policy_grid_size, dtype=dtype),
            jnp.asarray(kappa_grid, dtype=dtype),
            p,
        )
    )
    policy_df = pd.DataFrame(
        {
            "kappa": kappa_grid,
            "investment_share_nn": np.asarray(grid_values["i"][0]) / grid_output_per_z,
            # The learned object itself: unit_softplus of the head, since the
            # grid fixes z = 1 so k = kappa.
            "i_over_delta_k_nn": np.asarray(grid_values["i"][0])
            / (delta * np.asarray(kappa_grid)),
            "i_per_z_nn": np.asarray(grid_values["i"][0]),
            "c_tilde_nn": np.asarray(grid_values["c"][0]),
            "dkappa_nn": np.asarray(grid_values["dkappa"][0]),
            **scale_invariance_columns(policy, kappa_grid, p, dtype, "nn"),
        }
    )
    c_benchmark_grid = benchmark_policy_grid(
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        kappa_grid,
        data_set,
    )
    policy_df["c_tilde_benchmark"] = c_benchmark_grid
    policy_df["dkappa_benchmark"] = (
        np.asarray(kappa_grid) ** a
        - c_benchmark_grid
        - (delta + g) * np.asarray(kappa_grid)
    )

    data_hyperparameters = asdict(data_set)
    hyperparameters = {
        "hc_set": asdict(hc_set),
        "data_set": data_hyperparameters,
        "ic_set": asdict(ic_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "g": g,
        },
        "initial_conditions": {
            "z_0": np.asarray(train_initial_conditions[:, 0]).tolist(),
            "k_0": np.asarray(train_initial_conditions[:, 1]).tolist(),
        },
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")
    extras = {
        "model": policy,
        "parameters": p,
        "train_initial_conditions": train_initial_conditions,
        "test_initial_conditions": test_initial_conditions,
        "train_df": train_df,
        "test_df": test_df,
        "policy_df": policy_df,
        "hyperparameters": hyperparameters,
        "test_summary_df": test_summary_df,
    }
    # A cache hit loads the policy from disk instead of solving, so there is
    # no primal solve to describe.
    if not cache_hit:
        extras["solve"] = SolveSpec(
            solver=solver,
            theta_0=theta_0,
            p=p,
            residual_args=residual_args,
            multi_start=multi_start,
            options=solve_options,
        )
        extras["result"] = result
    structural = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
    }
    policy_extra = {
        "model": "neoclassical_growth_nn_recursive",
        "problem_key": problem_key,
        "run_key": run_key,
        "cache_key": cache_key,
        "config": {"problem": problem_config, "run": run_config},
        "solution_accepted": solver_accepted,
        "train_loss": train_loss,
        "test_euler_mean_abs": results["test_euler_mean_abs"],
    }
    if cache_path is not None and not cache_hit:
        save_investment_policy(
            cache_path, policy, hc_set, structural=structural, extra=policy_extra
        )
    if save_policy is not None:
        save_investment_policy(
            save_policy, policy, hc_set, structural=structural, extra=policy_extra
        )
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn_recursive)
