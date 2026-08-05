from __future__ import annotations

import hashlib
import math
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
from nlls_gram import LevenbergMarquardt, LMAction, LMStatus, MultiStart
from tinydiffeq import SRA1, EulerMaruyama, Milstein, SaveAt, solve_sde

from open_economy_growth import ModelParameters, marginal_product, production
from open_economy_growth_benchmark import open_economy_growth_benchmark
from utilities import (
    ACTIVATIONS,
    NEOCLASSICAL_GROWTH_CONSTANTS,
    OptimizerSettings,
    unit_softplus,
)

DOMAIN_EPS = 1e-8
WORLD_RATE_AR1_RHO = 0.572
WORLD_RATE_AR1_INNOVATION_SD = 0.0196
WORLD_RATE_KAPPA = -math.log(WORLD_RATE_AR1_RHO)
WORLD_RATE_SIGMA = WORLD_RATE_AR1_INNOVATION_SD * math.sqrt(
    2.0 * WORLD_RATE_KAPPA / (1.0 - WORLD_RATE_AR1_RHO**2)
)

SDE_SOLVERS = {
    "sra1": SRA1(),
    "milstein": Milstein(),
    "euler_maruyama": EulerMaruyama(),
}

SMOOTH_ACTIVATIONS = {
    "gelu",
    "identity",
    "silu",
    "softplus",
    "tanh",
    "unit_softplus",
}

STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


@struct.dataclass
class StructuralParameters:
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    preference_rate: jax.Array
    investment_eta: jax.Array
    interest_rate_mean: jax.Array
    chi: jax.Array
    kappa_z: jax.Array
    sigma_z: jax.Array
    log_z_bar: jax.Array
    kappa_r: jax.Array
    sigma_r: jax.Array
    corr_zr: jax.Array
    output_scale: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class DataSettings:
    train_points: int = 21
    test_points: int = 41
    tvc_points: int = 41
    train_T: float = 10.0
    test_T: float = 10.0
    tvc_T: float = 20.0


@dataclass(frozen=True)
class InitialConditionSettings:
    data_seed: int = 0
    num_train_trajectories: int = 8
    num_test_trajectories: int = 16
    num_tvc_trajectories: int = 128
    train_log_z_variance: float | None = None
    train_log_k_variance: float = 0.01
    train_asset_variance: float = 0.25
    train_rate_variance: float | None = None
    test_log_z_variance: float | None = None
    test_log_k_variance: float | None = None
    test_asset_variance: float | None = None
    test_rate_variance: float | None = None


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 12
    train_loss_threshold: float | None = None
    test_euler_mean_abs_threshold: float = 5e-3
    test_euler_max_threshold: float = 5e-2
    test_diffusion_max_abs_threshold: float = 5e-2
    test_terminal_capital_relative_threshold: float = 0.25
    num_attempts: int = 1
    num_epochs: int = 8
    epoch_early_stopping: bool = True
    shock_continuation: bool = True
    initial_shock_scale: float = 0.25
    continuation_epochs: int = 3


@dataclass(frozen=True)
class SequentialPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "gelu"
    investment_head_init_scale: float = 1e-3
    log_z_center: float = 0.0
    log_z_scale: float = 1.0
    log_k_center: float = 0.0
    asset_center: float = 0.0
    asset_scale: float = 1.0
    rate_scale: float = 1.0
    output_scale: float = 1.0
    investment_anchor: float = 1.0
    current_account_anchor: float = 0.0


@dataclass(frozen=True)
class InitializationSettings:
    enabled: bool = True
    grid_size: int = 65
    max_steps: int = 40
    loss_threshold: float = 1e-10


@dataclass(frozen=True, eq=False)
class DeterministicPolicyReference:
    coefficients: np.ndarray
    k_lo: float
    k_hi: float
    approximation_max_abs_x: float
    approximation_max_abs_current_account: float
    cache_key: str

    def __hash__(self):
        return hash((DeterministicPolicyReference, self.cache_key))

    def __eq__(self, other):
        return (
            isinstance(other, DeterministicPolicyReference)
            and self.cache_key == other.cache_key
        )


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    initial_states: jax.Array
    initial_center: jax.Array
    initial_scales: jax.Array
    noise: object
    epoch_key: jax.Array
    epoch: jax.Array
    shock_scales: jax.Array
    shock_scale: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    solver: object = struct.field(pytree_node=False)
    data_set: DataSettings = struct.field(pytree_node=False)


@struct.dataclass
class PolicyInitializationArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    states: jax.Array
    targets: jax.Array


def capital_only_kernel_init(key, shape, dtype):
    kernel = nnx.initializers.lecun_normal()(key, shape, dtype)
    mask = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=dtype)[:, None]
    return kernel * mask


class SequentialPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: SequentialPolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.log_z_center = nnx.Variable(
            jnp.asarray(settings.log_z_center, dtype=dtype)
        )
        self.log_z_scale = nnx.Variable(jnp.asarray(settings.log_z_scale, dtype=dtype))
        self.log_k_center = nnx.Variable(
            jnp.asarray(settings.log_k_center, dtype=dtype)
        )
        self.asset_center = nnx.Variable(
            jnp.asarray(settings.asset_center, dtype=dtype)
        )
        self.asset_scale = nnx.Variable(jnp.asarray(settings.asset_scale, dtype=dtype))
        self.rate_scale = nnx.Variable(jnp.asarray(settings.rate_scale, dtype=dtype))
        self.output_scale = nnx.Variable(
            jnp.asarray(settings.output_scale, dtype=dtype)
        )
        self.investment_anchor = nnx.Variable(
            jnp.asarray(settings.investment_anchor, dtype=dtype)
        )
        self.current_account_anchor = nnx.Variable(
            jnp.asarray(settings.current_account_anchor, dtype=dtype)
        )
        self.layers = nnx.List(
            [
                nnx.Linear(
                    4,
                    settings.width,
                    kernel_init=capital_only_kernel_init,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(settings.depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.width,
                    settings.width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.investment_head = nnx.Linear(
            settings.width,
            1,
            kernel_init=nnx.initializers.normal(settings.investment_head_init_scale),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.current_account_head = nnx.Linear(
            settings.width,
            1,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, state):
        log_z, log_k, asset_scaled, rate_gap = state
        normalized_asset = 2.0 * jnp.tanh(
            (asset_scaled - self.asset_center[...]) / (2.0 * self.asset_scale[...])
        )
        h = jnp.stack(
            [
                (log_z - self.log_z_center[...]) / self.log_z_scale[...],
                log_k - self.log_k_center[...],
                normalized_asset,
                rate_gap / self.rate_scale[...],
            ]
        )
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        raw_investment = self.investment_head(h)[0]
        raw_current_account = self.current_account_head(h)[0]
        investment = self.investment_anchor[...] * unit_softplus(raw_investment)
        current_account = (
            self.current_account_anchor[...]
            + self.output_scale[...] * raw_current_account
        )
        return jnp.stack([investment, current_account])


class DrawSequentialPolicy:
    def __init__(self, settings, dtype):
        self.settings = settings
        self.dtype = dtype

    def __call__(self, key, _, args_old):
        module = SequentialPolicyNN(
            self.settings,
            dtype=self.dtype,
            rngs=nnx.Rngs(key),
        )
        _, theta, _ = nnx.split(module, nnx.Param, ...)
        return theta, args_old

    def __hash__(self):
        return hash((DrawSequentialPolicy, self.settings, self.dtype))

    def __eq__(self, other):
        return (
            isinstance(other, DrawSequentialPolicy)
            and self.settings == other.settings
            and self.dtype == other.dtype
        )


def evaluate_chebyshev(coefficients, coordinate):
    next_value = jnp.zeros_like(coefficients[0])
    next_next_value = jnp.zeros_like(coefficients[0])
    for index in range(coefficients.shape[0] - 1, 0, -1):
        value = 2.0 * coordinate * next_value - next_next_value + coefficients[index]
        next_next_value = next_value
        next_value = value
    return coordinate * next_value - next_next_value + coefficients[0]


@lru_cache(maxsize=None)
def create_deterministic_policy_reference(
    parameters,
    k_lo=0.4,
    k_hi=8.0,
    degree=20,
):
    if not 0.0 < k_lo < k_hi:
        raise ValueError("reference capital bounds must satisfy zero < lo < hi")
    if degree < 1:
        raise ValueError("reference degree must be positive")
    time_grid = np.linspace(0.0, 160.0, 5001)
    path_values = []
    for initial_capital in (k_lo, k_hi):
        evaluate, diagnostics = open_economy_growth_benchmark(
            initial_capital,
            0.0,
            time_grid,
            parameters=parameters,
        )
        if abs(diagnostics["present_value_residual"]) > 1e-8:
            raise RuntimeError("deterministic reference present-value check failed")
        values = evaluate(time_grid)
        path_values.append(
            np.column_stack([values["k"], values["x"], values["current_account"]])
        )
    raw = np.concatenate(path_values)
    raw = raw[np.argsort(raw[:, 0])]
    fitting_capital = np.linspace(k_lo, k_hi, 2001)
    fitting_values = np.column_stack(
        [np.interp(fitting_capital, raw[:, 0], raw[:, column]) for column in (1, 2)]
    )
    coordinate = 2.0 * (fitting_capital - k_lo) / (k_hi - k_lo) - 1.0
    coefficients = np.column_stack(
        [
            np.polynomial.chebyshev.chebfit(
                coordinate, fitting_values[:, column], degree
            )
            for column in range(fitting_values.shape[1])
        ]
    )
    approximation = np.column_stack(
        [
            np.polynomial.chebyshev.chebval(coordinate, coefficients[:, column])
            for column in range(coefficients.shape[1])
        ]
    )
    approximation_max_abs = np.max(np.abs(approximation - fitting_values), axis=0)
    digest = hashlib.sha256()
    digest.update(repr((parameters, k_lo, k_hi, degree)).encode())
    digest.update(coefficients.tobytes())
    return DeterministicPolicyReference(
        coefficients=coefficients,
        k_lo=k_lo,
        k_hi=k_hi,
        approximation_max_abs_x=float(approximation_max_abs[0]),
        approximation_max_abs_current_account=float(approximation_max_abs[1]),
        cache_key=digest.hexdigest(),
    )


def deterministic_reference_values(reference, capital, dtype):
    coordinate = (
        2.0 * (capital - reference.k_lo) / (reference.k_hi - reference.k_lo) - 1.0
    )
    return evaluate_chebyshev(
        jnp.asarray(reference.coefficients, dtype=dtype), coordinate
    )


def policy_initialization_residual(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.fixed_state)
    predictions = jax.vmap(policy)(args.states)
    investment_residual = jnp.log(predictions[:, 0] / args.targets[:, 0])
    current_account_residual = (predictions[:, 1] - args.targets[:, 1]) / p.output_scale
    residual = jnp.stack([investment_residual, current_account_residual], axis=-1)
    count = jnp.asarray(residual.size, dtype=residual.dtype)
    return residual.reshape(-1) / jnp.sqrt(count)


def state_levels(state, p):
    log_z, log_k, asset_scaled, rate_gap = state
    return {
        "z": jnp.exp(log_z),
        "k": jnp.exp(log_k),
        "a": p.output_scale * asset_scaled,
        "interest_rate": p.interest_rate_mean + rate_gap,
    }


def policy_values(policy, state, p):
    levels = state_levels(state, p)
    z = levels["z"]
    k = levels["k"]
    a = levels["a"]
    interest_rate = levels["interest_rate"]
    investment, current_account = policy(state)
    investment_rate = investment / k
    q = investment_rate ** (1.0 - p.investment_eta) / (p.chi * p.investment_eta)
    output = production(k, z, p)
    consumption = interest_rate * a + output - investment - current_account
    capital_drift = k * (p.chi * investment_rate**p.investment_eta - p.delta)
    return {
        **levels,
        "q": q,
        "c": consumption,
        "x": investment,
        "output": output,
        "current_account": current_account,
        "capital_drift": capital_drift,
    }


def state_drift(policy, state, p):
    values = policy_values(policy, state, p)
    log_z, _, _, rate_gap = state
    return jnp.stack(
        [
            p.kappa_z * (p.log_z_bar - log_z),
            values["capital_drift"] / values["k"],
            values["current_account"] / p.output_scale,
            -p.kappa_r * rate_gap,
        ]
    )


def state_diffusion(state, p):
    zero = jnp.zeros_like(state[0])
    return jnp.stack([p.sigma_z, zero, zero, p.sigma_r])


def diffusion_directions(p):
    zero = jnp.zeros_like(p.sigma_z)
    second_loading = p.sigma_r * jnp.sqrt(1.0 - p.corr_zr**2)
    return jnp.stack(
        [
            jnp.stack([p.sigma_z, zero, zero, p.corr_zr * p.sigma_r]),
            jnp.stack([zero, zero, zero, second_loading]),
        ]
    )


def log_policy_quantities(policy, state, p):
    values = policy_values(policy, state, p)
    return jnp.stack([jnp.log(values["c"]), jnp.log(values["q"])])


# With l = log(c), v = log(q), and Gamma the diffusion covariance product:
#   L l - gamma Gamma(l,l)/2 = (r_star - rho)/gamma
#   L v + Gamma(v,v)/2 - gamma Gamma(v,l)
#       = r_star + delta - (1-eta) chi s^eta - MPK/q.
def euler_residual_at_state(policy, state, p):
    drift = state_drift(policy, state, p)
    directions = diffusion_directions(p)

    def logs(candidate):
        return log_policy_quantities(policy, candidate, p)

    log_values, drift_derivative = jax.jvp(logs, (state,), (drift,))

    def directional_terms(direction):
        first = jax.jvp(logs, (state,), (direction,))[1]
        second = jax.jvp(
            lambda candidate: jax.jvp(logs, (candidate,), (direction,))[1],
            (state,),
            (direction,),
        )[1]
        return first, second

    first, second = jax.vmap(directional_terms)(directions)
    covariance = first.T @ first
    generator = drift_derivative + 0.5 * jnp.sum(second, axis=0)
    values = policy_values(policy, state, p)
    investment_rate = values["x"] / values["k"]
    q_rhs = (
        values["interest_rate"]
        + p.delta
        - (1.0 - p.investment_eta) * p.chi * investment_rate**p.investment_eta
        - marginal_product(values["k"], values["z"], p) / values["q"]
    )
    consumption_residual = (
        generator[0]
        - 0.5 * p.gamma * covariance[0, 0]
        - (values["interest_rate"] - p.preference_rate) / p.gamma
    )
    q_residual = (
        generator[1] + 0.5 * covariance[1, 1] - p.gamma * covariance[1, 0] - q_rhs
    )
    diffusion_terms = jnp.stack(
        [
            0.5 * jnp.sum(second[:, 0]) - 0.5 * p.gamma * covariance[0, 0],
            0.5 * jnp.sum(second[:, 1])
            + 0.5 * covariance[1, 1]
            - p.gamma * covariance[1, 0],
        ]
    )
    return jnp.stack([consumption_residual, q_residual]), diffusion_terms, log_values


def euler_residual_terms(policy, states, p):
    residuals, diffusion_terms, log_values = jax.vmap(
        lambda state: euler_residual_at_state(policy, state, p)
    )(states)
    return residuals, diffusion_terms, log_values


def standardized_initial_states(key, center, scales, num_trajectories):
    if num_trajectories == 1:
        return center[None, :]
    raw = jax.random.normal(key, (num_trajectories, center.size), dtype=center.dtype)
    centered = raw - jnp.mean(raw, axis=0)
    standardized = centered / jnp.sqrt(jnp.mean(centered**2, axis=0))
    return center + standardized * scales


def sample_trajectory_noise(solver, key, num_trajectories, n_steps, dt, dtype):
    reference_state = jnp.zeros(4, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    return jax.vmap(
        lambda trajectory_key: solver.sample_noise(
            reference_state, trajectory_key, n_steps, dt, dtype
        )
    )(jax.random.split(key, num_trajectories))


def correlate_noise(noise, corr_zr):
    orthogonal_loading = jnp.sqrt(1.0 - corr_zr**2)

    def transform(increments):
        rate_increments = (
            corr_zr * increments[..., 0] + orthogonal_loading * increments[..., 3]
        )
        return increments.at[..., 3].set(rate_increments)

    return jax.tree.map(transform, noise)


@dataclass(frozen=True)
class EpochCallback:
    num_train_trajectories: int
    n_steps: int
    dt: float
    epoch_early_stopping: bool
    resample_initial_states: bool

    def next_epoch(self, ctx):
        epoch_key, draw_key, noise_key = jax.random.split(ctx.args.epoch_key, 3)
        dtype = ctx.args.initial_states.dtype
        epoch = ctx.args.epoch + 1
        shock_scale = ctx.args.shock_scales[epoch]
        if self.resample_initial_states:
            scale_mask = jnp.asarray([shock_scale, 1.0, 1.0, shock_scale], dtype=dtype)
            initial_states = standardized_initial_states(
                draw_key,
                ctx.args.initial_center,
                ctx.args.initial_scales * scale_mask,
                self.num_train_trajectories,
            )
        else:
            initial_states = ctx.args.initial_states
        fresh = ctx.args.replace(
            initial_states=initial_states,
            noise=sample_trajectory_noise(
                ctx.args.solver,
                noise_key,
                self.num_train_trajectories,
                self.n_steps,
                self.dt,
                dtype,
            ),
            epoch_key=epoch_key,
            epoch=epoch,
            shock_scale=shock_scale,
        )
        residual = residual_fn(ctx.x, fresh, ctx.p)
        loss = jnp.sum(residual * residual)
        nonfinite = ~jnp.isfinite(loss)
        converged = jnp.asarray(False)
        if self.epoch_early_stopping:
            converged = (shock_scale == 1.0) & (loss < fresh.train_loss_threshold)
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

    def continue_epoch(self, ctx):
        return (
            jnp.asarray(False),
            jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
            ctx.args,
            ctx.lm_state.damping,
        )

    def finish_epoch(self, ctx):
        residual = residual_fn(ctx.x, ctx.args, ctx.p)
        loss = jnp.sum(residual * residual)
        nonfinite = ~jnp.isfinite(loss)
        converged = (ctx.args.shock_scale == 1.0) & (
            loss < ctx.args.train_loss_threshold
        )
        status = jnp.where(
            nonfinite,
            jnp.asarray(LMStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(LMStatus.CONVERGED, dtype=jnp.int32),
                jnp.asarray(LMStatus.RUNNING, dtype=jnp.int32),
            ),
        )
        return nonfinite | converged, status, ctx.args, ctx.lm_state.damping

    def mid_epoch(self, ctx):
        if not self.epoch_early_stopping:
            return self.continue_epoch(ctx)
        return jax.lax.cond(
            ctx.step == ctx.args.total_steps,
            self.finish_epoch,
            self.continue_epoch,
            ctx,
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


def drift_field(state, _, args, p):
    if args is not None:
        raise ValueError("drift_field does not accept solver args")
    policy, structural = p
    return state_drift(policy, state, structural)


def diffusion_field(state, _, args, p):
    if args is not None:
        raise ValueError("diffusion_field does not accept solver args")
    _, structural = p
    return state_diffusion(state, structural)


def batched_sde_rollout(solver, policy, initial_states, noise, T, n_steps, p):
    correlated_noise = correlate_noise(noise, p.corr_zr)

    def solve_one(initial_state, trajectory_noise):
        return solve_sde(
            drift_field,
            diffusion_field,
            solver,
            0.0,
            T,
            initial_state,
            noise=trajectory_noise,
            n_steps=n_steps,
            save_at=SaveAt(steps=True),
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_states, correlated_noise)


@partial(jax.jit, static_argnames=("data_set",))
def collocation_rollout(solver, policy, initial_states, noise, p, data_set):
    return batched_sde_rollout(
        solver,
        policy,
        initial_states,
        noise,
        data_set.train_T,
        data_set.train_points - 1,
        p,
    )


@partial(jax.jit, static_argnames=("T", "n_steps"))
def rollout_steps(solver, policy, initial_states, noise, T, n_steps, p):
    return batched_sde_rollout(solver, policy, initial_states, noise, T, n_steps, p)


def collocation_residuals(solver, policy, initial_states, noise, p, data_set):
    solution = collocation_rollout(solver, policy, initial_states, noise, p, data_set)
    states = solution.xs
    residuals = euler_residual_terms(policy, states.reshape(-1, 4), p)[0].reshape(
        *states.shape[:2], 2
    )
    count = jnp.asarray(residuals.size, dtype=residuals.dtype)
    return jnp.where(
        solution.ok[:, None, None],
        residuals / jnp.sqrt(count),
        jnp.full_like(residuals, jnp.inf),
    )


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.fixed_state)
    stage_parameters = p.replace(
        sigma_z=p.sigma_z * args.shock_scale,
        sigma_r=p.sigma_r * args.shock_scale,
    )
    return collocation_residuals(
        args.solver,
        policy,
        args.initial_states,
        args.noise,
        stage_parameters,
        args.data_set,
    ).reshape(-1)


def accept_training(_, result):
    residual = residual_fn(result.x, result.args, result.p)
    loss = jnp.sum(residual * residual)
    return (
        (result.args.shock_scale == 1.0)
        & jnp.isfinite(loss)
        & (loss < result.args.train_loss_threshold)
    )


@nnx.jit
def trajectory_values(policy, states, p):
    shape = states.shape[:-1]
    flat_states = states.reshape(-1, 4)
    values = jax.vmap(lambda state: policy_values(policy, state, p))(flat_states)
    residuals, diffusion_terms, _ = euler_residual_terms(policy, flat_states, p)
    return {
        **{name: value.reshape(shape) for name, value in values.items()},
        "c_euler": residuals[:, 0].reshape(shape),
        "q_euler": residuals[:, 1].reshape(shape),
        "c_diffusion_term": diffusion_terms[:, 0].reshape(shape),
        "q_diffusion_term": diffusion_terms[:, 1].reshape(shape),
    }


def transversality_terms(values, times, p):
    marginal_utility = jnp.exp(-p.gamma * jnp.log(values["c"]))
    discounted_marginal_utility = jnp.exp(-p.preference_rate * times) * marginal_utility
    return {
        "discounted_marginal_utility": discounted_marginal_utility,
        "absolute_asset_tvc": discounted_marginal_utility * jnp.abs(values["a"]),
        "signed_asset_tvc": discounted_marginal_utility * values["a"],
        "capital_tvc": discounted_marginal_utility * jnp.abs(values["q"] * values["k"]),
    }


@nnx.jit
def transversality_trajectory_values(policy, states, times, p):
    shape = states.shape[:-1]
    flat_states = states.reshape(-1, 4)
    values = jax.vmap(lambda state: policy_values(policy, state, p))(flat_states)
    values = {name: value.reshape(shape) for name, value in values.items()}
    return {**values, **transversality_terms(values, times, p)}


def monte_carlo_standard_error(values, axis=0):
    sample_size = values.shape[axis]
    if sample_size == 1:
        shape = values.shape[:axis] + values.shape[axis + 1 :]
        return jnp.full(shape, jnp.nan, dtype=values.dtype)
    return jnp.std(values, axis=axis, ddof=1) / jnp.sqrt(
        jnp.asarray(sample_size, dtype=values.dtype)
    )


def transversality_statistics(values):
    absolute_asset_tvc = values["absolute_asset_tvc"][:, -1]
    signed_asset_tvc = values["signed_asset_tvc"][:, -1]
    capital_tvc = values["capital_tvc"][:, -1]
    expected_absolute_asset_tvc = jnp.mean(absolute_asset_tvc)
    expected_capital_tvc = jnp.mean(capital_tvc)
    return {
        "expected_absolute_asset_tvc": expected_absolute_asset_tvc,
        "expected_capital_tvc": expected_capital_tvc,
        "signed_expected_asset_tvc": jnp.mean(signed_asset_tvc),
        "absolute_asset_tvc_mc_se": monte_carlo_standard_error(absolute_asset_tvc),
        "signed_asset_tvc_mc_se": monte_carlo_standard_error(signed_asset_tvc),
        "capital_tvc_mc_se": monte_carlo_standard_error(capital_tvc),
        "log_expected_absolute_asset_tvc": jnp.log(expected_absolute_asset_tvc),
        "log_expected_capital_tvc": jnp.log(expected_capital_tvc),
        "max_log_absolute_asset_tvc": jnp.max(jnp.log(absolute_asset_tvc)),
        "max_log_capital_tvc": jnp.max(jnp.log(capital_tvc)),
    }


def trajectory_frame(values, times):
    num_trajectories, num_points = times.shape
    columns = {
        "trajectory": np.repeat(np.arange(num_trajectories), num_points),
        "t": np.asarray(times).reshape(-1),
    }
    columns.update(
        {name: np.asarray(value).reshape(-1) for name, value in values.items()}
    )
    return pd.DataFrame(columns)


def transversality_expectation_frame(values, times):
    return pd.DataFrame(
        {
            "t": np.asarray(times[0]),
            "expected_absolute_asset_tvc": np.asarray(
                jnp.mean(values["absolute_asset_tvc"], axis=0)
            ),
            "expected_capital_tvc": np.asarray(jnp.mean(values["capital_tvc"], axis=0)),
            "signed_expected_asset_tvc": np.asarray(
                jnp.mean(values["signed_asset_tvc"], axis=0)
            ),
            "absolute_asset_tvc_mc_se": np.asarray(
                monte_carlo_standard_error(values["absolute_asset_tvc"])
            ),
            "signed_asset_tvc_mc_se": np.asarray(
                monte_carlo_standard_error(values["signed_asset_tvc"])
            ),
            "capital_tvc_mc_se": np.asarray(
                monte_carlo_standard_error(values["capital_tvc"])
            ),
        }
    )


def residual_jacobian_diagnostics(theta, args, p, damping):
    theta_flat, unravel = ravel_pytree(theta)
    jacobian = jax.jacrev(lambda flat: residual_fn(unravel(flat), args, p))(theta_flat)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    jax.block_until_ready(singular_values)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    tolerance = (
        np.finfo(np.asarray(singular_values).dtype).eps * max(jacobian.shape) * largest
    )
    rank = int(jnp.sum(singular_values > tolerance))
    smallest_retained = float(singular_values[rank - 1]) if rank > 0 else 0.0
    condition_number = (
        largest / smallest_retained if smallest_retained > 0.0 else math.inf
    )
    gram_minimum_eigenvalue = smallest**2 if rank == jacobian.shape[0] else 0.0
    damped_gram_condition_number = (largest**2 + damping) / (
        gram_minimum_eigenvalue + damping
    )
    return {
        "jacobian_sigma_max": largest,
        "jacobian_sigma_min": smallest,
        "jacobian_sigma_min_retained": smallest_retained,
        "jacobian_condition_number": condition_number,
        "undamped_gram_condition_number": (
            condition_number**2 if rank == jacobian.shape[0] else math.inf
        ),
        "damped_gram_condition_number": damped_gram_condition_number,
        "jacobian_effective_rank": rank,
        "jacobian_nullity": int(jacobian.shape[1] - rank),
        "jacobian_row_rank_deficiency": int(jacobian.shape[0] - rank),
        "jacobian_row_count": int(jacobian.shape[0]),
        "jacobian_column_count": int(jacobian.shape[1]),
    }


def stochastic_open_economy_growth_nn_recursive(
    k_0: float = 1.0,
    a_0: float = 0.0,
    z_0: float = 1.0,
    rate_gap_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    kappa_z: float = NEOCLASSICAL_GROWTH_CONSTANTS.eta,
    sigma_z: float = NEOCLASSICAL_GROWTH_CONSTANTS.sigma_z_high,
    z_bar: float = 1.0,
    kappa_r: float = WORLD_RATE_KAPPA,
    sigma_r: float = WORLD_RATE_SIGMA,
    corr_zr: float = 0.0,
    seed: int = 0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    nn_set: SequentialPolicySettings = SequentialPolicySettings(),
    opt_set: OptimizerSettings = OptimizerSettings(
        init_damping=3.26e-5,
        damping_decrease=0.482,
        damping_increase=2.71,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.541,
    ),
    stop_set: StoppingCriteria = StoppingCriteria(),
    init_set: InitializationSettings = InitializationSettings(),
    asset_shift_scale: float = 0.5,
    deterministic_reference: bool = True,
    reference_k_lo: float = 0.4,
    reference_k_hi: float = 8.0,
    reference_degree: int = 20,
    diagnose_jacobian: bool = False,
    verbose: bool = True,
):
    driver_start = time.perf_counter()
    if use_float64 and not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    if stop_set.train_loss_threshold is None:
        threshold = 1e-8 if dtype == jnp.float64 else 1e-6
        stop_set = replace(stop_set, train_loss_threshold=threshold)
    if k_0 <= 0.0:
        raise ValueError("k_0 must be positive")
    if z_0 <= 0.0 or z_bar <= 0.0:
        raise ValueError("z_0 and z_bar must be positive")
    if kappa_z <= 0.0 or kappa_r <= 0.0:
        raise ValueError("shock mean-reversion rates must be positive")
    if sigma_z < 0.0 or sigma_r < 0.0:
        raise ValueError("shock diffusion scales must be nonnegative")
    if not -1.0 < corr_zr < 1.0:
        raise ValueError("corr_zr must lie strictly between -1 and one")
    if sde_solver not in SDE_SOLVERS:
        raise ValueError(f"sde_solver must be one of {sorted(SDE_SOLVERS)}")
    if data_set.train_T <= 0.0 or data_set.test_T <= 0.0 or data_set.tvc_T <= 0.0:
        raise ValueError("train_T, test_T, and tvc_T must be positive")
    if data_set.train_points < 2 or data_set.test_points < 2 or data_set.tvc_points < 2:
        raise ValueError(
            "train_points, test_points, and tvc_points must be at least two"
        )
    if ic_set.num_train_trajectories < 1 or ic_set.num_test_trajectories < 1:
        raise ValueError("training and test trajectory counts must be positive")
    if ic_set.num_tvc_trajectories < 2:
        raise ValueError("num_tvc_trajectories must be at least two")
    if stop_set.max_steps < 1 or stop_set.num_epochs < 1:
        raise ValueError("max_steps and num_epochs must be positive")
    if stop_set.num_attempts < 1:
        raise ValueError("num_attempts must be positive")
    if not 1 <= stop_set.continuation_epochs <= stop_set.num_epochs:
        raise ValueError("continuation_epochs must lie in [1, num_epochs]")
    if not 0.0 < stop_set.initial_shock_scale <= 1.0:
        raise ValueError("initial_shock_scale must lie in (0, 1]")
    if nn_set.width < 1 or nn_set.depth < 1:
        raise ValueError("network width and depth must be positive")
    if nn_set.hidden_activation not in SMOOTH_ACTIVATIONS:
        raise ValueError(
            f"hidden_activation must be one of {sorted(SMOOTH_ACTIVATIONS)}"
        )
    if nn_set.investment_head_init_scale < 0.0:
        raise ValueError("investment_head_init_scale must be nonnegative")
    if asset_shift_scale <= 0.0:
        raise ValueError("asset_shift_scale must be positive")
    if init_set.grid_size < 2 or init_set.max_steps < 1:
        raise ValueError("initialization grid_size and max_steps must be positive")
    if not math.isfinite(init_set.loss_threshold) or init_set.loss_threshold <= 0.0:
        raise ValueError("initialization loss_threshold must be positive")
    if init_set.enabled and not deterministic_reference:
        raise ValueError("policy initialization requires a deterministic reference")
    if init_set.enabled and stop_set.num_attempts != 1:
        raise ValueError("reference-initialized training requires num_attempts=1")
    if deterministic_reference and z_bar != 1.0:
        raise ValueError("deterministic reference requires unit mean productivity")
    if deterministic_reference and not math.isclose(
        parameters.interest_rate, parameters.r
    ):
        raise ValueError(
            "deterministic reference requires equal interest and preference rates"
        )
    if deterministic_reference and not 0.0 < reference_k_lo < reference_k_hi:
        raise ValueError("reference capital bounds must satisfy zero < lo < hi")
    if deterministic_reference and reference_degree < 1:
        raise ValueError("reference_degree must be positive")
    if deterministic_reference and not reference_k_lo <= k_0 <= reference_k_hi:
        raise ValueError("initial capital must lie inside the reference bounds")
    if stop_set.train_loss_threshold is None or not (
        math.isfinite(stop_set.train_loss_threshold)
        and stop_set.train_loss_threshold > 0.0
    ):
        raise ValueError("train_loss_threshold must be positive")
    test_thresholds = (
        stop_set.test_euler_mean_abs_threshold,
        stop_set.test_euler_max_threshold,
        stop_set.test_diffusion_max_abs_threshold,
        stop_set.test_terminal_capital_relative_threshold,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in test_thresholds):
        raise ValueError("test thresholds must be positive")

    z_ergodic_variance = sigma_z**2 / (2.0 * kappa_z)
    rate_ergodic_variance = sigma_r**2 / (2.0 * kappa_r)
    train_variances = (
        z_ergodic_variance
        if ic_set.train_log_z_variance is None
        else ic_set.train_log_z_variance,
        ic_set.train_log_k_variance,
        ic_set.train_asset_variance,
        rate_ergodic_variance
        if ic_set.train_rate_variance is None
        else ic_set.train_rate_variance,
    )
    test_variances = (
        train_variances[0]
        if ic_set.test_log_z_variance is None
        else ic_set.test_log_z_variance,
        train_variances[1]
        if ic_set.test_log_k_variance is None
        else ic_set.test_log_k_variance,
        train_variances[2]
        if ic_set.test_asset_variance is None
        else ic_set.test_asset_variance,
        train_variances[3]
        if ic_set.test_rate_variance is None
        else ic_set.test_rate_variance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in train_variances):
        raise ValueError("training initial-condition variances must be nonnegative")
    if any(not math.isfinite(value) or value < 0.0 for value in test_variances):
        raise ValueError("test initial-condition variances must be nonnegative")
    if ic_set.num_train_trajectories == 1 and any(
        value > 0.0 for value in train_variances
    ):
        raise ValueError(
            "multiple training trajectories are required for nonzero spread"
        )
    if ic_set.num_test_trajectories == 1 and any(
        value > 0.0 for value in test_variances
    ):
        raise ValueError("multiple test trajectories are required for nonzero spread")

    output_scale = float(production(k_0, z_0, parameters))
    initial_center = jnp.asarray(
        [math.log(z_0), math.log(k_0), a_0 / output_scale, rate_gap_0],
        dtype=dtype,
    )
    train_scales = jnp.sqrt(jnp.asarray(train_variances, dtype=dtype))
    test_scales = jnp.sqrt(jnp.asarray(test_variances, dtype=dtype))
    reference_start = time.perf_counter()
    if deterministic_reference:
        reference = create_deterministic_policy_reference(
            parameters,
            k_lo=reference_k_lo,
            k_hi=reference_k_hi,
            degree=reference_degree,
        )
        reference_terminal = max(200.0, data_set.test_T)
        reference_grid = np.linspace(0.0, reference_terminal, 401)
        reference_policy, reference_diagnostics = open_economy_growth_benchmark(
            k_0,
            a_0,
            reference_grid,
            parameters=parameters,
        )
        reference_values = reference_policy(np.asarray([data_set.test_T]))
        deterministic_terminal_capital = float(reference_values["k"][0])
        deterministic_terminal_asset = float(reference_values["a"][0])
        if abs(reference_diagnostics["present_value_residual"]) > 1e-8:
            raise RuntimeError("deterministic reference present-value check failed")
    else:
        reference = None
        deterministic_terminal_capital = math.nan
        deterministic_terminal_asset = math.nan
    reference_setup_time = time.perf_counter() - reference_start
    investment_anchor = max(parameters.delta * k_0, DOMAIN_EPS)
    current_account_anchor = 0.0
    nn_set = replace(
        nn_set,
        log_z_center=math.log(z_bar),
        log_z_scale=max(
            math.sqrt(z_ergodic_variance),
            math.sqrt(train_variances[0]),
            DOMAIN_EPS,
        ),
        log_k_center=math.log(k_0),
        asset_center=a_0 / output_scale,
        asset_scale=max(math.sqrt(train_variances[2]), 1.0),
        rate_scale=max(
            math.sqrt(rate_ergodic_variance),
            math.sqrt(train_variances[3]),
            DOMAIN_EPS,
        ),
        output_scale=output_scale,
        investment_anchor=investment_anchor,
        current_account_anchor=current_account_anchor,
    )
    p = StructuralParameters.create(
        dtype,
        alpha=parameters.alpha,
        gamma=parameters.gamma,
        delta=parameters.delta,
        preference_rate=parameters.r,
        investment_eta=parameters.eta,
        interest_rate_mean=parameters.interest_rate,
        chi=parameters.chi,
        kappa_z=kappa_z,
        sigma_z=sigma_z,
        log_z_bar=math.log(z_bar),
        kappa_r=kappa_r,
        sigma_r=sigma_r,
        corr_zr=corr_zr,
        output_scale=output_scale,
    )
    solver = SDE_SOLVERS[sde_solver]
    n_train_steps = data_set.train_points - 1
    n_test_steps = data_set.test_points - 1
    n_tvc_steps = data_set.tvc_points - 1
    train_dt = data_set.train_T / n_train_steps
    test_dt = data_set.test_T / n_test_steps
    tvc_dt = data_set.tvc_T / n_tvc_steps

    if stop_set.shock_continuation:
        continuation_scales = (
            np.ones(1)
            if stop_set.continuation_epochs == 1
            else np.geomspace(
                stop_set.initial_shock_scale,
                1.0,
                stop_set.continuation_epochs,
            )
        )
        shock_scales = jnp.asarray(
            np.concatenate(
                [
                    continuation_scales,
                    np.ones(stop_set.num_epochs - stop_set.continuation_epochs),
                ]
            ),
            dtype=dtype,
        )
    else:
        shock_scales = jnp.ones(stop_set.num_epochs, dtype=dtype)
    initial_scale_mask = jnp.asarray(
        [shock_scales[0], 1.0, 1.0, shock_scales[0]], dtype=dtype
    )

    data_key = jax.random.key(ic_set.data_seed)
    train_ic_key, test_ic_key = jax.random.split(data_key)
    train_initial_states = standardized_initial_states(
        train_ic_key,
        initial_center,
        train_scales * initial_scale_mask,
        ic_set.num_train_trajectories,
    )
    test_initial_states = standardized_initial_states(
        test_ic_key,
        initial_center,
        test_scales,
        ic_set.num_test_trajectories,
    )

    fit_start = time.perf_counter()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    initial_model = SequentialPolicyNN(
        nn_set,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, fixed_state = nnx.split(initial_model, nnx.Param, ...)
    diagnostic_k_lo = reference_k_lo if reference is not None else 0.8 * k_0
    diagnostic_k_hi = reference_k_hi if reference is not None else 1.2 * k_0
    initialization_capital = jnp.linspace(
        diagnostic_k_lo,
        diagnostic_k_hi,
        init_set.grid_size,
        dtype=dtype,
    )
    initialization_states = jnp.column_stack(
        [
            jnp.full_like(initialization_capital, math.log(z_bar)),
            jnp.log(initialization_capital),
            jnp.full_like(initialization_capital, a_0 / output_scale),
            jnp.zeros_like(initialization_capital),
        ]
    )
    raw_predictions = jax.vmap(initial_model)(initialization_states)
    jax.block_until_ready(raw_predictions)
    raw_investment_relative_range = float(
        (jnp.max(raw_predictions[:, 0]) - jnp.min(raw_predictions[:, 0]))
        / investment_anchor
    )
    raw_current_account_max_abs = float(jnp.max(jnp.abs(raw_predictions[:, 1])))

    initialization_loss_before = math.nan
    initialization_loss = math.nan
    initialization_steps = 0
    initialization_status = "disabled"
    initialization_time = 0.0
    initialization_investment_log_error_max = math.nan
    initialization_current_account_error_max = math.nan
    if reference is not None:
        initialization_targets = jax.vmap(
            lambda capital: deterministic_reference_values(reference, capital, dtype)
        )(initialization_capital)
        initialization_args = PolicyInitializationArgs(
            graphdef=graphdef,
            fixed_state=fixed_state,
            states=initialization_states,
            targets=initialization_targets,
        )
        residual_before = policy_initialization_residual(
            theta_0, initialization_args, p
        )
        initialization_loss_before = float(jnp.sum(residual_before * residual_before))
        if init_set.enabled:
            initialization_solver = LevenbergMarquardt(
                policy_initialization_residual,
                init_damping=opt_set.init_damping,
                damping_decrease=opt_set.damping_decrease,
                damping_increase=opt_set.damping_increase,
                geodesic_acceleration=opt_set.geodesic_acceleration,
                geodesic_acceptance_ratio=opt_set.geodesic_acceptance_ratio,
            )
            initialization_start = time.perf_counter()
            initialization_result = initialization_solver.solve(
                theta_0,
                initialization_args,
                p=p,
                max_steps=init_set.max_steps,
                atol=math.sqrt(init_set.loss_threshold),
                max_steps_is_success=False,
            )
            jax.block_until_ready(initialization_result)
            initialization_time = time.perf_counter() - initialization_start
            theta_0 = initialization_result.x
            initialization_residual = policy_initialization_residual(
                theta_0, initialization_args, p
            )
            initialization_loss = float(
                jnp.sum(initialization_residual * initialization_residual)
            )
            initialization_steps = int(initialization_result.steps)
            initialization_status = STATUS_TO_REASON.get(
                int(initialization_result.status), "unknown"
            )
            initialized_policy = nnx.merge(graphdef, theta_0, fixed_state)
            initialized_predictions = jax.vmap(initialized_policy)(
                initialization_states
            )
            jax.block_until_ready(initialized_predictions)
            initialization_investment_log_error_max = float(
                jnp.max(
                    jnp.abs(
                        jnp.log(
                            initialized_predictions[:, 0] / initialization_targets[:, 0]
                        )
                    )
                )
            )
            initialization_current_account_error_max = float(
                jnp.max(
                    jnp.abs(
                        initialized_predictions[:, 1] - initialization_targets[:, 1]
                    )
                )
            )
            if not math.isfinite(initialization_loss) or (
                initialization_loss > init_set.loss_threshold
            ):
                raise RuntimeError(
                    "deterministic policy initialization did not meet its loss "
                    f"threshold: {initialization_loss:.3e}"
                )

    train_noise_key = jax.random.fold_in(train_ic_key, 1)
    test_noise_key = jax.random.fold_in(test_ic_key, 1)
    tvc_noise_key = jax.random.fold_in(test_ic_key, 3)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        fixed_state=fixed_state,
        initial_states=train_initial_states,
        initial_center=initial_center,
        initial_scales=train_scales,
        noise=sample_trajectory_noise(
            solver,
            train_noise_key,
            ic_set.num_train_trajectories,
            n_train_steps,
            train_dt,
            dtype,
        ),
        epoch_key=jax.random.fold_in(train_ic_key, 2),
        epoch=jnp.asarray(0, dtype=jnp.int32),
        shock_scales=shock_scales,
        shock_scale=shock_scales[0],
        train_loss_threshold=jnp.asarray(stop_set.train_loss_threshold, dtype=dtype),
        epoch_steps=jnp.asarray(stop_set.max_steps, dtype=jnp.int32),
        total_steps=jnp.asarray(
            stop_set.num_epochs * stop_set.max_steps, dtype=jnp.int32
        ),
        solver=solver,
        data_set=data_set,
    )
    lm_solver = LevenbergMarquardt(
        residual_fn,
        init_damping=opt_set.init_damping,
        damping_decrease=opt_set.damping_decrease,
        damping_increase=opt_set.damping_increase,
        geodesic_acceleration=opt_set.geodesic_acceleration,
        geodesic_acceptance_ratio=opt_set.geodesic_acceptance_ratio,
    )
    callback = EpochCallback(
        num_train_trajectories=ic_set.num_train_trajectories,
        n_steps=n_train_steps,
        dt=train_dt,
        epoch_early_stopping=stop_set.epoch_early_stopping,
        resample_initial_states=any(value > 0.0 for value in train_variances),
    )
    atol = math.sqrt(stop_set.train_loss_threshold) if stop_set.num_epochs == 1 else 0.0

    start_time = time.perf_counter()
    result = lm_solver.solve(
        theta_0,
        residual_args,
        p=p,
        max_steps=stop_set.num_epochs * stop_set.max_steps,
        atol=atol,
        callback=callback,
        multi_start=MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=DrawSequentialPolicy(nn_set, dtype),
            accept=accept_training,
        ),
        max_steps_is_success=False,
    )
    jax.block_until_ready(result)
    train_time = time.perf_counter() - start_time
    total_fit_time = time.perf_counter() - fit_start
    setup_time = start_time - fit_start
    theta_final = result.x
    solve_args = result.args
    policy = nnx.merge(graphdef, theta_final, fixed_state)
    train_residual = residual_fn(theta_final, solve_args, p)
    train_loss = float(jnp.sum(train_residual * train_residual))
    evaluation_start = time.perf_counter()

    train_stage_parameters = p.replace(
        sigma_z=p.sigma_z * solve_args.shock_scale,
        sigma_r=p.sigma_r * solve_args.shock_scale,
    )
    train_solution = collocation_rollout(
        solver,
        policy,
        solve_args.initial_states,
        solve_args.noise,
        train_stage_parameters,
        data_set,
    )
    train_values = trajectory_values(policy, train_solution.xs, train_stage_parameters)
    test_noise = sample_trajectory_noise(
        solver,
        test_noise_key,
        ic_set.num_test_trajectories,
        n_test_steps,
        test_dt,
        dtype,
    )
    test_solution = rollout_steps(
        solver,
        policy,
        test_initial_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    jax.block_until_ready((train_solution, train_values, test_solution))
    test_states = jnp.where(test_solution.ok[:, None, None], test_solution.xs, jnp.nan)
    test_values = trajectory_values(policy, test_states, p)
    jax.block_until_ready(test_values)

    tvc_evaluation_start = time.perf_counter()
    tvc_initial_states = jnp.broadcast_to(
        initial_center,
        (ic_set.num_tvc_trajectories, initial_center.size),
    )
    tvc_noise = sample_trajectory_noise(
        solver,
        tvc_noise_key,
        ic_set.num_tvc_trajectories,
        n_tvc_steps,
        tvc_dt,
        dtype,
    )
    tvc_solution = rollout_steps(
        solver,
        policy,
        tvc_initial_states,
        tvc_noise,
        data_set.tvc_T,
        n_tvc_steps,
        p,
    )
    tvc_states = jnp.where(tvc_solution.ok[:, None, None], tvc_solution.xs, jnp.nan)
    tvc_values = transversality_trajectory_values(
        policy,
        tvc_states,
        tvc_solution.ts,
        p,
    )
    tvc_statistics = transversality_statistics(tvc_values)
    jax.block_until_ready((tvc_solution, tvc_values, tvc_statistics))
    tvc_evaluation_time = time.perf_counter() - tvc_evaluation_start

    train_df = trajectory_frame(train_values, train_solution.ts)
    test_df = trajectory_frame(test_values, test_solution.ts)
    tvc_df = trajectory_frame(tvc_values, tvc_solution.ts)
    expected_tvc_df = transversality_expectation_frame(tvc_values, tvc_solution.ts)
    train_euler = jnp.stack([train_values["c_euler"], train_values["q_euler"]], axis=-1)
    test_euler = jnp.stack([test_values["c_euler"], test_values["q_euler"]], axis=-1)
    test_mean_abs = jnp.mean(jnp.abs(test_euler), axis=(0, 1))
    test_max_abs = jnp.max(jnp.abs(test_euler), axis=(0, 1))
    test_p95_abs = jnp.quantile(jnp.abs(test_euler), 0.95, axis=(0, 1))
    test_diffusion = jnp.stack(
        [
            test_values["c_diffusion_term"],
            test_values["q_diffusion_term"],
        ],
        axis=-1,
    )
    test_diffusion_max_abs = jnp.max(jnp.abs(test_diffusion), axis=(0, 1))
    test_diffusion_p95_abs = jnp.quantile(jnp.abs(test_diffusion), 0.95, axis=(0, 1))
    test_terminal_capital_mean = jnp.mean(test_values["k"][:, -1])
    terminal_capital_relative_error = (
        jnp.asarray(jnp.nan, dtype=dtype)
        if reference is None
        else jnp.abs(test_terminal_capital_mean / deterministic_terminal_capital - 1.0)
    )
    train_path_ok = bool(
        jnp.all(train_solution.ok)
        and jnp.all(jnp.isfinite(train_solution.xs))
        and jnp.all(jnp.isfinite(train_euler))
        and jnp.min(train_values["k"]) > 0.0
        and jnp.min(train_values["q"]) > 0.0
        and jnp.min(train_values["c"]) > 0.0
        and jnp.min(train_values["x"]) > 0.0
    )
    test_path_ok = bool(
        jnp.all(test_solution.ok)
        and jnp.all(jnp.isfinite(test_states))
        and jnp.all(jnp.isfinite(test_euler))
        and jnp.min(test_values["k"]) > 0.0
        and jnp.min(test_values["q"]) > 0.0
        and jnp.min(test_values["c"]) > 0.0
        and jnp.min(test_values["x"]) > 0.0
    )
    tvc_feasible_observations = (
        jnp.isfinite(tvc_values["discounted_marginal_utility"])
        & jnp.isfinite(tvc_values["absolute_asset_tvc"])
        & jnp.isfinite(tvc_values["capital_tvc"])
        & (tvc_values["k"] > 0.0)
        & (tvc_values["q"] > 0.0)
        & (tvc_values["c"] > 0.0)
        & (tvc_values["x"] > 0.0)
    )
    tvc_paths_ok = bool(
        jnp.all(tvc_solution.ok)
        and jnp.all(jnp.isfinite(tvc_states))
        and jnp.all(tvc_feasible_observations)
    )

    paired_states = jnp.broadcast_to(
        initial_center,
        (ic_set.num_test_trajectories, initial_center.size),
    )
    asset_shift = asset_shift_scale * output_scale
    low_states = paired_states.at[:, 2].set((a_0 - asset_shift) / output_scale)
    high_states = paired_states.at[:, 2].set((a_0 + asset_shift) / output_scale)
    low_solution = rollout_steps(
        solver,
        policy,
        low_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    high_solution = rollout_steps(
        solver,
        policy,
        high_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    low_solution_states = jnp.where(
        low_solution.ok[:, None, None], low_solution.xs, jnp.nan
    )
    high_solution_states = jnp.where(
        high_solution.ok[:, None, None], high_solution.xs, jnp.nan
    )
    low_values = trajectory_values(policy, low_solution_states, p)
    high_values = trajectory_values(policy, high_solution_states, p)
    jax.block_until_ready((low_values, high_values))

    low_euler = jnp.stack([low_values["c_euler"], low_values["q_euler"]], axis=-1)
    high_euler = jnp.stack([high_values["c_euler"], high_values["q_euler"]], axis=-1)
    hysteresis_euler = jnp.concatenate([low_euler, high_euler], axis=0)
    hysteresis_euler_mean_abs = jnp.mean(jnp.abs(hysteresis_euler), axis=(0, 1))
    hysteresis_euler_max_abs = jnp.max(jnp.abs(hysteresis_euler), axis=(0, 1))
    hysteresis_paths_ok = bool(
        jnp.all(low_solution.ok)
        and jnp.all(high_solution.ok)
        and jnp.all(jnp.isfinite(low_solution_states))
        and jnp.all(jnp.isfinite(high_solution_states))
        and jnp.all(jnp.isfinite(low_euler))
        and jnp.all(jnp.isfinite(high_euler))
        and jnp.min(low_values["k"]) > 0.0
        and jnp.min(high_values["k"]) > 0.0
        and jnp.min(low_values["q"]) > 0.0
        and jnp.min(high_values["q"]) > 0.0
        and jnp.min(low_values["c"]) > 0.0
        and jnp.min(high_values["c"]) > 0.0
        and jnp.min(low_values["x"]) > 0.0
        and jnp.min(high_values["x"]) > 0.0
    )

    numerical_rejection_reasons = []
    economic_rejection_reasons = []
    solver_accepted = bool(result.multi_start.accepted)
    if not solver_accepted or train_loss > stop_set.train_loss_threshold:
        numerical_rejection_reasons.append("train_loss")
    if not train_path_ok:
        numerical_rejection_reasons.append("train_path")
    if not test_path_ok:
        numerical_rejection_reasons.append("test_path")
    if not hysteresis_paths_ok:
        numerical_rejection_reasons.append("hysteresis_path")
    if bool(
        jnp.any(hysteresis_euler_mean_abs > stop_set.test_euler_mean_abs_threshold)
    ):
        numerical_rejection_reasons.append("hysteresis_euler_mean")
    if bool(jnp.any(hysteresis_euler_max_abs > stop_set.test_euler_max_threshold)):
        numerical_rejection_reasons.append("hysteresis_euler_max")
    if reference is not None and (
        float(jnp.min(train_values["k"])) < reference.k_lo
        or float(jnp.max(train_values["k"])) > reference.k_hi
        or float(jnp.min(test_values["k"])) < reference.k_lo
        or float(jnp.max(test_values["k"])) > reference.k_hi
        or float(jnp.min(low_values["k"])) < reference.k_lo
        or float(jnp.max(low_values["k"])) > reference.k_hi
        or float(jnp.min(high_values["k"])) < reference.k_lo
        or float(jnp.max(high_values["k"])) > reference.k_hi
    ):
        economic_rejection_reasons.append("reference_domain")
    if bool(jnp.any(test_mean_abs > stop_set.test_euler_mean_abs_threshold)):
        numerical_rejection_reasons.append("test_euler_mean")
    if bool(jnp.any(test_max_abs > stop_set.test_euler_max_threshold)):
        numerical_rejection_reasons.append("test_euler_max")
    if bool(
        jnp.any(test_diffusion_max_abs > stop_set.test_diffusion_max_abs_threshold)
    ):
        economic_rejection_reasons.append("test_diffusion_max")
    if reference is not None and bool(
        terminal_capital_relative_error
        > stop_set.test_terminal_capital_relative_threshold
    ):
        economic_rejection_reasons.append("test_terminal_capital")
    numerical_solution_accepted = not numerical_rejection_reasons
    economic_solution_accepted = not economic_rejection_reasons
    rejection_reasons = numerical_rejection_reasons + economic_rejection_reasons
    solution_accepted = numerical_solution_accepted and economic_solution_accepted

    theta_flat, _ = ravel_pytree(theta_final)
    observed_assets = np.asarray(test_values["a"]).reshape(-1)
    observed_rates = np.asarray(test_values["interest_rate"]).reshape(-1)
    test_asset_change = test_values["a"][:, -1] - test_values["a"][:, 0]
    hysteresis_terminal_asset_difference = jnp.mean(
        high_values["a"][:, -1] - low_values["a"][:, -1]
    )
    results = {
        **asdict(parameters),
        "variant": "stochastic_recursive_open_economy_nn",
        "state_coordinate": "(log z, log k, a/output_scale, rate_gap)",
        "policy": "investment_current_account",
        "policy_initialization": (
            "deterministic_reference_pretraining"
            if init_set.enabled
            else "flat_replacement_investment_zero_current_account"
        ),
        "deterministic_reference": deterministic_reference,
        "initial_investment_anchor": investment_anchor,
        "initial_current_account_anchor": current_account_anchor,
        "raw_initial_investment_relative_range": (raw_investment_relative_range),
        "raw_initial_current_account_max_abs": raw_current_account_max_abs,
        "initialization_loss_before": initialization_loss_before,
        "initialization_loss": initialization_loss,
        "initialization_loss_threshold": init_set.loss_threshold,
        "initialization_steps": initialization_steps,
        "initialization_status": initialization_status,
        "initialization_time": initialization_time,
        "initialization_investment_log_error_max": (
            initialization_investment_log_error_max
        ),
        "initialization_current_account_error_max": (
            initialization_current_account_error_max
        ),
        "reference_k_lo": reference_k_lo,
        "reference_k_hi": reference_k_hi,
        "reference_degree": reference_degree,
        "reference_approximation_max_abs_x": (
            np.nan if reference is None else reference.approximation_max_abs_x
        ),
        "reference_approximation_max_abs_current_account": (
            np.nan
            if reference is None
            else reference.approximation_max_abs_current_account
        ),
        "reference_setup_time": reference_setup_time,
        "deterministic_terminal_capital": deterministic_terminal_capital,
        "deterministic_terminal_asset": deterministic_terminal_asset,
        "k_0": k_0,
        "a_0": a_0,
        "z_0": z_0,
        "rate_gap_0": rate_gap_0,
        "interest_rate_mean": parameters.interest_rate,
        "kappa_z": kappa_z,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "kappa_r": kappa_r,
        "sigma_r": sigma_r,
        "world_rate_ar1_rho": WORLD_RATE_AR1_RHO,
        "world_rate_ar1_innovation_sd": WORLD_RATE_AR1_INNOVATION_SD,
        "corr_zr": corr_zr,
        "rate_ergodic_sd": math.sqrt(rate_ergodic_variance),
        "z_ergodic_sd": math.sqrt(z_ergodic_variance),
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "sde_solver": sde_solver,
        "train_loss_threshold": stop_set.train_loss_threshold,
        "train_loss": train_loss,
        "train_euler_mean_abs": float(jnp.mean(jnp.abs(train_euler))),
        "train_euler_max_abs": float(jnp.max(jnp.abs(train_euler))),
        "test_c_euler_mean_abs": float(test_mean_abs[0]),
        "test_q_euler_mean_abs": float(test_mean_abs[1]),
        "test_c_euler_max_abs": float(test_max_abs[0]),
        "test_q_euler_max_abs": float(test_max_abs[1]),
        "test_c_euler_p95_abs": float(test_p95_abs[0]),
        "test_q_euler_p95_abs": float(test_p95_abs[1]),
        "test_euler_mean_abs_threshold": stop_set.test_euler_mean_abs_threshold,
        "test_euler_max_threshold": stop_set.test_euler_max_threshold,
        "test_c_diffusion_max_abs": float(test_diffusion_max_abs[0]),
        "test_q_diffusion_max_abs": float(test_diffusion_max_abs[1]),
        "test_c_diffusion_p95_abs": float(test_diffusion_p95_abs[0]),
        "test_q_diffusion_p95_abs": float(test_diffusion_p95_abs[1]),
        "test_diffusion_max_abs_threshold": (stop_set.test_diffusion_max_abs_threshold),
        "min_test_consumption": float(jnp.min(test_values["c"])),
        "min_test_interest_rate": float(np.min(observed_rates)),
        "test_negative_rate_fraction": float(np.mean(observed_rates < 0.0)),
        "test_asset_p025": float(np.quantile(observed_assets, 0.025)),
        "test_asset_p50": float(np.quantile(observed_assets, 0.5)),
        "test_asset_p975": float(np.quantile(observed_assets, 0.975)),
        "test_terminal_asset_mean": float(jnp.mean(test_values["a"][:, -1])),
        "test_terminal_asset_sd": float(jnp.std(test_values["a"][:, -1])),
        "test_terminal_capital_mean": float(test_terminal_capital_mean),
        "test_terminal_capital_sd": float(jnp.std(test_values["k"][:, -1])),
        "test_terminal_capital_relative_error": float(terminal_capital_relative_error),
        "test_terminal_capital_relative_threshold": (
            stop_set.test_terminal_capital_relative_threshold
        ),
        "test_min_capital": float(jnp.min(test_values["k"])),
        "test_max_capital": float(jnp.max(test_values["k"])),
        "test_asset_change_mean": float(jnp.mean(test_asset_change)),
        "test_asset_change_sd": float(jnp.std(test_asset_change)),
        "test_terminal_current_account_mean": float(
            jnp.mean(test_values["current_account"][:, -1])
        ),
        "tvc_definition": (
            "E[exp(-preference_rate*T)*c_T^(-gamma)*|a_T|] and "
            "E[exp(-preference_rate*T)*c_T^(-gamma)*q_T*k_T]"
        ),
        "tvc_T": data_set.tvc_T,
        "tvc_points": data_set.tvc_points,
        "num_tvc_trajectories": ic_set.num_tvc_trajectories,
        "tvc_paths_ok": tvc_paths_ok,
        "tvc_solver_path_fraction": float(jnp.mean(tvc_solution.ok.astype(dtype))),
        "tvc_feasible_observation_fraction": float(
            jnp.mean(tvc_feasible_observations.astype(dtype))
        ),
        "min_tvc_consumption": float(jnp.nanmin(tvc_values["c"])),
        "min_tvc_capital": float(jnp.nanmin(tvc_values["k"])),
        "ex_post_expected_absolute_asset_tvc": float(
            tvc_statistics["expected_absolute_asset_tvc"]
        ),
        "ex_post_expected_capital_tvc": float(tvc_statistics["expected_capital_tvc"]),
        "ex_post_signed_expected_asset_tvc": float(
            tvc_statistics["signed_expected_asset_tvc"]
        ),
        "ex_post_absolute_asset_tvc_mc_se": float(
            tvc_statistics["absolute_asset_tvc_mc_se"]
        ),
        "ex_post_signed_asset_tvc_mc_se": float(
            tvc_statistics["signed_asset_tvc_mc_se"]
        ),
        "ex_post_capital_tvc_mc_se": float(tvc_statistics["capital_tvc_mc_se"]),
        "ex_post_log_expected_absolute_asset_tvc": float(
            tvc_statistics["log_expected_absolute_asset_tvc"]
        ),
        "ex_post_log_expected_capital_tvc": float(
            tvc_statistics["log_expected_capital_tvc"]
        ),
        "ex_post_max_log_absolute_asset_tvc": float(
            tvc_statistics["max_log_absolute_asset_tvc"]
        ),
        "ex_post_max_log_capital_tvc": float(tvc_statistics["max_log_capital_tvc"]),
        "tvc_evaluation_time": tvc_evaluation_time,
        "asset_shift": asset_shift,
        "hysteresis_terminal_k_mean_difference": float(
            jnp.mean(high_values["k"][:, -1] - low_values["k"][:, -1])
        ),
        "hysteresis_terminal_c_mean_difference": float(
            jnp.mean(high_values["c"][:, -1] - low_values["c"][:, -1])
        ),
        "hysteresis_terminal_current_account_mean_difference": float(
            jnp.mean(
                high_values["current_account"][:, -1]
                - low_values["current_account"][:, -1]
            )
        ),
        "hysteresis_terminal_asset_mean_difference": float(
            hysteresis_terminal_asset_difference
        ),
        "hysteresis_asset_persistence_ratio": float(
            hysteresis_terminal_asset_difference / (2.0 * asset_shift)
        ),
        "hysteresis_paths_ok": hysteresis_paths_ok,
        "hysteresis_c_euler_mean_abs": float(hysteresis_euler_mean_abs[0]),
        "hysteresis_q_euler_mean_abs": float(hysteresis_euler_mean_abs[1]),
        "hysteresis_c_euler_max_abs": float(hysteresis_euler_max_abs[0]),
        "hysteresis_q_euler_max_abs": float(hysteresis_euler_max_abs[1]),
        "steps": int(result.steps),
        "epochs_run": int(solve_args.epoch) + 1,
        "max_steps_per_epoch": stop_set.max_steps,
        "num_epochs": stop_set.num_epochs,
        "shock_continuation_scales": tuple(
            float(value) for value in np.asarray(shock_scales)
        ),
        "final_training_shock_scale": float(solve_args.shock_scale),
        "attempt": int(result.multi_start.attempt) + 1,
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "final_damping": float(result.lm_state.damping),
        "train_time": train_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "trainable_params": int(theta_flat.size),
        "reference_coefficient_count": (
            0 if reference is None else int(reference.coefficients.size)
        ),
        "residual_count": int(train_residual.size),
        "parameter_to_residual_ratio": float(theta_flat.size / train_residual.size),
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        "train_points": data_set.train_points,
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
        "train_path_ok": train_path_ok,
        "test_path_ok": test_path_ok,
        "solver_accepted": solver_accepted,
        "numerical_solution_accepted": numerical_solution_accepted,
        "economic_solution_accepted": economic_solution_accepted,
        "solution_accepted": solution_accepted,
        "acceptance_scope": "numerical_and_economic_gates",
        "numerical_rejection_reason": (
            "accepted"
            if numerical_solution_accepted
            else ",".join(numerical_rejection_reasons)
        ),
        "economic_rejection_reason": (
            "accepted"
            if economic_solution_accepted
            else ",".join(economic_rejection_reasons)
        ),
        "rejection_reason": (
            "accepted" if solution_accepted else ",".join(rejection_reasons)
        ),
        "lm_linear_system": "default_auto_dual_gram_when_overparameterized",
        "parameter_derivative": (
            "minimum_euclidean_norm_svd_conditional_on_fixed_reference_"
            "pretraining_and_carried_training_batch"
        ),
    }
    results["postfit_evaluation_time"] = time.perf_counter() - evaluation_start
    diagnostic_start = time.perf_counter()
    if diagnose_jacobian and bool(jnp.all(jnp.isfinite(train_residual))):
        results.update(
            residual_jacobian_diagnostics(
                theta_final,
                solve_args,
                p,
                float(result.lm_state.damping),
            )
        )
    else:
        results.update(
            jacobian_sigma_max=np.nan,
            jacobian_sigma_min=np.nan,
            jacobian_sigma_min_retained=np.nan,
            jacobian_condition_number=np.nan,
            undamped_gram_condition_number=np.nan,
            damped_gram_condition_number=np.nan,
            jacobian_effective_rank=0,
            jacobian_nullity=np.nan,
            jacobian_row_rank_deficiency=np.nan,
            jacobian_row_count=int(train_residual.size),
            jacobian_column_count=int(theta_flat.size),
        )
    results["jacobian_diagnostic_time"] = time.perf_counter() - diagnostic_start
    results["total_run_time"] = time.perf_counter() - driver_start

    hyperparameters = {
        "parameters": asdict(parameters),
        "shocks": {
            "kappa_z": kappa_z,
            "sigma_z": sigma_z,
            "z_bar": z_bar,
            "kappa_r": kappa_r,
            "sigma_r": sigma_r,
            "corr_zr": corr_zr,
        },
        "deterministic_reference": {
            "enabled": deterministic_reference,
            "k_lo": reference_k_lo,
            "k_hi": reference_k_hi,
            "degree": reference_degree,
        },
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "nn_set": asdict(nn_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
        "init_set": asdict(init_set),
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")
    return results, {
        "model": policy,
        "parameters": p,
        "train_initial_states": solve_args.initial_states,
        "test_initial_states": test_initial_states,
        "train_df": train_df,
        "test_df": test_df,
        "tvc_df": tvc_df,
        "expected_tvc_df": expected_tvc_df,
        "asset_low_df": trajectory_frame(low_values, low_solution.ts),
        "asset_high_df": trajectory_frame(high_values, high_solution.ts),
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jsonargparse.CLI(stochastic_open_economy_growth_nn_recursive)
