from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import LevenbergMarquardt, LMAction, LMStatus, MultiStart
from tinydiffeq import SRA1, EulerMaruyama, Milstein, SaveAt, solve_sde

from neoclassical_growth_nn_recursive import (
    policy_cache_key,
    standardized_initial_states,
)
from open_economy_growth import ModelParameters, marginal_product, production
from open_economy_growth_benchmark import open_economy_growth_benchmark
from stochastic_growth_nn_recursive import sample_trajectory_noise
from utilities import (
    ACTIVATIONS,
    NEOCLASSICAL_GROWTH_CONSTANTS,
    STATUS_TO_REASON,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    unit_softplus,
)

# Recursive stochastic small open economy on (log z, log k, a/output_scale,
# rate_gap). No benchmark or steady-state information enters the residual or
# initialization; the deterministic benchmark is ex-post diagnostics only.
DOMAIN_EPS = 1e-8

# Annual AR(1) -> OU: e^(-kappa_r) = rho, sigma_r^2/(2 kappa_r) = sd^2/(1-rho^2).
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

# The residual differentiates the policy twice, so no kinked activation.
SMOOTH_ACTIVATIONS = frozenset(ACTIVATIONS) - {"relu"}


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
    # Fixed-step SDE rollout on the integrator's own states; train_T = 40
    # covers ~5 half-lives of the capital transition (stable root ~ -0.089).
    train_points: int = 65
    test_points: int = 51
    tvc_points: int = 161
    train_T: float = 40.0
    test_T: float = 50.0
    tvc_T: float = 160.0


@dataclass(frozen=True)
class InitialConditionSettings:
    # Spreads are absolute, applied additively per coordinate (never
    # proportional to the center: that degenerates at a_0 = 0). Default is a
    # single initial condition -- trajectories differ only by their shock
    # realization; dispersion is an experiment. None test values inherit train.
    data_seed: int = 0
    num_train_trajectories: int = 4
    num_test_trajectories: int = 8
    num_tvc_trajectories: int = 128
    train_log_z_variance: float = 0.0
    train_log_k_variance: float = 0.0
    train_asset_variance: float = 0.0
    train_rate_variance: float = 0.0
    test_log_z_variance: float | None = None
    test_log_k_variance: float | None = None
    test_asset_variance: float | None = None
    test_rate_variance: float | None = None


@dataclass(frozen=True)
class StoppingCriteria:
    # max_steps is the per-epoch LM budget; noise (and dispersed ICs) redraw
    # at epoch boundaries. The shock ladder rises geometrically to 1.0 over
    # continuation_epochs; convergence is only declared at scale 1.0.
    max_steps: int = 250
    train_loss_threshold: float = 1e-10
    num_attempts: int = 4
    num_epochs: int = 3
    epoch_early_stopping: bool = True
    shock_continuation: bool = True
    initial_shock_scale: float = 0.25
    continuation_epochs: int = 2


# Acceptance gates; the float32 replication driver passes its own bar.
@dataclass(frozen=True)
class ToleranceSettings:
    test_euler_mean_abs: float = 5e-3
    test_euler_max: float = 5e-2
    test_diffusion_max_abs: float = 5e-2


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


# ICs, noise, keys, and ladder position live in args, so the implicit
# derivative is conditional on the carried batch.
@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    initial_states: jax.Array
    initial_center: jax.Array
    initial_scales: jax.Array
    noise: object
    epoch_key: jax.Array
    restart_key: jax.Array
    epoch: jax.Array
    shock_scales: jax.Array
    shock_scale: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    solver: object = struct.field(pytree_node=False)
    data_set: DataSettings = struct.field(pytree_node=False)


# At the draw only log k feeds the net (investment ~ delta k_0, CA = 0);
# the z/asset/rate columns are zero at the draw, not frozen.
def capital_only_kernel_init(key, shape, dtype):
    kernel = nnx.initializers.lecun_normal()(key, shape, dtype)
    mask = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=dtype)[:, None]
    return kernel * mask


# Shared with the kernel policy; the asset squash is bounded in (-2, 2) and
# ~95% saturated near |u| ~ 3.7.
def normalize_state(state, center, scale):
    u = (state - center) / scale
    return u.at[2].set(2.0 * jnp.tanh(u[2] / 2.0))


# Shared with the kernel policy. The pre-scaling gives both heads the same
# O(output_scale) Jacobian; dropping it shrinks the investment rows ~16x.
def policy_outputs(raw, investment_anchor, current_account_anchor, output_scale):
    investment = investment_anchor * unit_softplus(
        (output_scale / investment_anchor) * raw[0]
    )
    current_account = current_account_anchor + output_scale * raw[1]
    return jnp.stack([investment, current_account])


# Save/load contract: only nnx.Param leaves persist, so every non-Param
# Variable must derive from (settings, dtype) or reload breaks silently.
class SequentialPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: SequentialPolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.state_center = nnx.Variable(
            jnp.asarray(
                [
                    settings.log_z_center,
                    settings.log_k_center,
                    settings.asset_center,
                    0.0,
                ],
                dtype=dtype,
            )
        )
        self.state_scale = nnx.Variable(
            jnp.asarray(
                [
                    settings.log_z_scale,
                    1.0,
                    settings.asset_scale,
                    settings.rate_scale,
                ],
                dtype=dtype,
            )
        )
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
        h = normalize_state(state, self.state_center[...], self.state_scale[...])
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        raw = jnp.stack([self.investment_head(h)[0], self.current_account_head(h)[0]])
        return policy_outputs(
            raw,
            self.investment_anchor[...],
            self.current_account_anchor[...],
            self.output_scale[...],
        )


# DrawNNXModule splits on nnx.Param alone -- non-exhaustive here. Restarts
# re-chain the epoch-key stream from restart_key; multi-start chains each
# attempt's initial args, so the ladder restarts at epoch 0.
class DrawSequentialPolicy:
    def __init__(self, settings, dtype):
        self.settings = settings
        self.dtype = dtype

    def __call__(self, key, _, args):
        module = SequentialPolicyNN(
            self.settings,
            dtype=self.dtype,
            rngs=nnx.Rngs(key),
        )
        _, theta, _ = nnx.split(module, nnx.Param, ...)
        restart_key, epoch_key = jax.random.split(args.restart_key)
        return theta, args.replace(
            epoch_key=epoch_key,
            restart_key=restart_key,
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def __hash__(self):
        return hash((DrawSequentialPolicy, self.settings, self.dtype))

    def __eq__(self, other):
        return (
            isinstance(other, DrawSequentialPolicy)
            and self.settings == other.settings
            and self.dtype == other.dtype
        )


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


# One covariance, written twice: diagonal diffusion + pre-mixed noise
# (correlate_noise) in the rollout, explicit direction rows
# (diffusion_directions) in the residual. Edit them together only; the
# directional-generator test guards the pairing.
def state_diffusion(state, p):
    zero = jnp.zeros_like(state[0])
    return jnp.stack([p.sigma_z, zero, zero, p.sigma_r])


# Residual half of the pairing: directions.T @ directions reproduces the
# rollout's mixed-noise covariance exactly.
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


# With l = log c, v = log q, L the generator, Gamma the diffusion covariance
# product, and r = interest_rate_mean + rate_gap:
#   L l - gamma Gamma(l, l)/2 = (r - rho)/gamma
#   L v + Gamma(v, v)/2 - gamma Gamma(v, l)
#     = r + delta - (1 - eta) chi s^eta - MPK/q
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


# Leaf-wise over the solver noise pytree (SRA1 carries (d_w, d_z)): valid
# because each leaf is independent Gaussian and the mixing is linear.
def correlate_noise(noise, corr_zr):
    orthogonal_loading = jnp.sqrt(1.0 - corr_zr**2)

    def transform(increments):
        rate_increments = (
            corr_zr * increments[..., 0] + orthogonal_loading * increments[..., 3]
        )
        return increments.at[..., 3].set(rate_increments)

    return jax.tree.map(transform, noise)


# Interior epoch boundaries redraw the batch, step the ladder, and stop on a
# nonfinite or (full-scale) converged fresh-batch loss.
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
            # The stochastic axes' IC spread rides the shock ladder: partial-
            # scale epochs draw z / rate starts with proportionally shrunk
            # dispersion (inert at the zero-variance default).
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
                4,
            ),
            epoch_key=epoch_key,
            epoch=epoch,
            shock_scale=shock_scale,
        )
        residual = residual_fn(ctx.x, fresh, ctx.p)
        loss = jnp.sum(residual * residual)
        nonfinite = ~jnp.isfinite(loss)
        if self.epoch_early_stopping:
            converged = (shock_scale == 1.0) & (loss < fresh.train_loss_threshold)
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


def drift_field(state, _, args, p):
    policy, structural = p
    return state_drift(policy, state, structural)


def diffusion_field(state, _, args, p):
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
    # Mean-squared normalization; a failed trajectory is a loud inf, never a
    # fitted prefix.
    count = jnp.asarray(residuals.size, dtype=residuals.dtype)
    return jnp.where(
        solution.ok[:, None, None],
        residuals / jnp.sqrt(count),
        jnp.full_like(residuals, jnp.inf),
    )


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.fixed_state)
    # The ladder scales both sigmas through p (rollout and residual agree);
    # corr_zr is scale-free.
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
    return {
        "expected_absolute_asset_tvc": jnp.mean(absolute_asset_tvc),
        "expected_capital_tvc": jnp.mean(capital_tvc),
        "signed_expected_asset_tvc": jnp.mean(signed_asset_tvc),
        "absolute_asset_tvc_mc_se": monte_carlo_standard_error(absolute_asset_tvc),
        "signed_asset_tvc_mc_se": monte_carlo_standard_error(signed_asset_tvc),
        "capital_tvc_mc_se": monte_carlo_standard_error(capital_tvc),
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


# Problem half of the two-part artifact key, shared with the kernel-recursive
# consumer. Test/tvc/tolerance settings enter neither hash.
def problem_cache_config(
    parameters,
    kappa_z,
    sigma_z,
    z_bar,
    kappa_r,
    sigma_r,
    corr_zr,
    k_0,
    a_0,
    z_0,
    rate_gap_0,
    use_float64,
    num_train_trajectories,
    train_variances,
):
    return {
        **asdict(parameters),
        "kappa_z": kappa_z,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "kappa_r": kappa_r,
        "sigma_r": sigma_r,
        "corr_zr": corr_zr,
        "k_0": k_0,
        "a_0": a_0,
        "z_0": z_0,
        "rate_gap_0": rate_gap_0,
        "use_float64": use_float64,
        "train_ic": {
            "num_train_trajectories": num_train_trajectories,
            "train_log_z_variance": train_variances[0],
            "train_log_k_variance": train_variances[1],
            "train_asset_variance": train_variances[2],
            "train_rate_variance": train_variances[3],
        },
    }


# One .npz per policy: Param state plus a JSON meta block with the resolved
# settings, so a load is rollout-ready without the training driver.
def save_sequential_policy(path, policy, nn_set, *, structural, initial_state, extra):
    def flatten(prefix, node, out):
        if isinstance(node, dict):
            for key, value in node.items():
                flatten((*prefix, str(key)), value, out)
        else:
            out["/".join(prefix)] = np.asarray(node)
        return out

    arrays = flatten((), nnx.state(policy, nnx.Param).to_pure_dict(), {})
    meta = {
        "settings": asdict(nn_set),
        "dtype": str(next(iter(arrays.values())).dtype),
        "structural": structural,
        "initial_state": initial_state,
        **extra,
    }
    np.savez(path, __meta__=np.asarray(json.dumps(meta)), **arrays)


def load_sequential_policy(path):
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    settings = SequentialPolicySettings(**meta["settings"])
    dtype = jnp.dtype(meta["dtype"])
    module = SequentialPolicyNN(settings, dtype=dtype, rngs=nnx.Rngs(0))
    graphdef, theta, fixed_state = nnx.split(module, nnx.Param, ...)
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
    return nnx.merge(graphdef, theta, fixed_state), meta


# kappa_z / sigma_z reuse the neoclassical family's calibration, so every
# stochastic recursive variant faces the identical z process.
def stochastic_open_economy_growth_nn_recursive(
    k_0: float = 1.0,
    # Debtor default: net debt ~ half of steady-state output (cf. Mendoza &
    # Villalvazo 2020).
    a_0: float = -0.8,
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
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    asset_shift_scale: float = 0.5,
    # Content-addressed artifact dir: an existing artifact for this exact
    # configuration loads instead of training; a fresh solve saves there.
    policy_cache: str | None = None,
    save_policy: str | None = None,
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
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
    if not (
        math.isfinite(stop_set.train_loss_threshold)
        and stop_set.train_loss_threshold > 0.0
    ):
        raise ValueError("train_loss_threshold must be positive")
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
    tolerances = (
        tolerance_set.test_euler_mean_abs,
        tolerance_set.test_euler_max,
        tolerance_set.test_diffusion_max_abs,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("acceptance tolerances must be positive")

    train_variances = (
        ic_set.train_log_z_variance,
        ic_set.train_log_k_variance,
        ic_set.train_asset_variance,
        ic_set.train_rate_variance,
    )
    test_variances = tuple(
        train if test is None else test
        for train, test in zip(
            train_variances,
            (
                ic_set.test_log_z_variance,
                ic_set.test_log_k_variance,
                ic_set.test_asset_variance,
                ic_set.test_rate_variance,
            ),
            strict=True,
        )
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

    z_ergodic_variance = sigma_z**2 / (2.0 * kappa_z)
    rate_ergodic_variance = sigma_r**2 / (2.0 * kappa_r)
    output_scale = float(production(k_0, z_0, parameters))
    initial_center = jnp.asarray(
        [math.log(z_0), math.log(k_0), a_0 / output_scale, rate_gap_0],
        dtype=dtype,
    )
    train_scales = jnp.sqrt(jnp.asarray(train_variances, dtype=dtype))
    test_scales = jnp.sqrt(jnp.asarray(test_variances, dtype=dtype))
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
        # NFA travels ~an output unit per decade, so the squash scale floors
        # at 0.1 * train_T; a unit floor saturates the tanh mid-transition.
        asset_scale=max(math.sqrt(train_variances[2]), 0.1 * data_set.train_T, 1.0),
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

    fit_timer = Stopwatch()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    initial_model = SequentialPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key))
    graphdef, theta_0, fixed_state = nnx.split(initial_model, nnx.Param, ...)
    # fold_in(train_ic_key, 1) is the training noise stream; the kernel
    # consumer regenerates this exact stream to rebuild the frozen design.
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
            4,
        ),
        epoch_key=jax.random.fold_in(train_ic_key, 2),
        restart_key=jax.random.fold_in(train_ic_key, 4),
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
    # <problem>-<run> key: consumers glob *_policy_<problem>-*.npz; the run
    # hash covers training minutiae only.
    problem_config = problem_cache_config(
        parameters,
        kappa_z,
        sigma_z,
        z_bar,
        kappa_r,
        sigma_r,
        corr_zr,
        k_0,
        a_0,
        z_0,
        rate_gap_0,
        use_float64,
        ic_set.num_train_trajectories,
        train_variances,
    )
    run_config = {
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "sde_solver": sde_solver,
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
        "nn_set": asdict(nn_set),
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
        / f"stochastic_open_economy_growth_nn_recursive_policy_{cache_key}.npz"
    )
    cache_hit = cache_path is not None and cache_path.exists()
    if cache_hit:
        policy, cached_meta = load_sequential_policy(cache_path)
        theta_final = nnx.split(policy, nnx.Param, ...)[1]
        # The saved policy is the end of the ladder: diagnostics must run at
        # full shock scale, never the fresh args' fractional scale.
        solve_args = residual_args.replace(
            shock_scale=jnp.asarray(1.0, dtype=dtype),
            epoch=jnp.asarray(stop_set.num_epochs - 1, dtype=jnp.int32),
        )
        steps = 0
        epochs_run = 0
        attempt = 0
        stopping_reason = "cached"
        final_damping = float("nan")
        solver_accepted = bool(cached_meta["solver_accepted"])
        train_time = 0.0
        total_fit_time = fit_timer.elapsed()
        setup_time = total_fit_time
        train_residual = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(train_residual * train_residual))
        if verbose:
            print(f"loaded {cache_path}, train_loss={train_loss:.3e}")
    else:
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
        # Per-step atol is sound only for a never-changing batch; epochs are
        # judged at the boundaries and by the acceptance test.
        atol = (
            math.sqrt(stop_set.train_loss_threshold)
            if stop_set.num_epochs == 1
            else 0.0
        )
        train_timer = Stopwatch()
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
        train_time = train_timer.elapsed(result)
        total_fit_time = fit_timer.elapsed()
        setup_time = total_fit_time - train_time
        theta_final, solve_args = result.x, result.args
        train_residual = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(train_residual * train_residual))
        steps = int(result.steps)
        epochs_run = int(solve_args.epoch) + 1
        attempt = int(result.multi_start.attempt) + 1
        stopping_reason = STATUS_TO_REASON.get(int(result.status), "unknown")
        final_damping = float(result.lm_state.damping)
        solver_accepted = bool(result.multi_start.accepted)
        if verbose:
            print(
                f"attempt {attempt}: {stopping_reason} after {steps} steps "
                f"(epoch {epochs_run}/{stop_set.num_epochs}, shock scale "
                f"{float(solve_args.shock_scale):.3f}), "
                f"train_loss={train_loss:.3e} ({train_time:.2f}s)"
            )
        policy = nnx.merge(graphdef, theta_final, fixed_state)

    # Training diagnostics use the batch the solution was returned on.
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
        4,
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
        4,
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

    # Paired common-noise rollouts from shifted NFA positions exercise the
    # asset axis even at zero IC spread.
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
    hysteresis_euler_mean_abs = float(jnp.mean(jnp.abs(hysteresis_euler)))
    hysteresis_euler_max_abs = float(jnp.max(jnp.abs(hysteresis_euler)))
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
    hysteresis_terminal_asset_difference = float(
        jnp.mean(high_values["a"][:, -1] - low_values["a"][:, -1])
    )

    # Deterministic benchmark: reported ex post, never gated.
    benchmark_grid = np.linspace(0.0, max(200.0, data_set.test_T), 401)
    benchmark = open_economy_growth_benchmark(
        k_0,
        a_0,
        benchmark_grid,
        parameters=parameters,
    )[0]
    benchmark_values = benchmark(np.asarray([data_set.test_T]))
    deterministic_terminal_capital = float(benchmark_values["k"][0])
    deterministic_terminal_asset = float(benchmark_values["a"][0])
    test_terminal_capital_mean = float(jnp.mean(test_values["k"][:, -1]))
    test_terminal_capital_relative_error = abs(
        test_terminal_capital_mean / deterministic_terminal_capital - 1.0
    )

    rejection_reasons = []
    # A cached policy trained on other batches: only its saved solver
    # acceptance applies, never the fresh-batch loss.
    if not solver_accepted or (
        not cache_hit and train_loss > stop_set.train_loss_threshold
    ):
        rejection_reasons.append("train_loss")
    if not train_path_ok:
        rejection_reasons.append("train_path")
    if not test_path_ok:
        rejection_reasons.append("test_path")
    if bool(jnp.any(test_mean_abs > tolerance_set.test_euler_mean_abs)):
        rejection_reasons.append("test_euler_mean")
    if bool(jnp.any(test_max_abs > tolerance_set.test_euler_max)):
        rejection_reasons.append("test_euler_max")
    if bool(jnp.any(test_diffusion_max_abs > tolerance_set.test_diffusion_max_abs)):
        rejection_reasons.append("test_diffusion_max")
    if not hysteresis_paths_ok:
        rejection_reasons.append("hysteresis_path")
    if hysteresis_euler_mean_abs > tolerance_set.test_euler_mean_abs:
        rejection_reasons.append("hysteresis_euler_mean")
    if hysteresis_euler_max_abs > tolerance_set.test_euler_max:
        rejection_reasons.append("hysteresis_euler_max")
    acceptance = acceptance_fields(rejection_reasons)
    solution_accepted = acceptance["solution_accepted"]

    theta_flat, _ = ravel_pytree(theta_final)
    observed_assets = np.asarray(test_values["a"]).reshape(-1)
    observed_rates = np.asarray(test_values["interest_rate"]).reshape(-1)
    test_asset_change = test_values["a"][:, -1] - test_values["a"][:, 0]
    # Near one, the squash saturates and the heads lose asset sensitivity.
    test_normalized_asset = (
        test_states[..., 2] - nn_set.asset_center
    ) / nn_set.asset_scale
    test_asset_squash_saturation = float(
        jnp.nanmax(jnp.abs(jnp.tanh(test_normalized_asset / 2.0)))
    )
    results = {
        **asdict(parameters),
        "variant": "stochastic_recursive_open_economy_nn",
        "state_coordinate": "(log z, log k, a/output_scale, rate_gap)",
        "policy": "investment_current_account",
        "initialization": "replacement_investment_zero_current_account",
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
        "corr_zr": corr_zr,
        "z_ergodic_sd": math.sqrt(z_ergodic_variance),
        "rate_ergodic_sd": math.sqrt(rate_ergodic_variance),
        "seed": seed,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_residual.dtype),
        "sde_solver": sde_solver,
        "train_loss": train_loss,
        "train_euler_mean_abs": float(jnp.mean(jnp.abs(train_euler))),
        "train_euler_max_abs": float(jnp.max(jnp.abs(train_euler))),
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_euler))),
        "test_c_euler_mean_abs": float(test_mean_abs[0]),
        "test_q_euler_mean_abs": float(test_mean_abs[1]),
        "test_c_euler_max_abs": float(test_max_abs[0]),
        "test_q_euler_max_abs": float(test_max_abs[1]),
        "test_c_euler_p95_abs": float(test_p95_abs[0]),
        "test_q_euler_p95_abs": float(test_p95_abs[1]),
        "test_c_diffusion_max_abs": float(test_diffusion_max_abs[0]),
        "test_q_diffusion_max_abs": float(test_diffusion_max_abs[1]),
        "min_test_consumption": float(jnp.min(test_values["c"])),
        "min_test_interest_rate": float(np.min(observed_rates)),
        "test_negative_rate_fraction": float(np.mean(observed_rates < 0.0)),
        "test_asset_p025": float(np.quantile(observed_assets, 0.025)),
        "test_asset_p50": float(np.quantile(observed_assets, 0.5)),
        "test_asset_p975": float(np.quantile(observed_assets, 0.975)),
        "test_asset_squash_saturation": test_asset_squash_saturation,
        "test_terminal_asset_mean": float(jnp.mean(test_values["a"][:, -1])),
        "test_terminal_asset_sd": float(jnp.std(test_values["a"][:, -1])),
        "test_terminal_capital_mean": test_terminal_capital_mean,
        "test_terminal_capital_sd": float(jnp.std(test_values["k"][:, -1])),
        "test_min_capital": float(jnp.min(test_values["k"])),
        "test_max_capital": float(jnp.max(test_values["k"])),
        "test_asset_change_mean": float(jnp.mean(test_asset_change)),
        "test_asset_change_sd": float(jnp.std(test_asset_change)),
        "test_terminal_current_account_mean": float(
            jnp.mean(test_values["current_account"][:, -1])
        ),
        "deterministic_terminal_capital": deterministic_terminal_capital,
        "deterministic_terminal_asset": deterministic_terminal_asset,
        "test_terminal_capital_relative_error": test_terminal_capital_relative_error,
        "tvc_T": data_set.tvc_T,
        "tvc_points": data_set.tvc_points,
        "num_tvc_trajectories": ic_set.num_tvc_trajectories,
        "tvc_paths_ok": tvc_paths_ok,
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
        "asset_shift": asset_shift,
        "hysteresis_terminal_asset_mean_difference": (
            hysteresis_terminal_asset_difference
        ),
        "hysteresis_asset_persistence_ratio": (
            hysteresis_terminal_asset_difference / (2.0 * asset_shift)
        ),
        "hysteresis_paths_ok": hysteresis_paths_ok,
        "hysteresis_euler_mean_abs": hysteresis_euler_mean_abs,
        "hysteresis_euler_max_abs": hysteresis_euler_max_abs,
        "steps": steps,
        "num_epochs": stop_set.num_epochs,
        "epochs_run": epochs_run,
        "shock_continuation_scales": tuple(
            float(value) for value in np.asarray(shock_scales)
        ),
        "final_training_shock_scale": float(solve_args.shock_scale),
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "final_damping": final_damping,
        "policy_cache_key": cache_key,
        "policy_cache_hit": cache_hit,
        "train_time": train_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "train_time_success": train_time if solver_accepted else np.nan,
        "total_params": int(theta_flat.size),
        "residual_count": int(train_residual.size),
        "parameter_to_residual_ratio": float(theta_flat.size / train_residual.size),
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        "train_log_z_variance": train_variances[0],
        "train_log_k_variance": train_variances[1],
        "train_asset_variance": train_variances[2],
        "train_rate_variance": train_variances[3],
        "train_points": data_set.train_points,
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
        "train_path_ok": train_path_ok,
        "test_path_ok": test_path_ok,
        "solver_accepted": solver_accepted,
        "solver_converged": solver_accepted,
        **acceptance,
        "lm_linear_system": "default_auto_dual_gram_when_overparameterized",
        "parameter_derivative": (
            "minimum_euclidean_norm_svd_conditional_on_carried_training_batch"
        ),
    }

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
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "nn_set": asdict(nn_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
        "tolerance_set": asdict(tolerance_set),
        "initial_conditions": {
            "k_0": k_0,
            "a_0": a_0,
            "z_0": z_0,
            "rate_gap_0": rate_gap_0,
        },
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    structural = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "preference_rate": parameters.r,
        "investment_eta": parameters.eta,
        "interest_rate_mean": parameters.interest_rate,
        "chi": parameters.chi,
        "kappa_z": kappa_z,
        "sigma_z": sigma_z,
        "log_z_bar": math.log(z_bar),
        "kappa_r": kappa_r,
        "sigma_r": sigma_r,
        "corr_zr": corr_zr,
        "output_scale": output_scale,
    }
    initial_state = {"k_0": k_0, "a_0": a_0, "z_0": z_0, "rate_gap_0": rate_gap_0}
    policy_extra = {
        "model": "stochastic_open_economy_growth_nn_recursive",
        "problem_key": problem_key,
        "run_key": run_key,
        "cache_key": cache_key,
        "config": {"problem": problem_config, "run": run_config},
        "solver_accepted": solver_accepted,
        "solution_accepted": solution_accepted,
        "train_loss": train_loss,
        "test_euler_mean_abs": results["test_euler_mean_abs"],
    }
    if cache_path is not None and not cache_hit:
        save_sequential_policy(
            cache_path,
            policy,
            nn_set,
            structural=structural,
            initial_state=initial_state,
            extra=policy_extra,
        )
    if save_policy is not None:
        save_sequential_policy(
            save_policy,
            policy,
            nn_set,
            structural=structural,
            initial_state=initial_state,
            extra=policy_extra,
        )
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
    jsonargparse.CLI(stochastic_open_economy_growth_nn_recursive)
