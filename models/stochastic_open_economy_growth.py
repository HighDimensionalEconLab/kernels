from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx, struct
from nlls_gram import DrawNNXModule
from tinydiffeq import SaveAt, solve_sde

from models.utilities import (
    ACTIVATIONS,
    load_model,
    save_model,
    unit_softplus,
)

# Recursive stochastic small open economy on (log z, log k, a/output_scale,
# rate_gap). No benchmark or steady-state information enters the residual or
# initialization.

# Annual AR(1) -> OU: e^(-kappa_r) = rho, sigma_r^2/(2 kappa_r) = sd^2/(1-rho^2).
# World-rate process from Uribe & Yue (2006, eq. 15): quarterly rho = 0.83 on
# log-devs of the annualized gross rate with innovation sd 0.007, so annual
# rho = 0.83^4 and stationary sd ~1.04 * 0.007/sqrt(1-0.83^2) ~ 1.3 percentage
# points in the rate level (kappa ~ 0.75/yr, half-life ~0.9y). Corroborated by
# the international risk-free real rate in Fernandez-Villaverde et al. (2011
# AER "Risk Matters"): monthly rho = 0.95 (kappa ~ 0.62/yr) with a 38bp
# annualized innovation sd, stationary sd ~1.2pp.
WORLD_RATE_AR1_RHO = 0.83**4
WORLD_RATE_AR1_INNOVATION_SD = 0.0115
WORLD_RATE_KAPPA = -math.log(WORLD_RATE_AR1_RHO)
WORLD_RATE_SIGMA = WORLD_RATE_AR1_INNOVATION_SD * math.sqrt(
    2.0 * WORLD_RATE_KAPPA / (1.0 - WORLD_RATE_AR1_RHO**2)
)


# The deterministic sequential counterpart in open_economy_growth.py carries
# the same calibration.
@dataclass(frozen=True)
class ModelParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    r: float = -math.log(0.96)
    eta: float = 0.76
    r_star: float | None = None

    @property
    def interest_rate(self):
        if self.r_star is None:
            return self.r
        return self.r_star

    @property
    def chi(self):
        return self.delta ** (1.0 - self.eta)


def production(k, z, parameters):
    return z * k**parameters.alpha


def marginal_product(k, z, parameters):
    return parameters.alpha * z * k ** (parameters.alpha - 1.0)


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
    # Fixed-step SDE rollout on the integrator's own states; train_T covers
    # ~8 half-lives of the capital transition. Test/tvc dt is 1.0.
    train_points: int = 97
    test_points: int = 71
    tvc_points: int = 161
    train_T: float = 60.0
    test_T: float = 70.0
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


# Acceptance gates; the float32 replication driver passes its own bar.
@dataclass(frozen=True)
class ToleranceSettings:
    test_euler_mean_abs: float = 5e-3
    test_euler_max: float = 5e-2
    test_diffusion_max_abs: float = 5e-2


@dataclass(frozen=True)
class SequentialPolicySettings:
    width: int = 16
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


class SequentialPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: SequentialPolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
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


# DrawNNXModule redraws theta but passes args through unchanged; multi-start
# chains each attempt's initial args from the previous draw, so re-chaining
# restart_key/epoch_key here (and resetting epoch to 0) is what keeps each
# retry's epoch-boundary resampling independent of the others.
class DrawSequentialPolicy:
    def __init__(self, settings, dtype):
        self.draw_theta = DrawNNXModule(SequentialPolicyNN, settings, dtype=dtype)

    def __call__(self, key, _, args):
        theta, _ = self.draw_theta(key, _, args)
        restart_key, epoch_key = jax.random.split(args.restart_key)
        return theta, args.replace(
            epoch_key=epoch_key,
            restart_key=restart_key,
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def __hash__(self):
        return hash((DrawSequentialPolicy, self.draw_theta))

    def __eq__(self, other):
        return (
            isinstance(other, DrawSequentialPolicy)
            and self.draw_theta == other.draw_theta
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
    # Mean-squared normalization; a diverged trajectory surfaces as a
    # nonfinite loss the solver rejects (fixed-step solve_sde never sets
    # ok=False, so there is no mask to apply).
    count = jnp.asarray(residuals.size, dtype=residuals.dtype)
    return residuals / jnp.sqrt(count)


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


# The load template only needs the architecture from meta: the resolved
# scale/anchor Variables ride in the artifact's nondiff block.
def make_sequential_policy(meta, dtype):
    return SequentialPolicyNN(
        SequentialPolicySettings(**meta["settings"]), dtype=dtype, rngs=nnx.Rngs(0)
    )


def save_sequential_policy(path, policy, nn_set, *, structural, initial_state, extra):
    save_model(
        path,
        policy,
        {
            "settings": asdict(nn_set),
            "structural": structural,
            "initial_state": initial_state,
            **extra,
        },
    )


def load_sequential_policy(path):
    return load_model(path, make_sequential_policy)
