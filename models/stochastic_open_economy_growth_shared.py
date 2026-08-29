from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from tinydiffeq import SaveAt, solve_sde

from models.neural_networks import ACTIVATIONS, StateMLP
from models.utilities import ParameterPyTree

# Recursive stochastic small open economy on (log z, log k, a, rate_gap).
# No benchmark or steady-state information enters the residual or
# initialization.


# OU driving processes from AR(1) anchors (e^(-kappa) = rho, matched
# stationary variance): AFV-2006 TFP; world rate from Uribe & Yue (2006).
PROCESS_CONSTANTS = {
    "quarterly_tfp_rho": 0.95,
    "quarterly_tfp_sigma": 0.035,
    "annual_rate_rho": 0.83**4,
    "annual_rate_innovation_sd": 0.0115,
    "kappa_z": -4.0 * math.log(0.95),
    "sigma_z": math.sqrt(2.0 * (-4.0 * math.log(0.95)) * (0.035**2 / (1.0 - 0.95**2))),
    "kappa_r": -math.log(0.83**4),
    "sigma_r": 0.0115 * math.sqrt(2.0 * (-math.log(0.83**4)) / (1.0 - (0.83**4) ** 2)),
}


# The deterministic sequential counterpart in open_economy_growth.py carries
# the same calibration.
@dataclass(frozen=True)
class ModelParameters:
    alpha: float = 0.33
    gamma: float = 2.0
    delta: float = -math.log(0.94)
    r: float = -math.log(0.96)
    eta: float = 0.76
    r_star: float = -math.log(0.96)


def production(k, z, p):
    return z * k**p.alpha


def marginal_product(k, z, p):
    return p.alpha * z * k ** (p.alpha - 1.0)


class StructuralParameters(ParameterPyTree):
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


@dataclass(frozen=True)
class DataSettings:
    # Fixed-step SDE simulation on the integrator's own states; train_T covers
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


@dataclass(frozen=True)
class SequentialPolicySettings:
    width: int = 16
    depth: int = 3
    hidden_activation: str = "gelu"
    investment_head_init_scale: float = 1e-3


def state_levels(state, p):
    log_z, log_k, a, rate_gap = state
    return {
        "z": jnp.exp(log_z),
        "k": jnp.exp(log_k),
        "a": a,
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
            values["current_account"],
            -p.kappa_r * rate_gap,
        ]
    )


# One covariance, written twice: diagonal diffusion + pre-mixed noise
# (correlate_noise) in the simulation, explicit direction rows
# (diffusion_directions) in the residual. Edit them together only; the
# directional-generator test guards the pairing.
def state_diffusion(state, p):
    zero = jnp.zeros_like(state[0])
    return jnp.stack([p.sigma_z, zero, zero, p.sigma_r])


# Residual half of the pairing: directions.T @ directions reproduces the
# simulation's mixed-noise covariance exactly.
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


@partial(jax.jit, static_argnames=("T", "n_steps"))
def simulate(sde, policy, initial_states, noise, T, n_steps, p):
    correlated_noise = correlate_noise(noise, p.corr_zr)

    def solve_one(initial_state, trajectory_noise):
        return solve_sde(
            lambda state: state_drift(policy, state, p),
            lambda state: state_diffusion(state, p),
            sde,
            0.0,
            T,
            initial_state,
            noise=trajectory_noise,
            n_steps=n_steps,
            save_at=SaveAt(steps=True),
        )

    return jax.vmap(solve_one)(initial_states, correlated_noise)


@partial(jax.jit, static_argnames=("data_set",))
def simulate_train(sde, policy, initial_states, noise, p, data_set):
    return simulate(
        sde,
        policy,
        initial_states,
        noise,
        data_set.train_T,
        data_set.train_points - 1,
        p,
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


# The load template only needs the architecture from meta: the resolved
# normalization and anchor Variables ride in the artifact's nondiff block and
# overwrite the placeholder defaults.
def make_sequential_policy(meta, dtype):
    settings = SequentialPolicySettings(**meta["settings"])
    return StateMLP(
        4,
        width=settings.width,
        depth=settings.depth,
        activation=ACTIVATIONS[settings.hidden_activation],
        heads=((1, settings.investment_head_init_scale), (1, None)),
        positive=(True, False),
        squash=(False, False, True, False),
        input_kernel_mask=(0.0, 1.0, 0.0, 0.0),
        anchor=(1.0, 1.0),
        out_scale=(1.0, 1.0),
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )
