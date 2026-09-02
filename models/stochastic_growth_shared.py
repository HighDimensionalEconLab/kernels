from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tinydiffeq import SaveAt, solve_sde

from models.neural_networks import ACTIVATIONS, StateMLP, unit_softplus
from models.utilities import ParameterPyTree

# Stochastic neoclassical growth on the state (log z, log k): OU log-TFP
# deviation around log z_bar with smooth Cobb-Douglas technology, exactly the
# paper's F(k, z) = e^z k^a written here with z in levels.


# AFV-2006 (JEDC) calibration; quarterly anchors map exactly to the OU rates
# (e^(-rho_hat/4) = beta, e^(-eta/4) = rho, matched stationary variance).
STOCHASTIC_GROWTH_CONSTANTS = {
    "a": 1 / 3,
    "delta": 0.1,
    "sigma_crra": 2.0,
    "quarterly_beta": 0.9896,
    "quarterly_tfp_rho": 0.95,
    "quarterly_tfp_sigma": 0.035,
    "rho_hat": -4.0 * math.log(0.9896),
    "eta": -4.0 * math.log(0.95),
    "sigma_z": math.sqrt(2.0 * (-4.0 * math.log(0.95)) * (0.035**2 / (1.0 - 0.95**2))),
    "train_T": 60.0,
    "test_T": 40.0,
    "transversality_T": 200.0,
}


class StructuralParameters(ParameterPyTree):
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    eta: jax.Array
    sigma_z: jax.Array
    log_z_bar: jax.Array
    z_0: jax.Array
    k_0: jax.Array


@dataclass(frozen=True)
class DataSettings:
    # Fixed-step SDE simulation on the integrator's own states: n points from
    # n - 1 steps including the initial condition, nothing interpolated.
    train_points: int = 41
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = STOCHASTIC_GROWTH_CONSTANTS["train_T"]
    train_tvc_T: float = STOCHASTIC_GROWTH_CONSTANTS["transversality_T"]
    test_T: float = STOCHASTIC_GROWTH_CONSTANTS["test_T"]
    policy_grid_size: int = 100


def production(z, k, p):
    return z * k**p.a


def mpk(z, k, p):
    return p.a * z * k ** (p.a - 1.0)


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return production(z, k, p) - policy(log_state)[0]


def log_state_drift(policy, log_state, p):
    log_z, log_k = log_state
    k = jnp.exp(log_k)
    investment = policy(log_state)[0]
    return jnp.stack([p.eta * (p.log_z_bar - log_z), investment / k - p.delta])


def log_state_diffusion(log_state, p):
    return jnp.stack([p.sigma_z, jnp.zeros_like(p.sigma_z)])


# The problem half of the two-part artifact key, shared verbatim with any
# consumer that looks up a saved policy for the same problem (e.g. the
# stochastic kernel-recursive collocation generator).
def problem_cache_config(
    a, delta, rho_hat, sigma_crra, eta, sigma_z, z_bar, z_0, k_0, use_float64, ic_set
):
    return {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "z_0": z_0,
        "k_0": k_0,
        "use_float64": use_float64,
        "train_ic": {
            "num_train_trajectories": ic_set.num_train_trajectories,
            "train_log_z_variance": ic_set.train_log_z_variance,
            "train_log_k_variance": ic_set.train_log_k_variance,
        },
    }


# Stochastic Euler equation (the differentiated HJBE) along d log z =
# eta (log_z_bar - log z) dt + sigma_z dW, with l = log c:
#   grad(l) . mu + (sigma_z^2/2)(l_zz - sigma_crra l_z^2)
#     = (mpk - delta - rho_hat) / sigma_crra
# The Ito terms come from the drift of u'(c) = exp(-sigma_crra l); sigma_z = 0
# recovers the deterministic residual exactly.
def euler_residual_terms(policy, log_states, p):
    e_z = jnp.asarray([1.0, 0.0], dtype=log_states.dtype)

    def scalar_terms(log_state):
        state_drift = log_state_drift(policy, log_state, p)

        def log_consumption(state):
            return jnp.log(consumption(policy, state, p))

        log_c, dlog_c_dt = jax.jvp(log_consumption, (log_state,), (state_drift,))
        l_z, l_zz = jax.jvp(
            lambda state: jax.jvp(log_consumption, (state,), (e_z,))[1],
            (log_state,),
            (e_z,),
        )
        diffusion_term = 0.5 * p.sigma_z**2 * (l_zz - p.sigma_crra * l_z**2)
        z, k = jnp.exp(log_state)
        # The zero-scaled primal keeps the residual NaN outside the model's
        # domain (c <= 0) instead of quietly extending the Euler equation.
        domain_guard = 0.0 * log_c
        residual = (
            dlog_c_dt
            + diffusion_term
            + domain_guard
            - (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
        )
        return residual, diffusion_term

    return jax.vmap(scalar_terms)(log_states)


def euler_residuals(policy, log_states, p):
    return euler_residual_terms(policy, log_states, p)[0]


@partial(jax.jit, static_argnames=("T", "n_steps"))
def simulate(sde, policy, initial_log_states, noise, T, n_steps, p):
    def solve_one(initial_log_state, trajectory_noise):
        return solve_sde(
            lambda log_state: log_state_drift(policy, log_state, p),
            lambda log_state: log_state_diffusion(log_state, p),
            sde,
            0.0,
            T,
            initial_log_state,
            noise=trajectory_noise,
            n_steps=n_steps,
            save_at=SaveAt(steps=True),
        )

    return jax.vmap(solve_one)(initial_log_states, noise)


@partial(jax.jit, static_argnames=("data_set",))
def simulate_train(sde, policy, initial_log_states, noise, p, data_set):
    return simulate(
        sde,
        policy,
        initial_log_states,
        noise,
        data_set.train_T,
        data_set.train_points - 1,
        p,
    )


# Natural-log transversality certificate at the terminal date T, normalized
# by the initial-date certificate, per initial state: the saddle path sits
# far negative, a capital bubble drives it positive. A failed simulation is a
# loud +inf.
def log_tvc_residual(sde, policy, initial_log_states, noise, T, n_steps, p):
    solution = simulate(sde, policy, initial_log_states, noise, T, n_steps, p)
    terminal_log_states = solution.xs[:, -1, :]
    c_T = jax.vmap(lambda state: consumption(policy, state, p))(terminal_log_states)
    k_T = jnp.exp(terminal_log_states[:, 1])
    c_0 = jax.vmap(lambda state: consumption(policy, state, p))(initial_log_states)
    k_0 = jnp.exp(initial_log_states[:, 1])
    log_tvc = -p.rho_hat * T - p.sigma_crra * jnp.log(c_T) + jnp.log(k_T)
    initial_log_tvc = -p.sigma_crra * jnp.log(c_0) + jnp.log(k_0)
    return jnp.where(solution.ok, log_tvc - initial_log_tvc, jnp.inf)


# Common-noise pathwise errors against a reference policy: the exogenous z
# path is identical under both policies, so the comparison isolates the
# capital and consumption responses.
def add_reference_path_diagnostics(test_df, test_values, fd_values):
    k_benchmark = np.asarray(fd_values["k"])
    c_benchmark = np.asarray(fd_values["c"])
    k_error = (np.asarray(test_values["k"]) - k_benchmark) / np.abs(k_benchmark)
    c_error = (np.asarray(test_values["c"]) - c_benchmark) / np.abs(c_benchmark)
    columns = {
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "k_rel_error": k_error,
        "c_rel_error": c_error,
    }
    for name, value in columns.items():
        test_df[name] = np.asarray(value).reshape(-1)
    return k_error, c_error


# nnx.jit so one compiled evaluation serves both the nnx policy and the FD
# interpolated policy pytree.
@nnx.jit
def trajectory_values(policy, log_states, p):
    shape = log_states.shape[:-1]
    flat_states = log_states.reshape(-1, 2)
    investment = jax.vmap(lambda state: policy(state)[0])(flat_states).reshape(shape)
    c = jax.vmap(lambda state: consumption(policy, state, p))(flat_states).reshape(
        shape
    )
    euler, diffusion_term = euler_residual_terms(policy, flat_states, p)
    z, k = jnp.moveaxis(jnp.exp(log_states), -1, 0)
    return {
        "z": z,
        "k": k,
        "kappa": k / z ** (1.0 / (1.0 - p.a)),
        "i": investment,
        "c": c,
        "dlog_z": p.eta * (p.log_z_bar - jnp.log(z)),
        "dk": investment - p.delta * k,
        "euler_residual": euler.reshape(shape),
        "diffusion_term": diffusion_term.reshape(shape),
    }


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


@dataclass(frozen=True)
class InvestmentPolicySettings:
    mlp_width: int = 8
    mlp_depth: int = 3
    hidden_activation: str = "silu"
    head_init_scale: float = 1e-3


# The load template spells out the architecture and the output map from
# meta; only the normalization and output-scale Variables ride in the
# artifact's nondiff block and overwrite the placeholder defaults.
def make_investment_policy(meta, dtype):
    settings = InvestmentPolicySettings(**meta["settings"])
    return StateMLP(
        2,
        1,
        width=settings.mlp_width,
        depth=settings.mlp_depth,
        activation=ACTIVATIONS[settings.hidden_activation],
        head_kernel_std=settings.head_init_scale,
        final_activation=unit_softplus,
        output_scale=(1.0,),
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )
