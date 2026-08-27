from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_ode

from models.utilities import ParameterPyTree

# AFV-2006 (JEDC, Tables 1-2) calibration shared by every deterministic
# variant; the quarterly anchor maps exactly via e^(-rho_hat/4) = beta.
NEOCLASSICAL_GROWTH_CONSTANTS = {
    "a": 1 / 3,
    "delta": 0.1,
    "sigma_crra": 2.0,
    "quarterly_beta": 0.9896,
    "rho_hat": -4.0 * math.log(0.9896),
    "train_T": 100.0,
    "test_T": 40.0,
    "sequence_test_T": 110.0,
    "transversality_T": 200.0,
}


# The concave-convex (kinked) production calibration.
KINKED_CALIBRATION = {
    "a": 1 / 3,
    "delta": 0.1,
    "rho_hat": 0.11,
    "sigma_crra": 1.0,
    "A": 0.5,
    "b_1": 3.0,
    "b_2": 2.5,
}


# (A, b_1, b_2, k_thr) = (1, 1, 0, inf) is smooth Cobb-Douglas; the kinked
# family sets all four, with k_thr = (b_2 / (b_1 - 1))^{1/a} the branch
# switch. The sequential kernel/NN residuals read the trend through batch
# z_trend data rather than p.g; only drivers that need z(t) analytically
# (consumption substituted from the resource constraint) differentiate
# through g.
class StructuralParameters(ParameterPyTree):
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    g: jax.Array
    A: jax.Array
    b_1: jax.Array
    b_2: jax.Array
    k_thr: jax.Array
    # The recursive drivers omit the initial conditions (they ride in the
    # solver args); to_dict skips None fields.
    z_0: jax.Array | None = None
    k_0: jax.Array | None = None


def production(z, k, p):
    base = k**p.a
    return z * p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


# Marginal product, kinked at k_thr: the branch switch is intrinsic to the
# model's first-order condition and measure-zero through the where.
def mpk(z, k, p):
    branch = jnp.where(k > p.k_thr, p.b_1, 1.0)
    return z * p.A * p.a * branch * k ** (p.a - 1.0)


@dataclass(frozen=True)
class DataSettings:
    train_T: float = NEOCLASSICAL_GROWTH_CONSTANTS["train_T"]
    train_points: int = 101
    train_points_list: list[float] | None = None
    test_T: float = NEOCLASSICAL_GROWTH_CONSTANTS["sequence_test_T"]
    test_points: int = 100
    validation_points: int = 801
    test_tvc_T: float = NEOCLASSICAL_GROWTH_CONSTANTS["transversality_T"]


# Threshold for the reported transversality_ok diagnostic; acceptance gates
# on the training loss only.
@dataclass(frozen=True)
class ToleranceSettings:
    log_tvc_threshold: float = 0.0


@dataclass(frozen=True)
class GrowthPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    hidden_bias: bool = True
    output_bias: bool = True
    # Trend e^{g_hat t} on the output heads: a learned Param (init g_hat_init)
    # or fixed at the constructor's g_hat.
    learn_g_hat: bool = False
    g_hat_init: float = 0.0


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
        # Zero on the feasible domain; keeps the residual NaN when c <= 0,
        # where the tangent alone would stay finite.
        domain_guard = 0.0 * log_c
        return (
            dlog_c_dt
            + domain_guard
            - (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
        )

    return jax.vmap(residual_scalar)(log_states)


# Flat steady-flow warm start c(0) = f(z_0, k_0) - delta k_0.
def steady_flow_consumption_guess(p):
    base = float(p.k_0) ** float(p.a)
    return float(p.z_0) * float(p.A) * max(
        base, float(p.b_1) * base - float(p.b_2)
    ) - float(p.delta) * float(p.k_0)


# Closed-form steady state of the static smooth model (z = 1, g = 0):
# f'(k_ss) = delta + rho_hat with f(k) = k^a gives
# k_ss = (a / (delta + rho_hat))^(1/(1 - a)) and c_ss = k_ss^a - delta k_ss.
def steady_state(p):
    a, delta, rho_hat = float(p.a), float(p.delta), float(p.rho_hat)
    k_ss = (a / (delta + rho_hat)) ** (1.0 / (1.0 - a))
    return k_ss, k_ss**a - delta * k_ss


# The AD-free target the solver sensitivities are checked against;
# sigma_crra and k_0 do not enter, so those derivatives are exactly zero.
def neoclassical_steady_state_sensitivities(p):
    a, delta, rho_hat = float(p.a), float(p.delta), float(p.rho_hat)
    mpk_ss = delta + rho_hat
    k_ss, c_ss = steady_state(p)
    d_k_ss_d_a = k_ss * (1.0 / (a * (1.0 - a)) + math.log(a / mpk_ss) / (1.0 - a) ** 2)
    d_k_ss_d_delta = -k_ss / ((1.0 - a) * mpk_ss)
    # a k_ss^(a-1) - delta = rho_hat collapses the chain rule for c_ss.
    return {
        "k_ss": k_ss,
        "c_ss": c_ss,
        "d_k_ss_d_a": d_k_ss_d_a,
        "d_k_ss_d_delta": d_k_ss_d_delta,
        "d_k_ss_d_rho_hat": d_k_ss_d_delta,
        "d_k_ss_d_sigma_crra": 0.0,
        "d_k_ss_d_k_0": 0.0,
        "d_c_ss_d_a": k_ss**a * math.log(k_ss) + rho_hat * d_k_ss_d_a,
        "d_c_ss_d_delta": rho_hat * d_k_ss_d_delta - k_ss,
        "d_c_ss_d_rho_hat": rho_hat * d_k_ss_d_delta,
        "d_c_ss_d_sigma_crra": 0.0,
        "d_c_ss_d_k_0": 0.0,
    }


def kinked_thresholds(
    a=KINKED_CALIBRATION["a"],
    delta=KINKED_CALIBRATION["delta"],
    rho_hat=KINKED_CALIBRATION["rho_hat"],
    A=KINKED_CALIBRATION["A"],
    b_1=KINKED_CALIBRATION["b_1"],
    b_2=KINKED_CALIBRATION["b_2"],
):
    # Branch switch k_thr where k^a = b_1 k^a - b_2, and the per-branch steady
    # states from f'(k_ss) = delta + rho_hat (ex-post classification only).
    k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a)
    k_ss_low = ((delta + rho_hat) / (A * a)) ** (1.0 / (a - 1.0))
    k_ss_high = ((delta + rho_hat) / (A * a * b_1)) ** (1.0 / (a - 1.0))
    return k_thr, k_ss_low, k_ss_high


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


def log_state_drift_field(log_state, _, args, p):
    policy, structural = p
    return log_state_drift(policy, log_state, structural)


# Fixed-step RK4 simulation of the policy from a batch of initial log states,
# saving the integrator's own steps.
@partial(jax.jit, static_argnames=("T", "n_steps"))
def simulate(policy, initial_log_states, T, n_steps, p):
    def solve_one(initial_log_state):
        return solve_ode(
            log_state_drift_field,
            RK4(),
            0.0,
            T,
            initial_log_state,
            dt_0=T / n_steps,
            controller=ConstantStepSize(),
            max_steps=n_steps,
            save_at=SaveAt(steps=True),
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_log_states)


# Natural-log transversality certificate at the terminal date T, normalized
# by the initial-date certificate, per initial state: the saddle path sits
# far negative, a capital bubble drives it positive. A failed simulation is a
# loud +inf.
def log_tvc_residual(policy, initial_log_states, T, n_steps, p):
    solution = simulate(policy, initial_log_states, T, n_steps, p)
    terminal_log_states = solution.xs[:, -1, :]
    c_T = jax.vmap(lambda state: consumption(policy, state, p))(terminal_log_states)
    k_T = jnp.exp(terminal_log_states[:, 1])
    c_0 = jax.vmap(lambda state: consumption(policy, state, p))(initial_log_states)
    k_0 = jnp.exp(initial_log_states[:, 1])
    log_tvc = -p.rho_hat * T - p.sigma_crra * jnp.log(c_T) + jnp.log(k_T)
    initial_log_tvc = -p.sigma_crra * jnp.log(c_0) + jnp.log(k_0)
    return jnp.where(solution.ok, log_tvc - initial_log_tvc, jnp.inf)


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
    scale = z ** (1.0 / (1.0 - p.a))
    kappa = k / scale
    dkappa = investment / scale - (p.delta + p.g) * kappa
    return {
        "z": z,
        "k": k,
        "kappa": kappa,
        "c": c,
        "dkappa": dkappa,
        "euler_residual": euler,
    }
