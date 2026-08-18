from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx, struct
from tinydiffeq import RK4, ConstantStepSize, IController, SaveAt, Tsit5, solve_ode

DOMAIN_EPS = 1e-8


# Concave-convex (kinked) production calibration shared by the three
# concave_convex replication sweeps and the basin-threshold geometry.
@dataclass(frozen=True)
class KinkedCalibration:
    a: float = 1 / 3
    delta: float = 0.1
    rho_hat: float = 0.11
    sigma_crra: float = 1.0
    A: float = 0.5
    b_1: float = 3.0
    b_2: float = 2.5


KINKED_CALIBRATION = KinkedCalibration()


# Sequence-space structural parameters: smooth Cobb-Douglas production is
# (A, b_1, b_2) = (1, 1, 0); all three set selects the kinked family.
@struct.dataclass
class StructuralParameters:
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    k_0: jax.Array
    A: jax.Array
    b_1: jax.Array
    b_2: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


# Recursive structural parameters: smooth production with balanced growth at
# per-capita rate g; the initial conditions ride in the solver args instead.
@struct.dataclass
class RecursiveParameters:
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    g: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


# All three kink parameters set selects the concave-convex family; none
# selects smooth Cobb-Douglas. A partial specification is an error.
def resolve_kink_family(A, b_1, b_2):
    kinked_parameter_count = sum(parameter is not None for parameter in (A, b_1, b_2))
    if kinked_parameter_count not in {0, 3}:
        raise ValueError("Set all of A, b_1, and b_2, or set none of them.")
    return (
        kinked_parameter_count == 3,
        1.0 if A is None else float(A),
        1.0 if b_1 is None else float(b_1),
        0.0 if b_2 is None else float(b_2),
    )


def production(k, p):
    base = k**p.a
    return p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


def marginal_product(k, p):
    base = k**p.a
    branch = jnp.where(p.b_1 * base - p.b_2 > base, p.b_1, 1.0)
    return p.A * branch * (p.a * k ** (p.a - 1.0))


# Flat steady-flow consumption warm start c(0) = f(k_0) - delta k_0, floored
# away from zero; pure host arithmetic on the caller's floats.
def steady_flow_consumption_guess(k_0, a, delta, A, b_1, b_2):
    base = k_0**a
    return max(A * max(base, b_1 * base - b_2) - delta * k_0, DOMAIN_EPS)


# Hicks-neutral technology: z is the TFP level, so Y = z k^a. Balanced growth
# at per-capita rate g needs the TFP trend (1 - a) g; the labor-augmenting
# index z^{1/(1-a)} grows at g and is the detrending scale for quantities.
def hicks_production(z, k, p):
    return z * k**p.a


def mpk(z, k, p):
    return p.a * z * k ** (p.a - 1.0)


def labor_augmenting_scale(z, p):
    return z ** (1.0 / (1.0 - p.a))


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return hicks_production(z, k, p) - policy(log_state, p)


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


# Closed-form steady state of the static smooth model (z = 1, g = 0):
# f'(k_ss) = delta + rho_hat with f(k) = k^a gives
# k_ss = (a / (delta + rho_hat))^(1/(1 - a)) and c_ss = k_ss^a - delta k_ss.
def steady_state(a, delta, rho_hat):
    k_ss = (a / (delta + rho_hat)) ** (1.0 / (1.0 - a))
    return k_ss, k_ss**a - delta * k_ss


# Exact steady-state derivatives, the AD-free target the solver sensitivities
# are checked against. sigma_crra and k_0 do not enter, so those derivatives
# are exactly zero.
def neoclassical_steady_state_sensitivities(a, delta, rho_hat):
    mpk_ss = delta + rho_hat
    k_ss, c_ss = steady_state(a, delta, rho_hat)
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
    a=KINKED_CALIBRATION.a,
    delta=KINKED_CALIBRATION.delta,
    rho_hat=KINKED_CALIBRATION.rho_hat,
    A=KINKED_CALIBRATION.A,
    b_1=KINKED_CALIBRATION.b_1,
    b_2=KINKED_CALIBRATION.b_2,
):
    # Branch switch k_thr where k^a = b_1 k^a - b_2, and the per-branch steady
    # states from f'(k_ss) = delta + rho_hat (ex-post classification only).
    k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a)
    k_ss_low = ((delta + rho_hat) / (A * a)) ** (1.0 / (a - 1.0))
    k_ss_high = ((delta + rho_hat) / (A * a * b_1)) ** (1.0 / (a - 1.0))
    return k_thr, k_ss_low, k_ss_high


# Well-posedness of the planner objective, not steady-state structure:
# finiteness of discounted utility needs rho_hat + (sigma - 1) g > 0, the
# discounted-capital certificate needs g < rho_hat, and detrended capital
# needs positive effective depreciation.
def validate_balanced_growth(delta, rho_hat, sigma_crra, g):
    if g >= rho_hat:
        raise ValueError("g must be less than rho_hat")
    if rho_hat + (sigma_crra - 1.0) * g <= 0.0:
        raise ValueError("rho_hat + (sigma_crra - 1) * g must be positive")
    if delta + g <= 0.0:
        raise ValueError("delta + g must be positive")


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


# The (z, k) train/test initial-condition design every recursive growth driver
# shares: independent draws from one data_seed, with the test center and
# variances defaulting to the training ones.
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
    return {
        "train_ic_key": train_ic_key,
        "test_ic_key": test_ic_key,
        "train_initial_conditions": train_initial_conditions,
        "test_initial_conditions": test_initial_conditions,
        "test_z_center": test_z_center,
        "test_k_center": test_k_center,
        "test_log_z_variance": test_log_z_variance,
        "test_log_k_variance": test_log_k_variance,
    }


def initial_kappa(initial_conditions, a):
    return np.asarray(
        initial_conditions[:, 1] / initial_conditions[:, 0] ** (1.0 / (1.0 - a))
    )


def positive_policy_grid(grid_lo, grid_hi, size):
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


# Windowed and terminal transversality certificates on the training and
# held-out initial conditions, as published results fields.
def tvc_certificates(
    policy, train_log_states, test_log_states, p, data_set, log_tvc_threshold
):
    max_log_tvc, train_terminal_log_tvc, tvc_ok = log_tvc_residual(
        policy,
        train_log_states,
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    max_log_tvc = float(max_log_tvc)
    transversality_ok = bool(jnp.all(tvc_ok) and max_log_tvc < log_tvc_threshold)
    _, test_terminal_log_tvc, test_tvc_ok = log_tvc_residual(
        policy,
        test_log_states,
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    test_max_log_tvc = float(jnp.max(test_terminal_log_tvc))
    test_transversality_ok = bool(
        jnp.all(test_tvc_ok) and test_max_log_tvc < log_tvc_threshold
    )
    return {
        "max_log_tvc_residual": max_log_tvc,
        "mean_log_tvc_residual": float(jnp.mean(train_terminal_log_tvc)),
        "test_mean_log_tvc_residual": float(jnp.mean(test_terminal_log_tvc)),
        "test_max_log_tvc_residual": test_max_log_tvc,
        "test_transversality_ok": test_transversality_ok,
        "tvc_normalization": "relative_to_initial_log_certificate",
        "transversality_ok": transversality_ok,
    }


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
    # Deferred import: neoclassical_growth_benchmark imports this module's
    # validate_balanced_growth.
    from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark

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


# Diagnostic grid bounds come from the realized states, never a steady
# state. Snapping to a coarse log lattice keeps the cached benchmark
# starts shared across seeds and data draws.
def kappa_diagnostic_grid(
    train_kappa, test_kappa, train_kappa_0, test_kappa_0, policy_grid_size, dtype
):
    observed_kappa = np.concatenate(
        [train_kappa, test_kappa, train_kappa_0, test_kappa_0]
    )
    observed_kappa = observed_kappa[np.isfinite(observed_kappa) & (observed_kappa > 0)]
    if observed_kappa.size == 0:
        observed_kappa = np.concatenate([train_kappa_0, test_kappa_0])
    grid_lo = math.exp(math.floor(math.log(observed_kappa.min()) / 0.05) * 0.05)
    grid_hi = math.exp(math.ceil(math.log(observed_kappa.max()) / 0.05) * 0.05)
    kappa_grid = positive_policy_grid(grid_lo, grid_hi, policy_grid_size)
    grid_log_states = jnp.stack(
        [
            jnp.zeros(policy_grid_size, dtype=dtype),
            jnp.log(jnp.asarray(kappa_grid, dtype=dtype)),
        ],
        axis=1,
    )
    return kappa_grid, grid_log_states
