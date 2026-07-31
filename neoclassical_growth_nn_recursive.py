from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import LevenbergMarquardt, LMStatus, MultiStart
from tinydiffeq import RK4, ConstantStepSize, Euler, SaveAt, solve_ode

from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from utilities import ACTIVATIONS

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


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
    A: jax.Array
    b_1: jax.Array
    b_2: jax.Array
    kappa_thr: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class DataSettings:
    dt: float = 1.0
    ode_solver: Literal["rk4", "euler"] = "rk4"
    train_steps: int = 32
    train_points: int = 65
    tvc_steps: int = 64
    train_T: float = 40.0
    train_tvc_T: float = 50.0
    test_T: float = 50.0
    num_test_trajectories: int = 50
    settle_T: float = 400.0
    policy_grid_size: int = 100
    benchmark_T: float = 80.0
    benchmark_points: int = 300


@dataclass(frozen=True)
class InitialConditionSettings:
    # Kept separate from the NN initialization seed so every NN seed and the
    # deterministic kernel face the same held-out initial conditions.
    data_seed: int = 0
    # Each of log z_0 and log k_0 receives an independent centered draw with
    # this exact population variance. The economic initial conditions remain
    # the separate pair (z_0, k_0); only the approximator forms log(k/z).
    test_log_variance: float = 0.0025


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    data_set: DataSettings = struct.field(pytree_node=False)


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 250
    train_loss_threshold: float = 1e-14
    num_attempts: int = 1
    log_tvc_threshold: float = 0.0


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 3.26e-5
    damping_decrease: float = 0.482
    damping_increase: float = 2.71
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.541


@dataclass(frozen=True)
class InvestmentPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "gelu"
    initial_investment_share: float = 0.2


class InvestmentPolicyNN(nnx.Module):
    def __init__(self, settings: InvestmentPolicySettings, *, dtype, rngs: nnx.Rngs):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.layers = nnx.List(
            [nnx.Linear(1, settings.width, dtype=dtype, param_dtype=dtype, rngs=rngs)]
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
        initial_logit = math.log(
            settings.initial_investment_share
            / (1.0 - settings.initial_investment_share)
        )
        self.head = nnx.Linear(
            settings.width,
            1,
            kernel_init=nnx.initializers.normal(1e-3),
            bias_init=nnx.initializers.constant(initial_logit),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, log_state, p):
        log_z, log_k = log_state
        h = jnp.atleast_1d(log_k - log_z)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        investment_share = jax.nn.sigmoid(self.head(h)[0])
        z, k = jnp.exp(log_state)
        return production(z, k, p) * investment_share


@dataclass(frozen=True)
class DrawInvestmentPolicy:
    settings: InvestmentPolicySettings
    dtype: object

    def __call__(self, key, _, args):
        module = InvestmentPolicyNN(
            self.settings,
            dtype=self.dtype,
            rngs=nnx.Rngs(key),
        )
        _, theta, _ = nnx.split(module, nnx.Param, ...)
        return theta, args


STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


def production(z, k, p):
    kappa = k / z
    base = kappa**p.a
    return z * p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


def mpk(z, k, p):
    kappa = k / z
    base = p.A * p.a * kappa ** (p.a - 1.0)
    return jnp.where(kappa > p.kappa_thr, p.b_1 * base, base)


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return production(z, k, p) - policy(log_state, p)


def log_state_drift(policy, log_state, p):
    _, log_k = log_state
    k = jnp.exp(log_k)
    investment = policy(log_state, p)
    return jnp.stack([p.g, investment / k - p.delta])


def euler_residuals(policy, log_states, p):
    def residual_scalar(log_state):
        state_drift = log_state_drift(policy, log_state, p)
        _, dlog_c_dt = jax.jvp(
            lambda state: jnp.log(consumption(policy, state, p)),
            (log_state,),
            (state_drift,),
        )
        z, k = jnp.exp(log_state)
        return dlog_c_dt - (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra

    return jax.vmap(residual_scalar)(log_states)


def initial_condition_array(z_0, k_0, dtype):
    z_values = np.atleast_1d(np.asarray(z_0, dtype=float))
    k_values = np.atleast_1d(np.asarray(k_0, dtype=float))
    if z_values.size == 1 and k_values.size > 1:
        z_values = np.full_like(k_values, z_values.item())
    if k_values.size == 1 and z_values.size > 1:
        k_values = np.full_like(z_values, k_values.item())
    if z_values.shape != k_values.shape:
        raise ValueError(
            "z_0 and k_0 must be scalars or broadcastable one-dimensional lists"
        )
    if np.any(z_values <= 0.0) or np.any(k_values <= 0.0):
        raise ValueError("all initial z_0 and k_0 values must be positive")
    return jnp.asarray(np.stack([z_values, k_values], axis=1), dtype=dtype)


def draw_test_initial_conditions(train_initial_conditions, data_set, ic_set):
    n_test = data_set.num_test_trajectories
    variance = ic_set.test_log_variance
    if n_test < 1:
        raise ValueError("num_test_trajectories must be positive")
    if variance < 0.0:
        raise ValueError("test_log_variance must be nonnegative")
    if variance > 0.0 and n_test < 2:
        raise ValueError(
            "positive test_log_variance requires at least two trajectories"
        )

    dtype = train_initial_conditions.dtype
    base_index = jnp.arange(n_test) % train_initial_conditions.shape[0]
    base = train_initial_conditions[base_index]
    if variance == 0.0:
        return base

    key = jax.random.key(ic_set.data_seed)
    draws = jax.random.normal(key, (n_test, 2), dtype=dtype)
    centered = draws - jnp.mean(draws, axis=0, keepdims=True)
    standardized = centered / jnp.sqrt(jnp.mean(centered**2, axis=0, keepdims=True))
    log_z_base = jnp.log(base[:, 0])
    log_k_base = jnp.log(base[:, 1])
    scale = jnp.sqrt(jnp.asarray(variance, dtype=dtype))
    log_z = log_z_base + scale * standardized[:, 0]
    log_k = log_k_base + scale * standardized[:, 1]
    z_test = jnp.exp(log_z)
    k_test = jnp.exp(log_k)
    return jnp.stack([z_test, k_test], axis=1)


def fixed_step_count(T, dt, *, horizon_name):
    if T <= 0.0 or dt <= 0.0:
        raise ValueError(f"{horizon_name} and dt must be positive")
    steps = round(T / dt)
    effective_T = steps * dt
    tolerance = 1e-12 * max(1.0, abs(T), abs(effective_T))
    if steps < 1 or not math.isclose(T, effective_T, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{horizon_name} must be an integer multiple of dt")
    return steps


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


def ode_components(data_set):
    if data_set.ode_solver == "rk4":
        return RK4(), ConstantStepSize()
    if data_set.ode_solver == "euler":
        return Euler(), ConstantStepSize()
    raise ValueError(f"unknown ode_solver: {data_set.ode_solver!r}")


def batched_rollout(
    policy,
    initial_log_states,
    T,
    budget,
    dt_0,
    p,
    data_set,
    save_at,
):
    ode_solver, controller = ode_components(data_set)

    def solve_one(initial_log_state):
        return solve_ode(
            log_state_drift_field,
            ode_solver,
            0.0,
            T,
            initial_log_state,
            dt_0=dt_0,
            controller=controller,
            max_steps=budget,
            save_at=save_at,
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_log_states)


@partial(jax.jit, static_argnames=("data_set",))
def collocation_rollout(policy, p, train_ts, data_set):
    initial_log_states = jnp.log(jnp.stack([p.z_0, p.k_0], axis=1))
    return batched_rollout(
        policy,
        initial_log_states,
        data_set.train_T,
        data_set.train_steps,
        data_set.train_T / data_set.train_steps,
        p,
        data_set,
        SaveAt(ts=train_ts),
    )


@partial(jax.jit, static_argnames=("T", "budget", "dt_0", "data_set"))
def rollout_steps(policy, initial_log_states, T, budget, dt_0, p, data_set):
    return batched_rollout(
        policy,
        initial_log_states,
        T,
        budget,
        dt_0,
        p,
        data_set,
        SaveAt(steps=True),
    )


def collocation_log_states(policy, p, data_set):
    train_ts = jnp.linspace(0.0, data_set.train_T, data_set.train_points)
    solution = collocation_rollout(policy, p, train_ts, data_set)
    ok = solution.ok[:, None, None]
    return jnp.where(ok, solution.xs, jnp.inf), solution


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.fixed_state)
    log_states, _ = collocation_log_states(policy, p, args.data_set)
    residual = euler_residuals(policy, log_states.reshape(-1, 2), p)
    return residual / jnp.sqrt(residual.size)


def log_tvc_residual(policy, initial_log_states, T_start, T_end, p, data_set):
    solution = rollout_steps(
        policy,
        initial_log_states,
        T_end,
        data_set.tvc_steps,
        T_end / data_set.tvc_steps,
        p,
        data_set,
    )
    valid = solution.accepted & (solution.ts >= T_start) & solution.ok[:, None]
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
    return jnp.max(jnp.where(valid, normalized_log_tvc, -jnp.inf)), solution.ok


@lru_cache(maxsize=2048)
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
    z_paths = np.asarray(initial_conditions)[:, :1] * np.exp(g * times[None, :])
    return z_paths, np.stack(k_paths), np.stack(c_paths)


def benchmark_policy_grid(a, delta, rho_hat, sigma_crra, g, kappa_grid, data_set):
    """Interpolate c/z from benchmark transitions covering both sides of BGP."""
    # Stay out of the terminal boundary layer of the truncated BVP. The exact
    # balanced-growth point below closes the small remaining interpolation gap.
    policy_sample_T = max(data_set.train_T, data_set.test_T)
    benchmark_times = np.linspace(0.0, policy_sample_T, data_set.benchmark_points)
    kappa_samples = []
    c_tilde_samples = []
    for kappa_initial in np.unique([kappa_grid[0], kappa_grid[-1]]):
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
        kappa_samples.append((k_path / z_path).reshape(-1))
        c_tilde_samples.append((c_path / z_path).reshape(-1))

    kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1.0 / (a - 1.0))
    c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
    kappa_path = np.concatenate((*kappa_samples, np.asarray([kappa_ss])))
    c_tilde_path = np.concatenate((*c_tilde_samples, np.asarray([c_tilde_ss])))
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
    kappa_benchmark = k_benchmark / z_benchmark
    c_tilde_benchmark = c_benchmark / z_benchmark
    i_benchmark = z_benchmark * kappa_benchmark**a - c_benchmark
    dk_benchmark = i_benchmark - delta * k_benchmark
    dkappa_benchmark = i_benchmark / z_benchmark - (delta + g) * kappa_benchmark

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
            "kappa_0": initial[:, 1] / initial[:, 0],
            "mean_k_abs_rel_error": np.mean(np.abs(k_error), axis=1),
            "median_k_abs_rel_error": np.median(np.abs(k_error), axis=1),
            "mean_c_abs_rel_error": np.mean(np.abs(c_error), axis=1),
            "median_c_abs_rel_error": np.median(np.abs(c_error), axis=1),
        }
    )
    return k_error, c_error, test_summary_df


def trajectory_values(policy, log_states, p):
    shape = log_states.shape[:-1]
    flat_states = log_states.reshape(-1, 2)
    investment = jax.vmap(lambda state: policy(state, p))(flat_states).reshape(shape)
    c = jax.vmap(lambda state: consumption(policy, state, p))(flat_states).reshape(
        shape
    )
    euler = euler_residuals(policy, flat_states, p).reshape(shape)
    z, k = jnp.moveaxis(jnp.exp(log_states), -1, 0)
    kappa = k / z
    dk = investment - p.delta * k
    dz = p.g * z
    dkappa = investment / z - (p.delta + p.g) * kappa
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
            "kappa_0": initial[trajectory, 1] / initial[trajectory, 0],
        }
    )
    for name, value in values.items():
        frame[name] = np.asarray(value).reshape(-1)
    return frame


def neoclassical_growth_nn_recursive(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    sigma_crra: float = 1.0,
    g: float = 0.0,
    z_0: float | list[float] = 1.0,
    k_0: float | list[float] = 1.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    nn_set: InvestmentPolicySettings = InvestmentPolicySettings(),
    lm_set: SolverSettings = SolverSettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    if not 0.0 < a < 1.0:
        raise ValueError("a must lie strictly between zero and one")
    if sigma_crra <= 0.0:
        raise ValueError("sigma_crra must be positive")
    if g >= rho_hat:
        raise ValueError("g must be less than rho_hat")
    if rho_hat + (sigma_crra - 1.0) * g <= 0.0:
        raise ValueError("rho_hat + (sigma_crra - 1) * g must be positive")
    if delta + g <= 0.0:
        raise ValueError("delta + g must be positive")
    if delta + rho_hat + sigma_crra * g <= 0.0:
        raise ValueError("delta + rho_hat + sigma_crra * g must be positive")
    if (delta + rho_hat + sigma_crra * g) / a <= delta + g:
        raise ValueError("balanced-growth consumption per unit of z must be positive")
    if not 0.0 < nn_set.initial_investment_share < 1.0:
        raise ValueError(
            "initial_investment_share must lie strictly between zero and one"
        )
    if data_set.train_T <= 0.0:
        raise ValueError("train_T must be positive")
    if data_set.train_steps < 1 or data_set.train_points < 2:
        raise ValueError("train_steps must be positive and train_points at least two")
    if data_set.dt <= 0.0 or data_set.tvc_steps < 1:
        raise ValueError("dt and tvc_steps must be positive")
    if data_set.train_tvc_T <= data_set.train_T:
        raise ValueError("train_tvc_T must exceed train_T")
    n_test_steps = fixed_step_count(data_set.test_T, data_set.dt, horizon_name="test_T")
    test_T = n_test_steps * data_set.dt
    if data_set.benchmark_T < max(data_set.train_T, test_T):
        raise ValueError("benchmark_T must cover both train_T and test_T")

    kinked_parameter_count = sum(parameter is not None for parameter in (A, b_1, b_2))
    if kinked_parameter_count not in {0, 3}:
        raise ValueError("Set all of A, b_1, and b_2, or set none of them.")
    use_kinked_production = kinked_parameter_count == 3
    A_value = 1.0 if A is None else float(A)
    b_1_value = 1.0 if b_1 is None else float(b_1)
    b_2_value = 0.0 if b_2 is None else float(b_2)
    kappa_thr = (
        (b_2_value / (b_1_value - 1.0)) ** (1.0 / a)
        if use_kinked_production
        else float("inf")
    )

    dtype = jnp.float64 if use_float64 else jnp.float32
    train_initial_conditions = initial_condition_array(z_0, k_0, dtype)
    test_initial_conditions = draw_test_initial_conditions(
        train_initial_conditions, data_set, ic_set
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
        A=A_value,
        b_1=b_1_value,
        b_2=b_2_value,
        kappa_thr=kappa_thr,
    )

    def branch_steady_state(branch_factor):
        return ((delta + rho_hat + sigma_crra * g) / (A_value * a * branch_factor)) ** (
            1.0 / (a - 1.0)
        )

    if use_kinked_production:
        kappa_ss_low = branch_steady_state(1.0)
        kappa_ss_high = branch_steady_state(b_1_value)
        steady_targets = (kappa_ss_low, kappa_ss_high)
    else:
        kappa_ss = branch_steady_state(1.0)
        steady_targets = (kappa_ss,)

    fit_start = time.perf_counter()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    graphdef, _, fixed_state = nnx.split(
        InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(0)), nnx.Param, ...
    )
    residual_args = ResidualArgs(
        graphdef=graphdef,
        fixed_state=fixed_state,
        data_set=data_set,
    )
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    _, theta_0, _ = nnx.split(
        InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key)),
        nnx.Param,
        ...,
    )
    atol = math.sqrt(stop_set.train_loss_threshold)
    draw = DrawInvestmentPolicy(nn_set, dtype)

    start_time = time.perf_counter()
    result = solver.solve(
        theta_0,
        residual_args,
        p=p,
        max_steps=stop_set.max_steps,
        atol=atol,
        multi_start=MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=draw,
        ),
        max_steps_is_success=False,
    )
    jax.block_until_ready(result)
    train_time = time.perf_counter() - start_time
    total_fit_time = time.perf_counter() - fit_start
    setup_time = start_time - fit_start
    policy = nnx.merge(graphdef, result.x, fixed_state)

    train_log_states, train_solution = collocation_log_states(policy, p, data_set)
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_solution.ts,
        train_initial_conditions,
    )

    test_initial_log_states = jnp.log(test_initial_conditions)
    test_solution = rollout_steps(
        policy,
        test_initial_log_states,
        test_T,
        n_test_steps,
        data_set.dt,
        p,
        data_set,
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

    train_initial_log_states = jnp.log(train_initial_conditions)
    max_log_tvc, tvc_ok = log_tvc_residual(
        policy,
        train_initial_log_states,
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    max_log_tvc = float(max_log_tvc)
    transversality_ok = bool(
        jnp.all(tvc_ok) and max_log_tvc < stop_set.log_tvc_threshold
    )
    solver_accepted = bool(result.multi_start.accepted)
    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))

    status = int(result.status)
    train_kappa_0 = np.asarray(
        train_initial_conditions[:, 1] / train_initial_conditions[:, 0]
    )
    test_kappa_0 = np.asarray(
        test_initial_conditions[:, 1] / test_initial_conditions[:, 0]
    )
    results = {
        "variant": "homothetic_investment_share",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": float(np.asarray(train_initial_conditions[0, 0])),
        "k_0": float(np.asarray(train_initial_conditions[0, 1])),
        "A": A_value,
        "b_1": b_1_value,
        "b_2": b_2_value,
        "use_kinked_production": use_kinked_production,
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric": "identity_parameter_metric",
        "selection": "minimum_identity_metric_parameter_step",
        "train_loss": float(result.info.loss),
        "max_train_euler": max_train_euler,
        "final_damping": float(result.lm_state.damping),
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_values["euler_residual"]))),
        "max_log_tvc_residual": max_log_tvc,
        "tvc_normalization": "relative_to_initial_log_certificate",
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_values["c"])),
        "test_path_ok": test_path_ok,
        "steps": int(result.steps),
        "attempt": int(result.multi_start.attempt) + 1,
        "stopping_reason": STATUS_TO_REASON.get(status, "max_steps"),
        "train_time": train_time,
        "solve_time": train_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "train_time_success": train_time if solver_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        "solution_accepted": solver_accepted,
        "solver_converged": solver_accepted,
        "num_train_trajectories": int(train_initial_conditions.shape[0]),
        "num_test_trajectories": data_set.num_test_trajectories,
        "num_collocation_states": int(
            train_initial_conditions.shape[0] * data_set.train_points
        ),
        "train_kappa_0_min": float(train_kappa_0.min()),
        "train_kappa_0_max": float(train_kappa_0.max()),
        "test_kappa_0_min": float(test_kappa_0.min()),
        "test_kappa_0_max": float(test_kappa_0.max()),
        "test_log_variance": ic_set.test_log_variance,
        "ode_solver": data_set.ode_solver,
        "train_steps": data_set.train_steps,
        "dt": data_set.dt,
        "train_T": data_set.train_T,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    if use_kinked_production:
        n_settle_steps = fixed_step_count(
            data_set.settle_T, data_set.dt, horizon_name="settle_T"
        )
        settle_solution = rollout_steps(
            policy,
            train_initial_log_states,
            n_settle_steps * data_set.dt,
            n_settle_steps,
            data_set.dt,
            p,
            data_set,
        )
        settled = jnp.exp(settle_solution.xs[:, -1])
        kappa_settled = np.asarray(settled[:, 1] / settled[:, 0])
        tol = 0.05 * kappa_thr
        targets = np.where(train_kappa_0 < kappa_thr, kappa_ss_low, kappa_ss_high)
        others = np.where(train_kappa_0 < kappa_thr, kappa_ss_high, kappa_ss_low)
        per_correct = np.abs(kappa_settled - targets) < tol
        per_fp = (~per_correct) & (np.abs(kappa_settled - others) < tol)
        results.update(
            kappa_thr=kappa_thr,
            kappa_ss_low=kappa_ss_low,
            kappa_ss_high=kappa_ss_high,
            kappa_settled=float(kappa_settled[0]),
            basin_correct=bool(np.all(per_correct)),
            false_positive=bool(np.any(per_fp)),
            basin_failed=bool(np.any(~(per_correct | per_fp))),
        )
    else:
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
            kappa_ss=kappa_ss,
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

    grid_lo = min(
        float(train_kappa_0.min()), float(test_kappa_0.min()), *steady_targets
    )
    grid_hi = max(
        float(train_kappa_0.max()), float(test_kappa_0.max()), *steady_targets
    )
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
            "i_per_z_nn": np.asarray(grid_values["i"][0]),
            "c_tilde_nn": np.asarray(grid_values["c"][0]),
            "dkappa_nn": np.asarray(grid_values["dkappa"][0]),
        }
    )
    if not use_kinked_production:
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

    hyperparameters = {
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": asdict(lm_set),
        "stop_set": asdict(stop_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "g": g,
            "A": A_value,
            "b_1": b_1_value,
            "b_2": b_2_value,
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
    }
    if not use_kinked_production:
        extras["test_summary_df"] = test_summary_df
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn_recursive)
