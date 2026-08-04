from __future__ import annotations

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
from nlls_gram import LevenbergMarquardt, LMStatus, MultiStart
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_ode

from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from utilities import ACTIVATIONS, unit_softplus

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

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class DataSettings:
    # Every rollout is fixed-step RK4 evaluated on the integrator's own states:
    # ``n`` points come from ``n - 1`` steps of ``T / (n - 1)``, including the
    # initial condition. Training places a residual on every knot; the held-out
    # path reports on the same states, so nothing is interpolated.
    train_points: int = 65
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = 100.0
    train_tvc_T: float = 200.0
    test_T: float = 40.0
    policy_grid_size: int = 100
    benchmark_T: float = 200.0
    benchmark_points: int = 801


@dataclass(frozen=True)
class InitialConditionSettings:
    # One data_seed fixes both the training and held-out draws (its key is
    # split into independent train and test streams), and it is kept separate
    # from the NN initialization seed so every NN seed and the deterministic
    # kernel face identical initial conditions. Train and test are independent
    # draws centered on the same scalar (z_0, k_0); each of log z_0 and
    # log k_0 receives an independent centered draw with the exact requested
    # population variance, so a zero variance pins that axis at its base.
    data_seed: int = 0
    num_train_trajectories: int = 1
    num_test_trajectories: int = 50
    train_log_z_variance: float = 0.0
    train_log_k_variance: float = 0.0
    test_log_z_variance: float = 0.0025
    test_log_k_variance: float = 0.0025
    # Held-out draws recenter on these when set (out-of-distribution testing);
    # None keeps the training center (z_0, k_0).
    test_z_0: float | None = None
    test_k_0: float | None = None


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    data_set: DataSettings = struct.field(pytree_node=False)


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 250
    train_loss_threshold: float | None = None
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
    head_init_scale: float = 1e-3


# i = delta * k * U(h(q)) with U the unit softplus (U(0) = U'(0) = 1), so
# kdot/k = delta * (U(h) - 1): the near-zero head starts capital at rest in
# levels, investment is positive, and consumption Y - i is unclamped (an
# infeasible iterate is a loud NaN residual). Only delta and the state enter
# the map; homotheticity appears only through the head input q = log(k/z).
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
        self.head = nnx.Linear(
            settings.width,
            1,
            kernel_init=nnx.initializers.normal(settings.head_init_scale),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, log_state, p):
        log_z, log_k = log_state
        h = jnp.atleast_1d(log_k - log_z)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        k = jnp.exp(log_k)
        return p.delta * k * unit_softplus(self.head(h)[0])


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
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


def production(z, k, p):
    return z * (k / z) ** p.a


def mpk(z, k, p):
    return p.a * (k / z) ** (p.a - 1.0)


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
def collocation_rollout(policy, p, data_set):
    initial_log_states = jnp.log(jnp.stack([p.z_0, p.k_0], axis=1))
    budget = data_set.train_points - 1
    return batched_rollout(
        policy,
        initial_log_states,
        data_set.train_T,
        budget,
        data_set.train_T / budget,
        p,
    )


@partial(jax.jit, static_argnames=("T", "budget", "dt_0"))
def rollout_steps(policy, initial_log_states, T, budget, dt_0, p):
    return batched_rollout(policy, initial_log_states, T, budget, dt_0, p)


def collocation_residuals(policy, p, data_set):
    solution = collocation_rollout(policy, p, data_set)
    log_states = solution.xs
    euler = euler_residuals(policy, log_states.reshape(-1, 2), p).reshape(
        log_states.shape[:2]
    )
    # Mean squared residual, so the loss threshold means the same thing at any
    # number of trajectories or collocation points.
    count = jnp.asarray(euler.size, dtype=euler.dtype)
    # A nonfinite trajectory did not cover the requested economic horizon. Make
    # that a loud optimizer failure instead of fitting a prefix.
    return jnp.where(
        solution.ok[:, None],
        euler / jnp.sqrt(count),
        jnp.full_like(euler, jnp.inf),
    )


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.fixed_state)
    return collocation_residuals(policy, p, args.data_set).reshape(-1)


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
    return jnp.max(jnp.where(valid, normalized_log_tvc, -jnp.inf)), solution.ok


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
    z_paths = np.asarray(initial_conditions)[:, :1] * np.exp(g * times[None, :])
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
        kappa_samples.append((k_path / z_path).reshape(-1))
        c_tilde_samples.append((c_path / z_path).reshape(-1))

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


def collocation_step_diagnostics(times):
    times = np.asarray(times)
    n_trajectories, states_per_trajectory = times.shape
    return {
        "num_collocation_states": int(n_trajectories * states_per_trajectory),
        "collocation_states_per_trajectory": int(states_per_trajectory),
        "collocation_dt": (
            float(np.diff(times, axis=1).max()) if states_per_trajectory > 1 else np.nan
        ),
    }


def neoclassical_growth_nn_recursive(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    sigma_crra: float = 1.0,
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
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
    dtype = jnp.float64 if use_float64 else jnp.float32
    if stop_set.train_loss_threshold is None:
        threshold = 1e-14 if dtype == jnp.float64 else 1e-10
        stop_set = replace(stop_set, train_loss_threshold=threshold)
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
    if stop_set.train_loss_threshold is None or not (
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
    test_initial_conditions = draw_initial_conditions(
        test_z_center,
        test_k_center,
        ic_set.num_test_trajectories,
        ic_set.test_log_z_variance,
        ic_set.test_log_k_variance,
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

    train_solution = collocation_rollout(policy, p, data_set)
    train_log_states = train_solution.xs
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_solution.ts,
        train_initial_conditions,
    )
    train_df["collocation_step_index"] = train_df.groupby("trajectory").cumcount()
    collocation_diagnostics = collocation_step_diagnostics(train_solution.ts)

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
    max_train_euler = float(train_df["euler_residual"].abs().max())
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))

    status = int(result.status)
    train_kappa_0 = np.asarray(
        train_initial_conditions[:, 1] / train_initial_conditions[:, 0]
    )
    test_kappa_0 = np.asarray(
        test_initial_conditions[:, 1] / test_initial_conditions[:, 0]
    )
    train_kappa = train_df["kappa"].to_numpy()
    collocation_kappa_spread = float(np.max(train_kappa) / np.min(train_kappa))
    results = {
        "variant": "recursive_nn",
        "policy": "delta_k_unit_softplus",
        "state_coordinate": "log(k / z)",
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
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric": "identity_parameter_metric",
        "selection": "minimum_identity_metric_parameter_step",
        "train_loss": float(result.info.loss),
        "max_train_euler": max_train_euler,
        "max_test_euler": max_test_euler,
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
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        **collocation_diagnostics,
        "collocation_path_ok": bool(jnp.all(train_solution.ok)),
        "collocation_kappa_spread": collocation_kappa_spread,
        "train_kappa_0_min": float(train_kappa_0.min()),
        "train_kappa_0_max": float(train_kappa_0.max()),
        "test_kappa_0_min": float(test_kappa_0.min()),
        "test_kappa_0_max": float(test_kappa_0.max()),
        "train_log_z_variance": ic_set.train_log_z_variance,
        "train_log_k_variance": ic_set.train_log_k_variance,
        "test_log_z_variance": ic_set.test_log_z_variance,
        "test_log_k_variance": ic_set.test_log_k_variance,
        "ode_solver": "rk4",
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
        "nn_set": asdict(nn_set),
        "data_set": data_hyperparameters,
        "ic_set": asdict(ic_set),
        "lm_set": asdict(lm_set),
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
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn_recursive)
