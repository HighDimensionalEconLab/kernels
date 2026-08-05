from __future__ import annotations

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
from nlls_gram import LevenbergMarquardt, LMAction, LMStatus, MultiStart
from tinydiffeq import SRA1, EulerMaruyama, Milstein, SaveAt, solve_sde

from neoclassical_growth_nn_recursive import (
    DrawInvestmentPolicy,
    InitialConditionSettings,
    InvestmentPolicyNN,
    InvestmentPolicySettings,
    collocation_step_diagnostics,
    consumption,
    draw_initial_conditions,
    load_investment_policy,
    mpk,
    policy_cache_key,
    production,
    save_investment_policy,
    standardized_initial_states,
    trajectory_frame,
)
from stochastic_growth_benchmark import stochastic_growth_benchmark
from utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from utilities import (
    STATUS_TO_REASON,
    OptimizerSettings,
)

DOMAIN_EPS = 1e-8

SDE_SOLVERS = {
    "sra1": SRA1(),
    "milstein": Milstein(),
    "euler_maruyama": EulerMaruyama(),
}


@struct.dataclass
class StructuralParameters:
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    eta: jax.Array
    sigma_z: jax.Array
    log_z_bar: jax.Array
    z_0: jax.Array
    k_0: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class DataSettings:
    # Fixed-step SDE rollout on the integrator's own states: n points from
    # n - 1 steps including the initial condition, nothing interpolated.
    train_points: int = 65
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = CONSTANTS.train_T
    train_tvc_T: float = CONSTANTS.transversality_T
    test_T: float = CONSTANTS.test_T
    policy_grid_size: int = 100
    benchmark_z_points: int = 161
    benchmark_k_points: int = 1201
    benchmark_z_sd_span: float = 5.0
    benchmark_k_lo: float = 0.4
    benchmark_k_hi: float | None = None


@dataclass(frozen=True)
class StoppingCriteria:
    # max_steps is the LM-update budget per epoch; the noise realization (and
    # the training ICs when their variance is positive) is redrawn at each
    # interior epoch boundary.
    max_steps: int = 250
    train_loss_threshold: float = 1e-14
    num_attempts: int = 4
    log_tvc_threshold: float = 0.0
    num_epochs: int = 3
    epoch_early_stopping: bool = True


# Training ICs, the noise realization, and keys live in args (the epoch
# callback may rewrite args only), so the implicit derivative is conditional
# on the carried batch of initial conditions and shocks.
@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    initial_log_states: jax.Array
    noise: object
    epoch_key: jax.Array
    restart_key: jax.Array
    epoch: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    solver: object = struct.field(pytree_node=False)
    data_set: DataSettings = struct.field(pytree_node=False)


def log_state_drift(policy, log_state, p):
    log_z, log_k = log_state
    k = jnp.exp(log_k)
    investment = policy(log_state, p)
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


def sample_trajectory_noise(
    solver, key, num_trajectories, n_steps, dt, dtype, state_size
):
    reference_state = jnp.zeros(state_size, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    return jax.vmap(
        lambda trajectory_key: solver.sample_noise(
            reference_state, trajectory_key, n_steps, dt, dtype
        )
    )(jax.random.split(key, num_trajectories))


# At each interior epoch boundary: redraw the shock realization (and the
# training ICs when configured), stop on a nonfinite or converged fresh-batch
# loss, else continue with damping reset.
@dataclass(frozen=True)
class EpochCallback:
    num_train_trajectories: int
    log_z_0: float
    log_k_0: float
    log_z_scale: float
    log_k_scale: float
    epoch_early_stopping: bool
    resample_initial_states: bool
    n_steps: int
    dt: float

    def next_epoch(self, ctx):
        epoch_key, draw_key, noise_key = jax.random.split(ctx.args.epoch_key, 3)
        dtype = ctx.args.initial_log_states.dtype
        if self.resample_initial_states:
            initial_log_states = standardized_initial_states(
                draw_key,
                jnp.asarray([self.log_z_0, self.log_k_0], dtype),
                jnp.asarray([self.log_z_scale, self.log_k_scale], dtype),
                self.num_train_trajectories,
            )
        else:
            initial_log_states = ctx.args.initial_log_states
        fresh = ctx.args.replace(
            initial_log_states=initial_log_states,
            noise=sample_trajectory_noise(
                ctx.args.solver,
                noise_key,
                self.num_train_trajectories,
                self.n_steps,
                self.dt,
                dtype,
                2,
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


def log_state_drift_field(log_state, _, args, p):
    policy, structural = p
    return log_state_drift(policy, log_state, structural)


def log_state_diffusion_field(log_state, _, args, p):
    _, structural = p
    return log_state_diffusion(log_state, structural)


def batched_sde_rollout(solver, policy, initial_log_states, noise, T, n_steps, p):
    def solve_one(initial_log_state, trajectory_noise):
        return solve_sde(
            log_state_drift_field,
            log_state_diffusion_field,
            solver,
            0.0,
            T,
            initial_log_state,
            noise=trajectory_noise,
            n_steps=n_steps,
            save_at=SaveAt(steps=True),
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_log_states, noise)


@partial(jax.jit, static_argnames=("data_set",))
def collocation_rollout(solver, policy, initial_log_states, noise, p, data_set):
    return batched_sde_rollout(
        solver,
        policy,
        initial_log_states,
        noise,
        data_set.train_T,
        data_set.train_points - 1,
        p,
    )


@partial(jax.jit, static_argnames=("T", "n_steps"))
def rollout_steps(solver, policy, initial_log_states, noise, T, n_steps, p):
    return batched_sde_rollout(solver, policy, initial_log_states, noise, T, n_steps, p)


def collocation_residuals(solver, policy, initial_log_states, noise, p, data_set):
    solution = collocation_rollout(
        solver, policy, initial_log_states, noise, p, data_set
    )
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
    return collocation_residuals(
        args.solver, policy, args.initial_log_states, args.noise, p, args.data_set
    ).reshape(-1)


def log_tvc_residual(
    solver, policy, initial_log_states, noise, T_start, T_end, p, data_set
):
    solution = rollout_steps(
        solver,
        policy,
        initial_log_states,
        noise,
        T_end,
        data_set.tvc_steps,
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


# The FD consumption policy as a rollout-compatible callable: bilinear
# interpolation on the benchmark grid, edge-clamped, i = f - c.
@struct.dataclass
class FDInterpolatedPolicy:
    log_z_grid: jax.Array
    log_k_grid: jax.Array
    consumption_grid: jax.Array

    def __call__(self, log_state, p):
        index_z = (log_state[0] - self.log_z_grid[0]) / (
            self.log_z_grid[1] - self.log_z_grid[0]
        )
        index_k = (log_state[1] - self.log_k_grid[0]) / (
            self.log_k_grid[1] - self.log_k_grid[0]
        )
        c = jax.scipy.ndimage.map_coordinates(
            self.consumption_grid,
            [index_z[None], index_k[None]],
            order=1,
            mode="nearest",
        )[0]
        z, k = jnp.exp(log_state)
        return production(z, k, p) - c


@lru_cache(maxsize=None)
def cached_stochastic_benchmark(
    a,
    delta,
    rho_hat,
    sigma_crra,
    eta,
    sigma_z,
    log_z_bar,
    num_z_points,
    num_k_points,
    z_sd_span,
    k_lo,
    k_hi,
):
    return stochastic_growth_benchmark(
        a,
        delta,
        rho_hat,
        sigma_crra,
        eta,
        sigma_z,
        log_z_bar=log_z_bar,
        num_z_points=num_z_points,
        num_k_points=num_k_points,
        z_sd_span=z_sd_span,
        k_lo=k_lo,
        k_hi=k_hi,
    )


# Common-noise pathwise errors against the FD policy: the exogenous z path is
# identical under both policies, so the comparison isolates the capital and
# consumption responses.
def add_fd_path_diagnostics(test_df, test_values, fd_values, test_initial_conditions):
    k_benchmark = np.asarray(fd_values["k"])
    c_benchmark = np.asarray(fd_values["c"])
    k_error = (np.asarray(test_values["k"]) - k_benchmark) / np.abs(k_benchmark).clip(
        min=DOMAIN_EPS
    )
    c_error = (np.asarray(test_values["c"]) - c_benchmark) / np.abs(c_benchmark).clip(
        min=DOMAIN_EPS
    )
    columns = {
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "i_benchmark": np.asarray(fd_values["i"]),
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
            "mean_k_abs_rel_error": np.mean(np.abs(k_error), axis=1),
            "median_k_abs_rel_error": np.median(np.abs(k_error), axis=1),
            "mean_c_abs_rel_error": np.mean(np.abs(c_error), axis=1),
            "median_c_abs_rel_error": np.median(np.abs(c_error), axis=1),
        }
    )
    return k_error, c_error, test_summary_df


# nnx.jit so one compiled evaluation serves both the nnx policy and the FD
# interpolated policy pytree.
@nnx.jit
def trajectory_values(policy, log_states, p):
    shape = log_states.shape[:-1]
    flat_states = log_states.reshape(-1, 2)
    investment = jax.vmap(lambda state: policy(state, p))(flat_states).reshape(shape)
    c = jax.vmap(lambda state: consumption(policy, state, p))(flat_states).reshape(
        shape
    )
    euler, diffusion_term = euler_residual_terms(policy, flat_states, p)
    z, k = jnp.moveaxis(jnp.exp(log_states), -1, 0)
    return {
        "z": z,
        "k": k,
        "kappa": k / z,
        "i": investment,
        "c": c,
        "dlog_z": p.eta * (p.log_z_bar - jnp.log(z)),
        "dk": investment - p.delta * k,
        "euler_residual": euler.reshape(shape),
        "diffusion_term": diffusion_term.reshape(shape),
    }


# The deterministic model of the shared NEOCLASSICAL_GROWTH_CONSTANTS
# (utilities.py) plus its OU log-TFP. The default innovation scale is the
# high-variance sensitivity column sigma = 0.035 of Aruoba et al. (ergodic
# sd ~ 0.112); their benchmark sigma = 0.007 (sigma_z_benchmark) is nearly
# certainty equivalent.
def stochastic_growth_nn_recursive(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    eta: float = CONSTANTS.eta,
    sigma_z: float = CONSTANTS.sigma_z_high,
    z_bar: float = 1.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    data_set: DataSettings = DataSettings(),
    # The initial log z spread matches the ergodic variance (sigma_z^2/(2 eta)).
    ic_set: InitialConditionSettings = InitialConditionSettings(
        num_train_trajectories=16,
        train_log_z_variance=CONSTANTS.z_ergodic_variance_high,
        train_log_k_variance=0.01,
    ),
    hc_set: InvestmentPolicySettings = InvestmentPolicySettings(
        homothetic_representation=False
    ),
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
    if rho_hat <= 0.0:
        raise ValueError("rho_hat must be positive")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if sigma_z < 0.0:
        raise ValueError("sigma_z must be nonnegative")
    if z_bar <= 0.0:
        raise ValueError("z_bar must be positive")
    if sde_solver not in SDE_SOLVERS:
        raise ValueError(f"sde_solver must be one of {sorted(SDE_SOLVERS)}")
    # Mean reversion in z fixes a scale, so the homothetic representation is
    # misspecified by construction for this model.
    if hc_set.homothetic_representation:
        raise ValueError(
            "the stochastic model requires homothetic_representation=False"
        )
    if data_set.train_T <= 0.0:
        raise ValueError("train_T must be positive")
    if data_set.train_points < 2:
        raise ValueError("train_points must be at least two")
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

    solver = SDE_SOLVERS[sde_solver]
    n_train_steps = data_set.train_points - 1
    n_test_steps = data_set.test_points - 1
    train_dt = data_set.train_T / n_train_steps
    test_T = data_set.test_T

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
        eta=eta,
        sigma_z=sigma_z,
        log_z_bar=math.log(z_bar),
        z_0=train_initial_conditions[:, 0],
        k_0=train_initial_conditions[:, 1],
    )

    fit_start = time.perf_counter()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    graphdef, _, fixed_state = nnx.split(
        InvestmentPolicyNN(
            hc_set, z_scale=z_bar, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(0)
        ),
        nnx.Param,
        ...,
    )
    # The initial training draws consume train_ic_key itself; the epoch,
    # restart, and noise streams chain from independent fold-ins of the same
    # key, and the test noise streams chain from test_ic_key.
    train_noise_key = jax.random.fold_in(train_ic_key, 3)
    tvc_noise_key = jax.random.fold_in(train_ic_key, 4)
    test_noise_key = jax.random.fold_in(test_ic_key, 1)
    test_tvc_noise_key = jax.random.fold_in(test_ic_key, 2)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        fixed_state=fixed_state,
        initial_log_states=jnp.log(train_initial_conditions),
        noise=sample_trajectory_noise(
            solver,
            train_noise_key,
            ic_set.num_train_trajectories,
            n_train_steps,
            train_dt,
            dtype,
            2,
        ),
        epoch_key=jax.random.fold_in(train_ic_key, 1),
        restart_key=jax.random.fold_in(train_ic_key, 2),
        epoch=jnp.asarray(0, dtype=jnp.int32),
        train_loss_threshold=jnp.asarray(stop_set.train_loss_threshold, dtype=dtype),
        epoch_steps=jnp.asarray(stop_set.max_steps, dtype=jnp.int32),
        total_steps=jnp.asarray(
            stop_set.num_epochs * stop_set.max_steps, dtype=jnp.int32
        ),
        solver=solver,
        data_set=data_set,
    )
    # Two-part artifact key <problem>-<run>: the problem hash covers the deep
    # structural parameters, dtype, and the training initial-condition design
    # shared by any solver of the same problem, so a consumer can glob
    # *_policy_<problem>-*.npz for a reference solution; the run hash covers
    # the training minutiae. Test/benchmark settings enter neither, so
    # changing them still reuses a cached policy.
    problem_config = problem_cache_config(
        a,
        delta,
        rho_hat,
        sigma_crra,
        eta,
        sigma_z,
        z_bar,
        z_0,
        k_0,
        use_float64,
        ic_set,
    )
    run_config = {
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "sde_solver": sde_solver,
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
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
        / f"stochastic_growth_nn_recursive_policy_{cache_key}.npz"
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
        lm_solver = LevenbergMarquardt(
            residual_fn,
            init_damping=opt_set.init_damping,
            damping_decrease=opt_set.damping_decrease,
            damping_increase=opt_set.damping_increase,
            geodesic_acceleration=opt_set.geodesic_acceleration,
            geodesic_acceptance_ratio=opt_set.geodesic_acceptance_ratio,
        )
        _, theta_0, _ = nnx.split(
            InvestmentPolicyNN(
                hc_set, z_scale=z_bar, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(init_key)
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
            resample_initial_states=(
                ic_set.train_log_z_variance > 0.0 or ic_set.train_log_k_variance > 0.0
            ),
            n_steps=n_train_steps,
            dt=train_dt,
        )
        draw = DrawInvestmentPolicy(hc_set, dtype, z_bar, k_0)
        # The per-step tolerance is sound only while the training batch never
        # changes; across epochs, convergence is judged at the boundaries
        # instead.
        atol = (
            math.sqrt(stop_set.train_loss_threshold)
            if stop_set.num_epochs == 1
            else 0.0
        )

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
                draw=draw,
            ),
            max_steps_is_success=not stop_set.epoch_early_stopping,
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
        policy = nnx.merge(graphdef, theta_final, fixed_state)

    # Training diagnostics use the batch (ICs and noise) the solution was
    # returned on.
    train_batch_log_states = solve_args.initial_log_states
    train_batch_initial_conditions = jnp.exp(train_batch_log_states)
    train_solution = collocation_rollout(
        solver, policy, train_batch_log_states, solve_args.noise, p, data_set
    )
    train_log_states = train_solution.xs
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_solution.ts,
        train_batch_initial_conditions,
    )
    train_df["collocation_step_index"] = train_df.groupby("trajectory").cumcount()
    collocation_diagnostics = collocation_step_diagnostics(train_solution.ts)

    test_initial_log_states = jnp.log(test_initial_conditions)
    test_noise = sample_trajectory_noise(
        solver,
        test_noise_key,
        ic_set.num_test_trajectories,
        n_test_steps,
        test_T / n_test_steps,
        dtype,
        2,
    )
    test_solution = rollout_steps(
        solver,
        policy,
        test_initial_log_states,
        test_noise,
        test_T,
        n_test_steps,
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

    train_tvc_noise = sample_trajectory_noise(
        solver,
        tvc_noise_key,
        ic_set.num_train_trajectories,
        data_set.tvc_steps,
        data_set.train_tvc_T / data_set.tvc_steps,
        dtype,
        2,
    )
    max_log_tvc, train_terminal_log_tvc, tvc_ok = log_tvc_residual(
        solver,
        policy,
        train_batch_log_states,
        train_tvc_noise,
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    max_log_tvc = float(max_log_tvc)
    transversality_ok = bool(
        jnp.all(tvc_ok) and max_log_tvc < stop_set.log_tvc_threshold
    )
    test_tvc_noise = sample_trajectory_noise(
        solver,
        test_tvc_noise_key,
        ic_set.num_test_trajectories,
        data_set.tvc_steps,
        data_set.train_tvc_T / data_set.tvc_steps,
        dtype,
        2,
    )
    _, test_terminal_log_tvc, test_tvc_ok = log_tvc_residual(
        solver,
        policy,
        test_initial_log_states,
        test_tvc_noise,
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

    z_ergodic_sd = sigma_z / math.sqrt(2.0 * eta)
    observed_log_z = np.concatenate(
        [
            np.asarray(train_log_states[..., 0]).reshape(-1),
            np.asarray(test_log_states[..., 0]).reshape(-1),
        ]
    )
    observed_log_k = np.concatenate(
        [
            np.asarray(train_log_states[..., 1]).reshape(-1),
            np.asarray(test_log_states[..., 1]).reshape(-1),
        ]
    )
    observed_log_z = observed_log_z[np.isfinite(observed_log_z)]
    observed_log_k = observed_log_k[np.isfinite(observed_log_k)]

    results = {
        "variant": "recursive_nn_stochastic",
        "policy": "delta_k0_unit_softplus",
        "state_coordinate": "(log z, log k)",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "z_ergodic_sd": z_ergodic_sd,
        "z_0": z_0,
        "k_0": k_0,
        "test_z_0": test_z_center,
        "test_k_0": test_k_center,
        "seed": seed,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(r.dtype),
        "sde_solver": sde_solver,
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
        "max_abs_diffusion_term": float(
            jnp.max(jnp.abs(test_values["diffusion_term"]))
        ),
        "mean_abs_diffusion_term": float(
            jnp.mean(jnp.abs(test_values["diffusion_term"]))
        ),
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
        "observed_log_z_min": float(observed_log_z.min()),
        "observed_log_z_max": float(observed_log_z.max()),
        "observed_log_z_p025": float(np.quantile(observed_log_z, 0.025)),
        "observed_log_z_p975": float(np.quantile(observed_log_z, 0.975)),
        "observed_log_k_min": float(observed_log_k.min()),
        "observed_log_k_max": float(observed_log_k.max()),
        "train_log_z_variance": ic_set.train_log_z_variance,
        "train_log_k_variance": ic_set.train_log_k_variance,
        "test_log_z_variance": test_log_z_variance,
        "test_log_k_variance": test_log_k_variance,
        "train_points": data_set.train_points,
        "train_steps": n_train_steps,
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    # FD-HJBE reference: policy-grid errors on the realized state box and
    # common-noise pathwise errors on the test trajectories.
    fd = cached_stochastic_benchmark(
        a,
        delta,
        rho_hat,
        sigma_crra,
        eta,
        sigma_z,
        math.log(z_bar),
        data_set.benchmark_z_points,
        data_set.benchmark_k_points,
        data_set.benchmark_z_sd_span,
        data_set.benchmark_k_lo,
        data_set.benchmark_k_hi,
    )
    fd_policy = FDInterpolatedPolicy(
        log_z_grid=jnp.asarray(fd["log_z_grid"], dtype=dtype),
        log_k_grid=jnp.asarray(fd["log_k_grid"], dtype=dtype),
        consumption_grid=jnp.asarray(fd["c"], dtype=dtype),
    )
    fd_solution = rollout_steps(
        solver,
        fd_policy,
        test_initial_log_states,
        test_noise,
        test_T,
        n_test_steps,
        p,
    )
    fd_values = trajectory_values(fd_policy, fd_solution.xs, p)
    k_error, c_error, test_summary_df = add_fd_path_diagnostics(
        test_df, test_values, fd_values, test_initial_conditions
    )
    results.update(
        fd_benchmark_iterations=int(fd["iterations"]),
        # The FD reference's own residual level in the same metric: NN-vs-FD
        # relative errors are only interpretable down to this scale, so
        # residuals are the headline accuracy metric whenever the NN's are
        # smaller.
        fd_test_euler_mean_abs=float(jnp.mean(jnp.abs(fd_values["euler_residual"]))),
        fd_test_euler_max=float(jnp.max(jnp.abs(fd_values["euler_residual"]))),
        test_mean_abs_k_rel_error=float(np.mean(np.abs(k_error))),
        test_mean_abs_c_rel_error=float(np.mean(np.abs(c_error))),
        test_median_abs_k_rel_error=float(np.median(np.abs(k_error))),
        test_median_abs_c_rel_error=float(np.median(np.abs(c_error))),
        test_median_trajectory_mean_abs_k_rel_error=float(
            test_summary_df["mean_k_abs_rel_error"].median()
        ),
        test_median_trajectory_mean_abs_c_rel_error=float(
            test_summary_df["mean_c_abs_rel_error"].median()
        ),
        test_max_abs_k_rel_error=float(np.max(np.abs(k_error))),
        test_max_abs_c_rel_error=float(np.max(np.abs(c_error))),
    )

    # Policy-function comparison on the realized state box: log k grid over
    # the observed range, z rows at the observed log z quantiles.
    log_k_lo = float(observed_log_k.min())
    log_k_hi = float(observed_log_k.max())
    log_k_grid = np.linspace(log_k_lo, log_k_hi, data_set.policy_grid_size)
    z_row_quantiles = (0.05, 0.5, 0.95)
    z_rows = np.quantile(observed_log_z, z_row_quantiles)
    grid_log_states = jnp.asarray(
        np.stack(
            [
                np.stack(
                    [np.full_like(log_k_grid, z_row), log_k_grid],
                    axis=-1,
                )
                for z_row in z_rows
            ]
        ),
        dtype=dtype,
    )
    grid_values = trajectory_values(policy, grid_log_states, p)
    c_fd_grid = fd["interpolate_consumption"](
        np.asarray(grid_log_states[..., 0]), np.asarray(grid_log_states[..., 1])
    )
    c_grid_error = (np.asarray(grid_values["c"]) - c_fd_grid) / np.abs(c_fd_grid).clip(
        min=DOMAIN_EPS
    )
    policy_rows = []
    for row, (tag, z_row) in enumerate(zip(("lo", "mid", "hi"), z_rows, strict=True)):
        policy_rows.append(
            pd.DataFrame(
                {
                    "z_tag": tag,
                    "log_z": z_row,
                    "z": np.exp(z_row),
                    "k": np.exp(log_k_grid),
                    "c_nn": np.asarray(grid_values["c"][row]),
                    "i_nn": np.asarray(grid_values["i"][row]),
                    "c_fd": c_fd_grid[row],
                    "c_rel_error": c_grid_error[row],
                    "c_abs_rel_error": np.abs(c_grid_error[row]),
                }
            )
        )
    policy_df = pd.concat(policy_rows, ignore_index=True)
    results.update(
        policy_mean_abs_c_rel_error=float(np.mean(np.abs(c_grid_error))),
        policy_max_abs_c_rel_error=float(np.max(np.abs(c_grid_error))),
    )

    hyperparameters = {
        "hc_set": asdict(hc_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "eta": eta,
            "sigma_z": sigma_z,
            "z_bar": z_bar,
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
    structural = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "log_z_bar": math.log(z_bar),
    }
    policy_extra = {
        "model": "stochastic_growth_nn_recursive",
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
    jsonargparse.CLI(stochastic_growth_nn_recursive)
