from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import struct
from jax.scipy.linalg import solve_triangular
from nlls_gram import (
    LevenbergMarquardt,
    LMSolveAction,
    LMStatus,
)
from tinydiffeq import (
    IController,
    Rodas5P,
    SaveAt,
    Tsit5,
    solve_ode,
)

from kernels import rbf_state_features, rbf_state_gram
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from utilities import unit_softplus

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
            **{name: jnp.asarray(value, dtype=dtype) for name, value in kwargs.items()}
        )


@struct.dataclass
class ResidualArgs:
    centers: jax.Array
    center_times: jax.Array
    active: jax.Array
    gram_cholesky: jax.Array


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 100.0
    ode_solver: Literal["tsit5", "rodas5p"] = "tsit5"
    # At most adaptive_steps attempts; accepted advances plus the initial state
    # form the active prefix, while rejected attempts are omitted.
    adaptive_steps: int | None = None
    dt: float = 0.25
    # A gradual cold-start ramp keeps the differentiated rollout informative;
    # it does not cap the large timesteps selected near the steady state.
    step_growth_max: float = 3.0
    rtol: float = 1e-8
    atol: float = 1e-10
    test_T: float = 100.0
    test_points: int = 401
    test_steps: int = 256
    train_tvc_T: float = 200.0
    tvc_points: int = 129
    tvc_steps: int = 256
    benchmark_T: float = 200.0
    benchmark_points: int = 800
    policy_grid_size: int = 100


@jax.tree_util.register_static
@dataclass(frozen=True)
class StateKernelSettings:
    amplitude: float = 1.0
    lengthscale: float = 0.5
    metric_epsilon: float = 1e-8
    metric_dtype: Literal["float32", "float64"] = "float64"
    center_relative_tolerance: float = 5e-3
    transfer_relative_tolerance: float = 2e-5


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-4
    damping_decrease: float = 0.4
    damping_increase: float = 4.0
    geodesic_acceleration: bool = False
    geodesic_acceptance_ratio: float = 0.75


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 100
    train_loss_threshold: float = 1e-14
    log_tvc_threshold: float = 0.0


def production(z, k, p):
    kappa = k / z
    return z * kappa**p.a


def mpk(z, k, p):
    kappa = k / z
    return p.a * kappa ** (p.a - 1.0)


def gram_factor(centers, kernel_set, active=None):
    if active is None:
        active = jnp.ones(centers.shape[0], dtype=bool)
    q_centers = centers[:, 1] - centers[:, 0]
    gram = rbf_state_gram(
        q_centers,
        jnp.asarray(kernel_set.amplitude, q_centers.dtype),
        jnp.asarray(kernel_set.lengthscale, q_centers.dtype),
    )
    active_outer = active[:, None] & active[None, :]
    diagonal = jnp.where(
        active,
        jnp.asarray(kernel_set.metric_epsilon, gram.dtype),
        jnp.asarray(1.0, gram.dtype),
    )
    shifted = jnp.where(active_outer, gram, 0.0) + jnp.diag(diagonal)
    return shifted, jnp.linalg.cholesky(shifted)


def whitened_coefficients(beta, args):
    metric_dtype = args.gram_cholesky.dtype
    return solve_triangular(
        args.gram_cholesky.T,
        beta.astype(metric_dtype),
        lower=False,
    )


def raw_policy_from_coefficients(coefficients, args, log_kappa, kernel_set):
    dtype = log_kappa.dtype
    features = rbf_state_features(
        log_kappa,
        (args.centers[:, 1] - args.centers[:, 0]).astype(dtype),
        jnp.asarray(kernel_set.amplitude, dtype),
        jnp.asarray(kernel_set.lengthscale, dtype),
    )
    return (features * args.active.astype(dtype)) @ coefficients.astype(dtype)


def raw_policy(beta, args, log_kappa, kernel_set):
    return raw_policy_from_coefficients(
        whitened_coefficients(beta, args), args, log_kappa, kernel_set
    )


def investment_from_coefficients(coefficients, args, log_state, p, kernel_set):
    log_z, log_k = log_state
    z = jnp.exp(log_z)
    kappa_0 = p.k_0 / p.z_0
    anchor = jnp.maximum((p.delta + p.g) * kappa_0, DOMAIN_EPS)
    output_0 = production(p.z_0, p.k_0, p) / p.z_0
    scale = jnp.maximum(output_0 - anchor, DOMAIN_EPS)
    raw = raw_policy_from_coefficients(coefficients, args, log_k - log_z, kernel_set)
    investment_per_z = anchor * unit_softplus((scale / anchor) * raw)
    return z * investment_per_z


def investment(beta, args, log_state, p, kernel_set):
    return investment_from_coefficients(
        whitened_coefficients(beta, args), args, log_state, p, kernel_set
    )


def consumption_from_coefficients(coefficients, args, log_state, p, kernel_set):
    z, k = jnp.exp(log_state)
    return production(z, k, p) - investment_from_coefficients(
        coefficients, args, log_state, p, kernel_set
    )


def consumption(beta, args, log_state, p, kernel_set):
    return consumption_from_coefficients(
        whitened_coefficients(beta, args), args, log_state, p, kernel_set
    )


def log_state_drift(log_state, _, solve_args, p):
    coefficients, args, kernel_set = solve_args
    _, log_k = log_state
    k = jnp.exp(log_k)
    i = investment_from_coefficients(coefficients, args, log_state, p, kernel_set)
    return jnp.stack([p.g, i / k - p.delta])


def ode_components(data_set):
    if data_set.ode_solver == "tsit5":
        return Tsit5(), IController(
            rtol=data_set.rtol,
            atol=data_set.atol,
            dt_min=1e-10,
            factor_max=data_set.step_growth_max,
        )
    if data_set.ode_solver == "rodas5p":
        return Rodas5P(), IController(
            rtol=data_set.rtol,
            atol=data_set.atol,
            dt_min=1e-10,
            factor_max=data_set.step_growth_max,
        )
    raise ValueError(f"unknown ode_solver: {data_set.ode_solver!r}")


def training_attempt_budget(data_set):
    if data_set.adaptive_steps is not None:
        return data_set.adaptive_steps
    return 64 if data_set.ode_solver == "tsit5" else 128


def rollout(beta, args, p, kernel_set, data_set, T, budget, save_at):
    ode_solver, controller = ode_components(data_set)
    initial_state = jnp.log(jnp.stack([p.z_0, p.k_0]))
    coefficients = whitened_coefficients(beta, args)
    return solve_ode(
        log_state_drift,
        ode_solver,
        0.0,
        T,
        initial_state,
        args=(coefficients, args, kernel_set),
        p=p,
        dt_0=data_set.dt,
        controller=controller,
        max_steps=budget,
        save_at=save_at,
    )


def dae_residuals(beta, args, log_states, p, kernel_set):
    coefficients = whitened_coefficients(beta, args)

    def residual(log_state):
        drift = log_state_drift(log_state, 0.0, (coefficients, args, kernel_set), p)
        c, dc_dt = jax.jvp(
            lambda state: consumption_from_coefficients(
                coefficients, args, state, p, kernel_set
            ),
            (log_state,),
            (drift,),
        )
        z, k = jnp.exp(log_state)
        euler = dc_dt - c * (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
        c_scale = production(p.z_0, p.k_0, p)
        return euler / c_scale, jnp.minimum((c - DOMAIN_EPS) / c_scale, 0.0)

    return jax.vmap(residual)(log_states)


def raw_policy_values(beta, args, q, kernel_set):
    coefficients = whitened_coefficients(beta, args)
    return jax.vmap(
        lambda point: raw_policy_from_coefficients(
            coefficients, args, point, kernel_set
        )
    )(q)


STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


def neoclassical_growth_kernel_recursive_adaptive(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    sigma_crra: float = 1.0,
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    kernel_set: StateKernelSettings = StateKernelSettings(),
    lm_set: SolverSettings = SolverSettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64 if use_float64 else jnp.float32
    metric_dtype = {
        "float32": jnp.float32,
        "float64": jnp.float64,
    }[kernel_set.metric_dtype]
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        g=g,
        z_0=z_0,
        k_0=k_0,
    )

    initial_log_state = jnp.log(jnp.asarray([z_0, k_0], dtype=metric_dtype))
    train_attempt_budget = training_attempt_budget(data_set)
    capacity = train_attempt_budget + 1
    provisional_centers = jnp.broadcast_to(initial_log_state, (capacity, 2))
    provisional_active = jnp.arange(capacity) == 0
    _, provisional_cholesky = gram_factor(
        provisional_centers, kernel_set, provisional_active
    )
    provisional_args = ResidualArgs(
        centers=provisional_centers,
        center_times=jnp.zeros(capacity, dtype=metric_dtype),
        active=provisional_active,
        gram_cholesky=provisional_cholesky,
    )
    beta_0 = jnp.zeros(capacity, dtype=dtype)
    initial_solution = rollout(
        beta_0,
        provisional_args,
        p,
        kernel_set,
        data_set,
        data_set.train_T,
        train_attempt_budget,
        SaveAt(steps=True),
    )
    if not bool(initial_solution.ok):
        raise RuntimeError("initial adaptive rollout exhausted its step budget")
    centers = initial_solution.xs.astype(metric_dtype)
    active = initial_solution.accepted
    _, initial_cholesky = gram_factor(centers, kernel_set, active)
    args = ResidualArgs(
        centers=centers,
        center_times=initial_solution.ts.astype(metric_dtype),
        active=active,
        gram_cholesky=initial_cholesky,
    )

    def residual_fn(beta, residual_args, parameters):
        solution = rollout(
            beta,
            residual_args,
            parameters,
            kernel_set,
            data_set,
            data_set.train_T,
            train_attempt_budget,
            SaveAt(steps=True),
        )
        log_states = jnp.where(solution.ok, solution.xs, jnp.inf)
        euler, domain = dae_residuals(
            beta, residual_args, log_states, parameters, kernel_set
        )
        row_active = solution.accepted
        active_scale = jnp.sqrt(2 * jnp.maximum(jnp.sum(row_active), 1))
        residual = (
            jnp.concatenate(
                [
                    jnp.where(row_active, euler, 0.0),
                    jnp.where(row_active, domain, 0.0),
                ]
            )
            / active_scale
        )
        residual = jnp.where(solution.ok, residual, jnp.inf)
        return residual, (log_states, solution.ts, row_active)

    solver = LevenbergMarquardt(
        residual_fn,
        has_aux=True,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
        linear_solve_dtype=metric_dtype,
    )

    def center_callback(ctx):
        solution = rollout(
            ctx.x,
            ctx.args,
            ctx.p,
            kernel_set,
            data_set,
            data_set.train_T,
            train_attempt_budget,
            SaveAt(steps=True),
        )
        proposed = solution.xs.astype(metric_dtype)
        proposed_times = solution.ts.astype(metric_dtype)
        proposed_active = solution.accepted
        finite = solution.ok & jnp.all(jnp.isfinite(proposed))
        # A rejected LM candidate leaves ctx.x unchanged, but the previous
        # basis transfer can still have moved its adaptive mesh. Refresh from
        # the current policy until the policy/mesh pair is a fixed point.
        can_refresh = finite
        candidate_centers = jnp.where(can_refresh, proposed, ctx.args.centers)
        candidate_times = jnp.where(can_refresh, proposed_times, ctx.args.center_times)
        candidate_active = jnp.where(can_refresh, proposed_active, ctx.args.active)
        _, candidate_cholesky = gram_factor(
            candidate_centers, kernel_set, candidate_active
        )

        candidate_args = ResidualArgs(
            centers=candidate_centers,
            center_times=candidate_times,
            active=candidate_active,
            gram_cholesky=candidate_cholesky,
        )
        candidate_q = candidate_centers[:, 1] - candidate_centers[:, 0]
        old_values = raw_policy_values(ctx.x, ctx.args, candidate_q, kernel_set)
        observations = jnp.where(
            candidate_active,
            old_values,
            0.0,
        ).astype(metric_dtype)
        transferred = solve_triangular(
            candidate_cholesky, observations, lower=True
        ).astype(ctx.x.dtype)

        check_q = jnp.concatenate(
            [
                ctx.args.centers[:, 1] - ctx.args.centers[:, 0],
                candidate_q,
            ]
        )
        old_check = raw_policy_values(ctx.x, ctx.args, check_q, kernel_set)
        new_check = raw_policy_values(transferred, candidate_args, check_q, kernel_set)
        check_active = jnp.concatenate([ctx.args.active, candidate_active])
        transfer_error = jnp.max(
            jnp.where(check_active, jnp.abs(new_check - old_check), 0.0)
        ) / (1.0 + jnp.max(jnp.where(check_active, jnp.abs(old_check), 0.0)))
        point_active = candidate_active | ctx.args.active
        center_movement = jnp.max(
            jnp.where(
                point_active[:, None],
                jnp.abs(candidate_centers - ctx.args.centers),
                0.0,
            )
        ) / (
            1.0
            + jnp.max(jnp.where(point_active[:, None], jnp.abs(ctx.args.centers), 0.0))
        )
        count_changed = jnp.any(candidate_active != ctx.args.active)
        basis_changed = count_changed | (
            center_movement > kernel_set.center_relative_tolerance
        )
        refresh = (
            can_refresh
            & basis_changed
            & (transfer_error < kernel_set.transfer_relative_tolerance)
        )
        new_args = jax.tree.map(
            lambda new, old: jnp.where(refresh, new, old),
            candidate_args,
            ctx.args,
        )
        new_beta = jnp.where(refresh, transferred, ctx.x)
        attempted = can_refresh & basis_changed
        num_points = jnp.sum(candidate_active, dtype=jnp.int32)
        user_state = {
            "refresh_count": ctx.user_state["refresh_count"]
            + refresh.astype(jnp.int32),
            "failed_transfers": ctx.user_state["failed_transfers"]
            + (attempted & ~refresh).astype(jnp.int32),
            "last_center_movement": jnp.where(
                can_refresh,
                center_movement,
                ctx.user_state["last_center_movement"],
            ),
            "max_transfer_error": jnp.maximum(
                ctx.user_state["max_transfer_error"],
                jnp.where(attempted, transfer_error, 0.0),
            ),
            "last_num_points": jnp.where(
                can_refresh, num_points, ctx.user_state["last_num_points"]
            ),
            "min_num_points": jnp.where(
                can_refresh,
                jnp.minimum(ctx.user_state["min_num_points"], num_points),
                ctx.user_state["min_num_points"],
            ),
            "max_num_points": jnp.where(
                can_refresh,
                jnp.maximum(ctx.user_state["max_num_points"], num_points),
                ctx.user_state["max_num_points"],
            ),
        }
        return LMSolveAction(x=new_beta, args=new_args, user_state=user_state)

    user_state_0 = {
        "refresh_count": jnp.asarray(0, jnp.int32),
        "failed_transfers": jnp.asarray(0, jnp.int32),
        "last_center_movement": jnp.asarray(jnp.inf, metric_dtype),
        "max_transfer_error": jnp.asarray(0.0, metric_dtype),
        "last_num_points": jnp.sum(active, dtype=jnp.int32),
        "min_num_points": jnp.sum(active, dtype=jnp.int32),
        "max_num_points": jnp.sum(active, dtype=jnp.int32),
    }
    atol = math.sqrt(stop_set.train_loss_threshold)

    start_time = time.perf_counter()
    result = solver.solve(
        beta_0,
        args,
        p=p,
        max_steps=stop_set.max_steps,
        atol=atol,
        callback=center_callback,
        user_state=user_state_0,
        max_steps_is_success=False,
    )
    jax.block_until_ready(result)
    train_time = time.perf_counter() - start_time

    test_ts = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    test_solution = rollout(
        result.x,
        result.args,
        p,
        kernel_set,
        data_set,
        data_set.test_T,
        data_set.test_steps,
        SaveAt(ts=test_ts),
    )
    test_log_states = jnp.where(test_solution.ok, test_solution.xs, jnp.nan)
    test_states = jnp.exp(test_log_states)
    test_z, test_k = test_states[:, 0], test_states[:, 1]
    test_i = jax.vmap(
        lambda state: investment(result.x, result.args, state, p, kernel_set)
    )(test_log_states)
    test_c = jax.vmap(
        lambda state: consumption(result.x, result.args, state, p, kernel_set)
    )(test_log_states)
    test_euler, _ = dae_residuals(result.x, result.args, test_log_states, p, kernel_set)
    test_path_ok = bool(
        test_solution.ok
        and jnp.all(jnp.isfinite(test_states))
        and jnp.min(test_states) > 0.0
        and jnp.min(test_c) > 0.0
    )

    tvc_ts = jnp.linspace(
        data_set.train_T, data_set.train_tvc_T, data_set.tvc_points, dtype=dtype
    )
    tvc_solution = rollout(
        result.x,
        result.args,
        p,
        kernel_set,
        data_set,
        data_set.train_tvc_T,
        data_set.tvc_steps,
        SaveAt(ts=tvc_ts),
    )
    tvc_log_states = jnp.where(tvc_solution.ok, tvc_solution.xs, jnp.nan)
    tvc_c = jax.vmap(
        lambda state: consumption(result.x, result.args, state, p, kernel_set)
    )(tvc_log_states)
    log_tvc = -p.rho_hat * tvc_ts - p.sigma_crra * jnp.log(tvc_c) + tvc_log_states[:, 1]
    max_log_tvc = float(jnp.nanmax(log_tvc))
    terminal_log_tvc = float(log_tvc[-1])
    log_tvc_tail_change = float(log_tvc[-1] - log_tvc[0])
    tvc_tail_declining = bool(log_tvc_tail_change < 0.0)
    transversality_ok = bool(
        tvc_solution.ok
        and max_log_tvc < stop_set.log_tvc_threshold
        and tvc_tail_declining
    )

    final_center_solution = rollout(
        result.x,
        result.args,
        p,
        kernel_set,
        data_set,
        data_set.train_T,
        train_attempt_budget,
        SaveAt(steps=True),
    )
    final_active = final_center_solution.accepted
    point_active = final_active | result.args.active
    center_fixed_point_error = float(
        jnp.max(
            jnp.where(
                point_active[:, None],
                jnp.abs(final_center_solution.xs - result.args.centers),
                0.0,
            )
        )
        / (
            1.0
            + jnp.max(
                jnp.where(point_active[:, None], jnp.abs(result.args.centers), 0.0)
            )
        )
    )
    center_fixed_point_ok = bool(
        jnp.array_equal(final_active, result.args.active)
        and center_fixed_point_error < 10 * kernel_set.center_relative_tolerance
    )
    final_accepted_times = np.asarray(final_center_solution.ts)[
        np.asarray(final_active)
    ]
    accepted_time_steps = np.diff(final_accepted_times)
    min_accepted_time_step = (
        float(np.min(accepted_time_steps)) if accepted_time_steps.size else np.nan
    )
    max_accepted_time_step = (
        float(np.max(accepted_time_steps)) if accepted_time_steps.size else np.nan
    )
    final_gram, _ = gram_factor(result.args.centers, kernel_set, result.args.active)
    active_indices = np.flatnonzero(np.asarray(result.args.active))
    active_gram = np.asarray(final_gram)[np.ix_(active_indices, active_indices)]
    eigenvalues = np.linalg.eigvalsh(active_gram)

    test_df = pd.DataFrame(
        {
            "t": np.asarray(test_ts),
            "z": np.asarray(test_z),
            "k": np.asarray(test_k),
            "c": np.asarray(test_c),
            "i": np.asarray(test_i),
            "euler_residual": np.asarray(test_euler),
        }
    )
    if g == 0.0 and z_0 == 1.0:
        benchmark_grid = np.linspace(
            0.0, data_set.benchmark_T, data_set.benchmark_points
        )
        benchmark_solution = neoclassical_growth_benchmark(
            a,
            delta,
            rho_hat,
            sigma_crra,
            k_0,
            benchmark_grid,
        )
        k_benchmark, c_benchmark = benchmark_solution(np.asarray(test_ts))
        test_df["k_benchmark"] = k_benchmark
        test_df["c_benchmark"] = c_benchmark
        test_df["k_rel_error"] = np.abs(test_df["k"] - k_benchmark) / np.maximum(
            np.abs(k_benchmark), DOMAIN_EPS
        )
        test_df["c_rel_error"] = np.abs(test_df["c"] - c_benchmark) / np.maximum(
            np.abs(c_benchmark), DOMAIN_EPS
        )

    kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1.0 / (a - 1.0))
    kappa_grid = np.linspace(
        max(min(k_0 / z_0, kappa_ss) * 0.9, DOMAIN_EPS),
        max(k_0 / z_0, kappa_ss) * 1.1,
        data_set.policy_grid_size,
    )
    policy_log_states = jnp.stack(
        [
            jnp.zeros(data_set.policy_grid_size, dtype),
            jnp.log(jnp.asarray(kappa_grid, dtype)),
        ],
        axis=1,
    )
    policy_i = jax.vmap(
        lambda state: investment(result.x, result.args, state, p, kernel_set)
    )(policy_log_states)
    policy_c = jax.vmap(
        lambda state: consumption(result.x, result.args, state, p, kernel_set)
    )(policy_log_states)
    policy_df = pd.DataFrame(
        {
            "kappa": kappa_grid,
            "i_kernel": np.asarray(policy_i),
            "c_kernel": np.asarray(policy_c),
        }
    )

    status = int(result.status)
    solver_converged = status == LMStatus.CONVERGED
    results = {
        "variant": "adaptive_state_kernel",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "residual_dtype": str(jnp.dtype(dtype)),
        "metric_dtype": str(jnp.dtype(metric_dtype)),
        "optimizer_metric": "identity_in_whitened_rkhs_coordinates",
        "train_loss": float(result.info.loss),
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_euler))),
        "min_test_c": float(jnp.min(test_c)),
        "test_path_ok": test_path_ok,
        "max_log_tvc_residual": max_log_tvc,
        "terminal_log_tvc_residual": terminal_log_tvc,
        "log_tvc_tail_change": log_tvc_tail_change,
        "tvc_tail_declining": tvc_tail_declining,
        "transversality_ok": transversality_ok,
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(status, "max_steps"),
        "solution_accepted": bool(
            solver_converged and test_path_ok and center_fixed_point_ok
        ),
        "train_time": train_time,
        "total_params": int(result.x.size),
        "num_collocation_states": int(jnp.sum(result.args.active)),
        "collocation_capacity": capacity,
        "integrator_attempt_budget": train_attempt_budget,
        "min_accepted_time_step": min_accepted_time_step,
        "max_accepted_time_step": max_accepted_time_step,
        "min_collocation_states": int(result.user_state["min_num_points"]),
        "max_collocation_states": int(result.user_state["max_num_points"]),
        "last_proposed_collocation_states": int(result.user_state["last_num_points"]),
        "refresh_count": int(result.user_state["refresh_count"]),
        "failed_transfers": int(result.user_state["failed_transfers"]),
        "max_transfer_error": float(result.user_state["max_transfer_error"]),
        "center_fixed_point_error": center_fixed_point_error,
        "center_fixed_point_ok": center_fixed_point_ok,
        "last_center_movement": float(result.user_state["last_center_movement"]),
        "gram_min_eigenvalue": float(eigenvalues[0]),
        "gram_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "ode_solver": data_set.ode_solver,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
    }
    if "k_rel_error" in test_df:
        results.update(
            test_mean_abs_k_rel_error=float(test_df["k_rel_error"].mean()),
            test_mean_abs_c_rel_error=float(test_df["c_rel_error"].mean()),
        )

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    return results, {
        "beta": result.x,
        "basis_args": result.args,
        "parameters": p,
        "test_df": test_df,
        "policy_df": policy_df,
        "hyperparameters": {
            "data_set": asdict(data_set),
            "kernel_set": asdict(kernel_set),
            "lm_set": asdict(lm_set),
            "stop_set": asdict(stop_set),
        },
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_recursive_adaptive)
