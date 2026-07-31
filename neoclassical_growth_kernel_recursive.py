from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import struct
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt

from kernels import rbf_state_features, rbf_state_gram, rkhs_norm_squared
from neoclassical_growth_nn_recursive import (
    InitialConditionSettings,
    StructuralParameters,
    add_benchmark_path_diagnostics,
    benchmark_policy_grid,
    benchmark_trajectories,
    draw_test_initial_conditions,
    euler_residuals,
    initial_condition_array,
    log_tvc_residual,
    positive_policy_grid,
    production,
    rollout_steps,
    trajectory_frame,
    trajectory_values,
)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


STATUS_TO_REASON = {
    LMStatus.CONVERGED: "converged",
    LMStatus.NONFINITE: "nonfinite",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class DataSettings:
    collocation_source: Literal["exact_path"] = "exact_path"
    train_T: float = 100.0
    # Number of fixed RKHS centers placed along the exact path. Unlike the
    # recursive NN, nothing is integrated during training, so this is a
    # representation size rather than a collocation grid over a rollout.
    train_points: int = 65
    # Held-out paths are fixed-step RK4 read off the integrator's own states:
    # ``test_points`` points from ``test_points - 1`` steps, initial condition
    # included, so nothing is interpolated.
    test_points: int = 41
    tvc_steps: int = 256
    train_tvc_T: float = 200.0
    test_T: float = 40.0
    num_test_trajectories: int = 50
    policy_grid_size: int = 100
    benchmark_T: float = 200.0
    benchmark_points: int = 801


@jax.tree_util.register_static
@dataclass(frozen=True)
class StateKernelSettings:
    amplitude: float = 1.0
    lengthscale: float = 0.4
    metric_epsilon: float = 3e-5
    initial_investment_share: float = 0.2


@dataclass(frozen=True)
class SolverSettings:
    ridge: float = 1e-12
    init_damping: float = 1e-6
    damping_decrease: float = 0.25
    damping_increase: float = 4.0
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.75


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 100
    residual_atol: float = 5e-6
    gtol: float = 1e-7
    train_euler_tolerance: float = 2e-5
    test_k_rel_error_tolerance: float = 5e-3
    test_c_rel_error_tolerance: float = 5e-3
    log_tvc_threshold: float = 0.0


@struct.dataclass
class ResidualArgs:
    centers: jax.Array
    train_log_states: jax.Array
    gram: jax.Array


def validate_settings(data_set, kernel_set, lm_set, ic_set):
    if data_set.collocation_source != "exact_path":
        raise ValueError(f"unknown collocation_source: {data_set.collocation_source!r}")
    if data_set.train_points < 2:
        raise ValueError("train_points must be at least two")
    if data_set.train_T <= 0.0 or data_set.test_T <= 0.0:
        raise ValueError("train_T and test_T must be positive")
    if data_set.test_points < 2 or data_set.tvc_steps < 1:
        raise ValueError("test_points must be at least two and tvc_steps positive")
    if data_set.train_tvc_T <= data_set.train_T:
        raise ValueError("train_tvc_T must exceed train_T")
    if data_set.benchmark_T < max(data_set.train_T, data_set.test_T):
        raise ValueError("benchmark_T must cover both train and test horizons")
    if data_set.benchmark_points < data_set.train_points:
        raise ValueError("benchmark_points must be at least train_points")
    if kernel_set.amplitude <= 0.0 or kernel_set.lengthscale <= 0.0:
        raise ValueError("kernel amplitude and lengthscale must be positive")
    if kernel_set.metric_epsilon <= 0.0:
        raise ValueError("metric_epsilon must be positive")
    if not 0.0 < kernel_set.initial_investment_share < 1.0:
        raise ValueError(
            "initial_investment_share must lie strictly between zero and one"
        )
    if lm_set.ridge <= 0.0:
        raise ValueError("ridge must be positive")
    if ic_set.test_log_variance < 0.0:
        raise ValueError("test_log_variance must be nonnegative")


def fixed_collocation_data(
    a,
    delta,
    rho_hat,
    sigma_crra,
    g,
    initial_conditions,
    data_set,
    *,
    dtype,
):
    if data_set.collocation_source != "exact_path":
        raise ValueError(f"unknown collocation_source: {data_set.collocation_source!r}")
    train_times = np.linspace(0.0, data_set.train_T, data_set.train_points)
    z, k, c = benchmark_trajectories(
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        initial_conditions,
        train_times,
        data_set,
    )
    log_states = np.stack([np.log(z), np.log(k)], axis=-1)
    return (
        jnp.asarray(np.broadcast_to(train_times, k.shape), dtype=dtype),
        jnp.asarray(log_states, dtype=dtype),
        jnp.asarray(c, dtype=dtype),
    )


def build_residual_args(train_log_states, kernel_set):
    flat_states = train_log_states.reshape(-1, 2)
    centers = flat_states[:, 1] - flat_states[:, 0]
    gram = rbf_state_gram(
        centers,
        jnp.asarray(kernel_set.amplitude, centers.dtype),
        jnp.asarray(kernel_set.lengthscale, centers.dtype),
    )
    gram = (gram + gram.T) / 2.0
    return ResidualArgs(
        centers=centers,
        train_log_states=flat_states,
        gram=gram,
    )


def raw_policy(theta, args, log_state, kernel_set):
    alpha, i_const = theta
    q = log_state[1] - log_state[0]
    features = rbf_state_features(
        q,
        args.centers,
        jnp.asarray(kernel_set.amplitude, q.dtype),
        jnp.asarray(kernel_set.lengthscale, q.dtype),
    )
    return i_const + features @ alpha


def investment(theta, args, log_state, p, kernel_set):
    z, k = jnp.exp(log_state)
    investment_share = jax.nn.sigmoid(raw_policy(theta, args, log_state, kernel_set))
    return production(z, k, p) * investment_share


@struct.dataclass
class KernelInvestmentPolicy:
    theta: tuple
    args: ResidualArgs
    p: StructuralParameters
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)

    def __call__(self, log_state, p):
        return investment(self.theta, self.args, log_state, p, self.kernel_set)


def make_policy(theta, args, p, kernel_set):
    return KernelInvestmentPolicy(
        theta=theta,
        args=args,
        p=p,
        kernel_set=kernel_set,
    )


def residual_fn(theta, args, p, kernel_set):
    policy = make_policy(theta, args, p, kernel_set)
    residual = euler_residuals(policy, args.train_log_states, p)
    return residual / jnp.sqrt(residual.size)


def neoclassical_growth_kernel_recursive(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    sigma_crra: float = 1.0,
    g: float = 0.02,
    z_0: float | list[float] = 1.0,
    k_0: float | list[float] = 1.0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: StateKernelSettings = StateKernelSettings(),
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
    validate_settings(data_set, kernel_set, lm_set, ic_set)
    n_test_steps = data_set.test_points - 1
    test_T = data_set.test_T
    if data_set.benchmark_T < test_T:
        raise ValueError("benchmark_T must cover the effective test_T")

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
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        kappa_thr=math.inf,
    )
    kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1.0 / (a - 1.0))

    fit_start = time.perf_counter()
    train_times, train_log_states, train_c_benchmark = fixed_collocation_data(
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        train_initial_conditions,
        data_set,
        dtype=dtype,
    )
    args = build_residual_args(train_log_states, kernel_set)
    n_train = args.centers.size
    shifted_gram = args.gram + kernel_set.metric_epsilon * jnp.eye(n_train, dtype=dtype)
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(shifted_gram, upper=True),
        repeats=1,
    )
    initial_logit = math.log(
        kernel_set.initial_investment_share
        / (1.0 - kernel_set.initial_investment_share)
    )
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.asarray(initial_logit, dtype=dtype),
    )
    solver = RidgeLevenbergMarquardt(
        partial(residual_fn, kernel_set=kernel_set),
        metric=metric,
        ridge=lm_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )

    solve_start = time.perf_counter()
    setup_time = solve_start - fit_start
    result = solver.solve(
        theta_0,
        args,
        p=p,
        max_steps=stop_set.max_steps,
        atol=stop_set.residual_atol,
        gtol=stop_set.gtol,
        max_steps_is_success=False,
    )
    jax.block_until_ready(result)
    solve_time = time.perf_counter() - solve_start
    total_fit_time = time.perf_counter() - fit_start

    theta = result.x
    alpha, i_const = theta
    policy = make_policy(theta, args, p, kernel_set)
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_times,
        train_initial_conditions,
    )
    train_c_benchmark_np = np.asarray(train_c_benchmark).reshape(-1)
    train_df["c_benchmark"] = train_c_benchmark_np
    train_df["i_benchmark"] = (
        np.asarray(train_df["z"] * train_df["kappa"] ** a) - train_c_benchmark_np
    )
    train_df["fixed_exact_center"] = True

    test_solution = rollout_steps(
        policy,
        jnp.log(test_initial_conditions),
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
        and jnp.min(test_values["c"]) > 0.0
    )
    test_df = trajectory_frame(test_values, test_solution.ts, test_initial_conditions)

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

    max_log_tvc, tvc_ok = log_tvc_residual(
        policy,
        jnp.log(train_initial_conditions),
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    max_log_tvc = float(max_log_tvc)
    transversality_ok = bool(
        jnp.all(tvc_ok) and max_log_tvc < stop_set.log_tvc_threshold
    )

    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))
    mean_k_error = float(np.mean(np.abs(k_error)))
    mean_c_error = float(np.mean(np.abs(c_error)))
    status = int(result.status)
    rejection_reasons = []
    if status != LMStatus.CONVERGED:
        rejection_reasons.append(STATUS_TO_REASON.get(status, "unknown_solver_status"))
    if max_train_euler > stop_set.train_euler_tolerance:
        rejection_reasons.append("train_euler")
    if not test_path_ok:
        rejection_reasons.append("test_path")
    if not transversality_ok:
        rejection_reasons.append("transversality")
    if mean_k_error > stop_set.test_k_rel_error_tolerance:
        rejection_reasons.append("benchmark_k_path")
    if mean_c_error > stop_set.test_c_rel_error_tolerance:
        rejection_reasons.append("benchmark_c_path")

    gram_eigenvalues = jnp.linalg.eigvalsh(args.gram)
    train_kappa_0 = np.asarray(
        train_initial_conditions[:, 1] / train_initial_conditions[:, 0]
    )
    test_kappa_0 = np.asarray(
        test_initial_conditions[:, 1] / test_initial_conditions[:, 0]
    )
    results = {
        "variant": "fixed_homothetic_recursive_kernel",
        "collocation_source": data_set.collocation_source,
        "collocation_design": "frozen_exact_benchmark_states",
        "parameter_derivative": "conditional_on_fixed_collocation_design",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": float(np.asarray(train_initial_conditions[0, 0])),
        "k_0": float(np.asarray(train_initial_conditions[0, 1])),
        "data_seed": ic_set.data_seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "policy": "gross_investment_share",
        "state_coordinate": "log(k / z)",
        "kernel": "rbf",
        "amplitude": kernel_set.amplitude,
        "lengthscale": kernel_set.lengthscale,
        "metric_epsilon": kernel_set.metric_epsilon,
        "metric": "shifted_rkhs_gram",
        "selection": "ridge_rkhs_coefficients_with_free_constant",
        "ridge": lm_set.ridge,
        "i_const": float(i_const),
        "rkhs_norm": float(rkhs_norm_squared(alpha, args.gram)),
        "shifted_metric_norm": float(rkhs_norm_squared(alpha, shifted_gram)),
        "penalized_parameter_count": n_train,
        "unpenalized_parameter_count": 1,
        "gram_min_eigenvalue": float(gram_eigenvalues[0]),
        "gram_max_eigenvalue": float(gram_eigenvalues[-1]),
        "shifted_gram_condition": float(jnp.linalg.cond(shifted_gram)),
        "train_loss": float(result.info.resid_loss),
        "max_train_euler": max_train_euler,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_values["euler_residual"]))),
        "max_log_tvc_residual": max_log_tvc,
        "tvc_normalization": "relative_to_initial_log_certificate",
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_values["c"])),
        "test_path_ok": test_path_ok,
        "test_mean_abs_k_rel_error": mean_k_error,
        "test_mean_abs_c_rel_error": mean_c_error,
        "test_median_abs_k_rel_error": float(np.median(np.abs(k_error))),
        "test_median_abs_c_rel_error": float(np.median(np.abs(c_error))),
        "test_max_abs_k_rel_error": float(np.max(np.abs(k_error))),
        "test_max_abs_c_rel_error": float(np.max(np.abs(c_error))),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(status, "unknown"),
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "solver_converged": status == LMStatus.CONVERGED,
        "solution_accepted": not rejection_reasons,
        "rejection_reason": (
            "accepted" if not rejection_reasons else ",".join(rejection_reasons)
        ),
        "kappa_ss": kappa_ss,
        "num_train_trajectories": int(train_initial_conditions.shape[0]),
        "num_test_trajectories": data_set.num_test_trajectories,
        "num_collocation_states": n_train,
        "train_kappa_0_min": float(train_kappa_0.min()),
        "train_kappa_0_max": float(train_kappa_0.max()),
        "test_kappa_0_min": float(test_kappa_0.min()),
        "test_kappa_0_max": float(test_kappa_0.max()),
        "test_log_variance": ic_set.test_log_variance,
        "train_point_dt": data_set.train_T / (data_set.train_points - 1),
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }
    results.update(
        test_pooled_median_abs_k_rel_error=float(np.median(np.abs(k_error))),
        test_pooled_median_abs_c_rel_error=float(np.median(np.abs(c_error))),
        test_median_trajectory_mean_abs_k_rel_error=float(
            test_summary_df["mean_k_abs_rel_error"].median()
        ),
        test_median_trajectory_mean_abs_c_rel_error=float(
            test_summary_df["mean_c_abs_rel_error"].median()
        ),
    )

    grid_lo = min(float(train_kappa_0.min()), float(test_kappa_0.min()), kappa_ss)
    grid_hi = max(float(train_kappa_0.max()), float(test_kappa_0.max()), kappa_ss)
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
    c_benchmark_grid = benchmark_policy_grid(
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        kappa_grid,
        data_set,
    )
    policy_df = pd.DataFrame(
        {
            "kappa": kappa_grid,
            "investment_share_kernel": np.asarray(grid_values["i"][0])
            / grid_output_per_z,
            "i_per_z_kernel": np.asarray(grid_values["i"][0]),
            "c_tilde_kernel": np.asarray(grid_values["c"][0]),
            "dkappa_kernel": np.asarray(grid_values["dkappa"][0]),
            "c_tilde_benchmark": c_benchmark_grid,
            "dkappa_benchmark": (
                kappa_grid**a - c_benchmark_grid - (delta + g) * kappa_grid
            ),
        }
    )
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
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
    return results, {
        "model": policy,
        "theta": theta,
        "residual_args": args,
        "parameters": p,
        "train_initial_conditions": train_initial_conditions,
        "test_initial_conditions": test_initial_conditions,
        "train_df": train_df,
        "test_df": test_df,
        "test_summary_df": test_summary_df,
        "policy_df": policy_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_recursive)
