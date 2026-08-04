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

from kernels import (
    additive_rbf_state_features,
    additive_rbf_state_gram,
    ard_rbf_state_features,
    ard_rbf_state_gram,
    rbf_state_features,
    rbf_state_gram,
    rkhs_norm_squared,
)
from neoclassical_growth_nn_recursive import (
    STATUS_TO_REASON,
    InitialConditionSettings,
    StructuralParameters,
    add_benchmark_path_diagnostics,
    benchmark_policy_grid,
    benchmark_trajectories,
    draw_initial_conditions,
    euler_residuals,
    log_tvc_residual,
    positive_policy_grid,
    production,
    rollout_steps,
    scale_invariance_columns,
    trajectory_frame,
    trajectory_values,
)
from utilities import OptimizerSettings, unit_softplus


@dataclass(frozen=True)
class DataSettings:
    collocation_source: Literal["exact_path"] = "exact_path"
    train_T: float = 100.0
    # Fixed RKHS centers on the exact path: a representation size, not a
    # rollout grid (nothing is integrated during training).
    train_points: int = 65
    # Held-out paths are fixed-step RK4 on the integrator's own states.
    test_points: int = 41
    tvc_steps: int = 256
    train_tvc_T: float = 200.0
    test_T: float = 40.0
    policy_grid_size: int = 100
    benchmark_T: float = 200.0
    benchmark_points: int = 801


@jax.tree_util.register_static
@dataclass(frozen=True)
class StateKernelSettings:
    amplitude: float = 1.0
    lengthscale: float = 0.4
    metric_epsilon: float = 3e-5
    # With the free constant, theta is (alpha, h_const) and only alpha is
    # penalized; without it the policy is the pure RKHS expansion.
    free_constant: bool = True
    # Homothetic: 1-D RBF on log(k/z). Non-homothetic: 2-D RBF on the centered
    # (log z, log k); lengthscales default per axis to `lengthscale`, and
    # linear_mean appends two unpenalized linear coefficients.
    homothetic_representation: bool = True
    state_kernel: Literal["ard_rbf", "additive_rbf"] = "ard_rbf"
    log_z_lengthscale: float | None = None
    log_k_lengthscale: float | None = None
    linear_mean: bool = False


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 100
    # RMS Euler at 1e-5 is far below economic significance.
    residual_atol: float = 1e-5
    gtol: float = 1e-7
    log_tvc_threshold: float = 0.0


@struct.dataclass
class ResidualArgs:
    centers: jax.Array
    train_log_states: jax.Array
    gram: jax.Array
    state_offset: jax.Array
    k_scale: jax.Array


def state_lengthscales(kernel_set):
    ell_z = kernel_set.log_z_lengthscale
    ell_k = kernel_set.log_k_lengthscale
    return (
        kernel_set.lengthscale if ell_z is None else ell_z,
        kernel_set.lengthscale if ell_k is None else ell_k,
    )


def validate_settings(data_set, kernel_set, ridge):
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
    if kernel_set.state_kernel not in ("ard_rbf", "additive_rbf"):
        raise ValueError(f"unknown state_kernel: {kernel_set.state_kernel!r}")
    if not all(ell > 0.0 for ell in state_lengthscales(kernel_set)):
        raise ValueError("per-dimension lengthscales must be positive when set")
    if kernel_set.linear_mean and kernel_set.homothetic_representation:
        raise ValueError("linear_mean requires homothetic_representation=False")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")


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


def build_residual_args(train_log_states, kernel_set, z_0=1.0, k_0=1.0):
    flat_states = train_log_states.reshape(-1, 2)
    dtype = flat_states.dtype
    state_offset = jnp.log(jnp.asarray([z_0, k_0], dtype))
    if kernel_set.homothetic_representation:
        centers = flat_states[:, 1] - flat_states[:, 0]
        gram = rbf_state_gram(
            centers,
            jnp.asarray(kernel_set.amplitude, dtype),
            jnp.asarray(kernel_set.lengthscale, dtype),
        )
    else:
        centers = flat_states - state_offset
        lengthscales = jnp.asarray(state_lengthscales(kernel_set), dtype)
        if kernel_set.state_kernel == "ard_rbf":
            gram = ard_rbf_state_gram(
                centers, jnp.asarray(kernel_set.amplitude, dtype), lengthscales
            )
        else:
            gram = additive_rbf_state_gram(
                centers, jnp.full(2, kernel_set.amplitude, dtype), lengthscales
            )
    gram = (gram + gram.T) / 2.0
    return ResidualArgs(
        centers=centers,
        train_log_states=flat_states,
        gram=gram,
        state_offset=state_offset,
        k_scale=jnp.asarray(k_0, dtype),
    )


def raw_policy(theta, args, log_state, kernel_set):
    alpha = theta[0]
    if kernel_set.homothetic_representation:
        q = log_state[1] - log_state[0]
        features = rbf_state_features(
            q,
            args.centers,
            jnp.asarray(kernel_set.amplitude, q.dtype),
            jnp.asarray(kernel_set.lengthscale, q.dtype),
        )
        h = features @ alpha
    else:
        x = log_state - args.state_offset
        lengthscales = jnp.asarray(state_lengthscales(kernel_set), x.dtype)
        if kernel_set.state_kernel == "ard_rbf":
            features = ard_rbf_state_features(
                x,
                args.centers,
                jnp.asarray(kernel_set.amplitude, x.dtype),
                lengthscales,
            )
        else:
            features = additive_rbf_state_features(
                x,
                args.centers,
                jnp.full(2, kernel_set.amplitude, x.dtype),
                lengthscales,
            )
        h = features @ alpha
        # The linear-mean coefficients are the last theta leaf, after the free
        # constant when both are present; like h_const they are unpenalized.
        if kernel_set.linear_mean:
            h = h + x @ theta[-1]
    if kernel_set.free_constant and len(theta) > 1:
        h = h + theta[1]
    return h


# Same output maps as the recursive NN: unit softplus of the raw policy
# anchored at delta * k (homothetic) or the constant delta * k_0.
def investment(theta, args, log_state, p, kernel_set):
    raw = raw_policy(theta, args, log_state, kernel_set)
    if kernel_set.homothetic_representation:
        k = jnp.exp(log_state[1])
        return p.delta * k * unit_softplus(raw)
    return p.delta * args.k_scale * unit_softplus(raw)


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
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    ridge: float = 1e-12,
    use_float64: bool = True,
    matmul_highest: bool = True,
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: StateKernelSettings = StateKernelSettings(),
    opt_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-6,
        damping_decrease=0.25,
        damping_increase=4.0,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", use_float64)
    if matmul_highest:
        jax.config.update("jax_default_matmul_precision", "highest")
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
    validate_settings(data_set, kernel_set, ridge)
    n_test_steps = data_set.test_points - 1
    test_T = data_set.test_T

    dtype = jnp.float64 if use_float64 else jnp.float32
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
        g=g,
        z_0=train_initial_conditions[:, 0],
        k_0=train_initial_conditions[:, 1],
    )

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
    args = build_residual_args(train_log_states, kernel_set, z_0, k_0)
    n_train = args.centers.shape[0]
    shifted_gram = args.gram + kernel_set.metric_epsilon * jnp.eye(n_train, dtype=dtype)
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(shifted_gram, upper=True),
        repeats=1,
    )
    theta_0 = (jnp.zeros(n_train, dtype=dtype),)
    if kernel_set.free_constant:
        theta_0 = theta_0 + (jnp.zeros((), dtype=dtype),)
    if kernel_set.linear_mean:
        theta_0 = theta_0 + (jnp.zeros(2, dtype=dtype),)
    solver = RidgeLevenbergMarquardt(
        partial(residual_fn, kernel_set=kernel_set),
        metric=metric,
        ridge=ridge,
        init_damping=opt_set.init_damping,
        damping_decrease=opt_set.damping_decrease,
        damping_increase=opt_set.damping_increase,
        geodesic_acceleration=opt_set.geodesic_acceleration,
        geodesic_acceptance_ratio=opt_set.geodesic_acceptance_ratio,
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
    steps = int(result.steps)
    train_loss = float(result.info.resid_loss)
    status = int(result.status)
    stopping_reason = STATUS_TO_REASON.get(status, "unknown")
    solver_converged = status == LMStatus.CONVERGED
    if verbose:
        print(
            f"solve: {stopping_reason} after {steps} steps, "
            f"train_loss={train_loss:.3e} ({solve_time:.2f}s)"
        )

    theta = result.x
    alpha = theta[0]
    h_const = float(theta[1]) if kernel_set.free_constant else 0.0
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
        and jnp.all(jnp.isfinite(test_values["c"]))
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

    max_log_tvc, train_terminal_log_tvc, tvc_ok = log_tvc_residual(
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
    _, test_terminal_log_tvc, test_tvc_ok = log_tvc_residual(
        policy,
        jnp.log(test_initial_conditions),
        data_set.train_T,
        data_set.train_tvc_T,
        p,
        data_set,
    )
    test_max_log_tvc = float(jnp.max(test_terminal_log_tvc))
    test_transversality_ok = bool(
        jnp.all(test_tvc_ok) and test_max_log_tvc < stop_set.log_tvc_threshold
    )

    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)
    mean_k_error = float(np.mean(np.abs(k_error)))
    mean_c_error = float(np.mean(np.abs(c_error)))

    gram_eigenvalues = jnp.linalg.eigvalsh(args.gram)
    train_kappa_0 = np.asarray(
        train_initial_conditions[:, 1] / train_initial_conditions[:, 0]
    )
    test_kappa_0 = np.asarray(
        test_initial_conditions[:, 1] / test_initial_conditions[:, 0]
    )
    train_kappa = train_df["kappa"].to_numpy()
    test_kappa = test_df["kappa"].to_numpy()
    collocation_kappa_spread = float(np.max(train_kappa) / np.min(train_kappa))
    results = {
        "variant": "fixed_recursive_kernel",
        "collocation_source": data_set.collocation_source,
        "collocation_design": "frozen_exact_benchmark_states",
        "parameter_derivative": "conditional_on_fixed_collocation_design",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "test_z_0": test_z_center,
        "test_k_0": test_k_center,
        "data_seed": ic_set.data_seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "homothetic_representation": kernel_set.homothetic_representation,
        "policy": (
            "delta_k_unit_softplus"
            if kernel_set.homothetic_representation
            else "delta_k0_unit_softplus"
        ),
        "state_coordinate": (
            "log(k / z)" if kernel_set.homothetic_representation else "(log z, log k)"
        ),
        "kernel": (
            "rbf" if kernel_set.homothetic_representation else kernel_set.state_kernel
        ),
        "amplitude": kernel_set.amplitude,
        "lengthscale": kernel_set.lengthscale,
        "metric_epsilon": kernel_set.metric_epsilon,
        "metric": "shifted_rkhs_gram",
        "selection": (
            "ridge_rkhs_coefficients_with_free_constant"
            if kernel_set.free_constant
            else "ridge_rkhs_coefficients"
        ),
        "ridge": ridge,
        "free_constant": kernel_set.free_constant,
        "linear_mean": kernel_set.linear_mean,
        "h_const": h_const,
        "rkhs_norm": float(rkhs_norm_squared(alpha, args.gram)),
        "shifted_metric_norm": float(rkhs_norm_squared(alpha, shifted_gram)),
        "penalized_parameter_count": n_train,
        "unpenalized_parameter_count": (1 if kernel_set.free_constant else 0)
        + (2 if kernel_set.linear_mean else 0),
        "gram_min_eigenvalue": float(gram_eigenvalues[0]),
        "gram_max_eigenvalue": float(gram_eigenvalues[-1]),
        "shifted_gram_condition": float(jnp.linalg.cond(shifted_gram)),
        "train_loss": train_loss,
        "max_train_euler": max_train_euler,
        "max_test_euler": max_test_euler,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_values["euler_residual"]))),
        "test_trajectory_euler_mean_abs_p50": float(
            jnp.quantile(test_trajectory_euler, 0.5)
        ),
        "test_trajectory_euler_mean_abs_p90": float(
            jnp.quantile(test_trajectory_euler, 0.9)
        ),
        "test_trajectory_euler_mean_abs_max": float(jnp.max(test_trajectory_euler)),
        "max_log_tvc_residual": max_log_tvc,
        "mean_log_tvc_residual": float(jnp.mean(train_terminal_log_tvc)),
        "test_mean_log_tvc_residual": float(jnp.mean(test_terminal_log_tvc)),
        "test_max_log_tvc_residual": test_max_log_tvc,
        "test_transversality_ok": test_transversality_ok,
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
        "steps": steps,
        "stopping_reason": stopping_reason,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "solver_converged": solver_converged,
        "solution_accepted": solver_converged,
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        "num_collocation_states": n_train,
        "collocation_kappa_spread": collocation_kappa_spread,
        "collocation_log_kappa_min": float(np.log(np.min(train_kappa))),
        "collocation_log_kappa_max": float(np.log(np.max(train_kappa))),
        "test_log_kappa_min": float(np.log(np.nanmin(test_kappa))),
        "test_log_kappa_max": float(np.log(np.nanmax(test_kappa))),
        "train_kappa_0_min": float(train_kappa_0.min()),
        "train_kappa_0_max": float(train_kappa_0.max()),
        "test_kappa_0_min": float(test_kappa_0.min()),
        "test_kappa_0_max": float(test_kappa_0.max()),
        "train_log_z_variance": ic_set.train_log_z_variance,
        "train_log_k_variance": ic_set.train_log_k_variance,
        "test_log_z_variance": test_log_z_variance,
        "test_log_k_variance": test_log_k_variance,
        "train_point_dt": data_set.train_T / (data_set.train_points - 1),
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }
    if not kernel_set.homothetic_representation:
        ell_z, ell_k = state_lengthscales(kernel_set)
        results.update(
            state_kernel=kernel_set.state_kernel,
            log_z_lengthscale=ell_z,
            log_k_lengthscale=ell_k,
        )
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
            # The learned object itself: unit_softplus of the raw policy,
            # since the grid fixes z = 1 so k = kappa.
            "i_over_delta_k_kernel": np.asarray(grid_values["i"][0])
            / (delta * np.asarray(kappa_grid)),
            "i_per_z_kernel": np.asarray(grid_values["i"][0]),
            "c_tilde_kernel": np.asarray(grid_values["c"][0]),
            "dkappa_kernel": np.asarray(grid_values["dkappa"][0]),
            **scale_invariance_columns(policy, kappa_grid, p, dtype, "kernel"),
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
        "ridge": ridge,
        "opt_set": asdict(opt_set),
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
