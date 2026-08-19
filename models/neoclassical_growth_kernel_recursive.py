from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from flax import struct
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt

from models.kernels import (
    MATERN_STATE_FAMILIES,
    additive_matern_state_features,
    additive_rbf_state_features,
    ard_matern_state_features,
    ard_rbf_state_features,
    rbf_state_features,
    rbf_state_gram,
    rkhs_norm_squared,
)
from models.neoclassical_growth import (
    InitialConditionSettings,
    RecursiveParameters,
    add_benchmark_path_diagnostics,
    benchmark_trajectories,
    euler_residuals,
    growth_initial_conditions,
    initial_kappa,
    policy_diagnostics_frame,
    rollout_steps,
    trajectory_frame,
    trajectory_values,
    tvc_certificates,
    validate_balanced_growth,
)
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.utilities import (
    STATUS_TO_REASON,
    OptimizerSettings,
    unit_softplus,
)


@dataclass(frozen=True)
class DataSettings:
    train_T: float = CONSTANTS.train_T
    # Fixed RKHS centers on the exact path: a representation size, not a
    # rollout grid (nothing is integrated during training).
    train_points: int = 65
    # Held-out paths are fixed-step RK4 on the integrator's own states.
    test_points: int = 41
    tvc_steps: int = 256
    train_tvc_T: float = CONSTANTS.transversality_T
    test_T: float = CONSTANTS.test_T
    policy_grid_size: int = 100


@jax.tree_util.register_static
@dataclass(frozen=True)
class StateKernelSettings:
    amplitude: float = 1.0
    lengthscale: float = 0.6
    metric_epsilon: float = 3e-5
    # With the free constant, theta is (alpha, h_const) and only alpha is
    # penalized; without it the policy is the pure RKHS expansion.
    free_constant: bool = True
    # Homothetic: 1-D RBF on log(k/z). Non-homothetic: a 2-D expansion on the
    # centered (log z, log k) with product (ard) or sum (additive) composition
    # of RBF or Matern 1-D kernels; lengthscales default per axis to
    # `lengthscale`, and linear_mean appends two unpenalized coefficients.
    homothetic_representation: bool = True
    state_kernel: Literal[
        "ard_rbf",
        "additive_rbf",
        "ard_matern32",
        "ard_matern52",
        "additive_matern32",
        "additive_matern52",
    ] = "ard_rbf"
    log_z_lengthscale: float | None = None
    log_k_lengthscale: float | None = None
    linear_mean: bool = False


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 100
    # float64-tuned; float32 callers pass looser values explicitly.
    residual_atol: float = 1e-5
    gtol: float = 1e-7
    ridge: float = 1e-12
    log_tvc_threshold: float = 0.0


@struct.dataclass
class ResidualArgs:
    centers: jax.Array
    train_log_states: jax.Array
    gram: jax.Array
    state_offset: jax.Array
    k_scale: jax.Array
    # 1/(1-a): log kappa = log k - z_detrend * log z for Hicks-neutral z.
    z_detrend: jax.Array


def state_lengthscales(kernel_set):
    ell_z = kernel_set.log_z_lengthscale
    ell_k = kernel_set.log_k_lengthscale
    return (
        kernel_set.lengthscale if ell_z is None else ell_z,
        kernel_set.lengthscale if ell_k is None else ell_k,
    )


def nonhomothetic_state_features(x, centers, kernel_set):
    dtype = x.dtype
    lengthscales = jnp.asarray(state_lengthscales(kernel_set), dtype)
    structure, _, family = kernel_set.state_kernel.partition("_")
    if structure == "ard":
        sigma = jnp.asarray(kernel_set.amplitude, dtype)
        if family == "rbf":
            return ard_rbf_state_features(x, centers, sigma, lengthscales)
        return ard_matern_state_features(family, x, centers, sigma, lengthscales)
    sigmas = jnp.full(2, kernel_set.amplitude, dtype)
    if family == "rbf":
        return additive_rbf_state_features(x, centers, sigmas, lengthscales)
    return additive_matern_state_features(family, x, centers, sigmas, lengthscales)


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
    train_times = np.linspace(0.0, data_set.train_T, data_set.train_points)
    z, k, c = benchmark_trajectories(
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        initial_conditions,
        train_times,
    )
    log_states = np.stack([np.log(z), np.log(k)], axis=-1)
    return (
        jnp.asarray(np.broadcast_to(train_times, k.shape), dtype=dtype),
        jnp.asarray(log_states, dtype=dtype),
        jnp.asarray(c, dtype=dtype),
    )


def build_residual_args(train_log_states, kernel_set, a, z_0=1.0, k_0=1.0):
    flat_states = train_log_states.reshape(-1, 2)
    dtype = flat_states.dtype
    state_offset = jnp.log(jnp.asarray([z_0, k_0], dtype))
    z_detrend = jnp.asarray(1.0 / (1.0 - a), dtype)
    if kernel_set.homothetic_representation:
        centers = flat_states[:, 1] - z_detrend * flat_states[:, 0]
        gram = rbf_state_gram(
            centers,
            jnp.asarray(kernel_set.amplitude, dtype),
            jnp.asarray(kernel_set.lengthscale, dtype),
        )
    else:
        centers = flat_states - state_offset
        gram = nonhomothetic_state_features(centers, centers, kernel_set)
    gram = (gram + gram.T) / 2.0
    return ResidualArgs(
        centers=centers,
        train_log_states=flat_states,
        gram=gram,
        state_offset=state_offset,
        k_scale=jnp.asarray(k_0, dtype),
        z_detrend=z_detrend,
    )


def raw_policy(theta, args, log_state, kernel_set):
    alpha = theta[0]
    if kernel_set.homothetic_representation:
        q = log_state[1] - args.z_detrend * log_state[0]
        features = rbf_state_features(
            q,
            args.centers,
            jnp.asarray(kernel_set.amplitude, q.dtype),
            jnp.asarray(kernel_set.lengthscale, q.dtype),
        )
        h = features @ alpha
    else:
        x = log_state - args.state_offset
        features = nonhomothetic_state_features(x, args.centers, kernel_set)
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
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)

    def __call__(self, log_state, p):
        return investment(self.theta, self.args, log_state, p, self.kernel_set)


def residual_fn(theta, args, p, kernel_set):
    policy = KernelInvestmentPolicy(theta, args, kernel_set)
    residual = euler_residuals(policy, args.train_log_states, p)
    return residual / jnp.sqrt(residual.size)


def neoclassical_growth_kernel_recursive(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    use_float64: bool = True,
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
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    validate_balanced_growth(delta, rho_hat, sigma_crra, g)
    state_kernels = tuple(
        f"{structure}_{family}"
        for structure in ("ard", "additive")
        for family in ("rbf", *MATERN_STATE_FAMILIES)
    )
    if kernel_set.state_kernel not in state_kernels:
        raise ValueError(f"unknown state_kernel: {kernel_set.state_kernel!r}")
    if kernel_set.linear_mean and kernel_set.homothetic_representation:
        raise ValueError("linear_mean requires homothetic_representation=False")
    n_test_steps = data_set.test_points - 1
    test_T = data_set.test_T

    ic_draw = growth_initial_conditions(z_0, k_0, ic_set, dtype)
    train_initial_conditions = ic_draw.train_initial_conditions
    test_initial_conditions = ic_draw.test_initial_conditions
    p = RecursiveParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        g=g,
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
    args = build_residual_args(train_log_states, kernel_set, a, z_0, k_0)
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
        ridge=stop_set.ridge,
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
    policy = KernelInvestmentPolicy(theta, args, kernel_set)
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_times,
        train_initial_conditions,
    )
    train_c_benchmark_np = np.asarray(train_c_benchmark).reshape(-1)
    train_df["c_benchmark"] = train_c_benchmark_np
    train_df["i_benchmark"] = (
        np.asarray(train_df["z"] * train_df["k"] ** a) - train_c_benchmark_np
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
    )
    test_summary_df, error_fields = add_benchmark_path_diagnostics(
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

    tvc_fields = tvc_certificates(
        policy,
        jnp.log(train_initial_conditions),
        jnp.log(test_initial_conditions),
        p,
        data_set,
        stop_set.log_tvc_threshold,
    )

    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    gram_eigenvalues = jnp.linalg.eigvalsh(args.gram)
    train_kappa_0 = initial_kappa(train_initial_conditions, a)
    test_kappa_0 = initial_kappa(test_initial_conditions, a)
    train_kappa = train_df["kappa"].to_numpy()
    test_kappa = test_df["kappa"].to_numpy()
    collocation_kappa_spread = float(np.max(train_kappa) / np.min(train_kappa))
    results = {
        "variant": "fixed_recursive_kernel",
        "collocation_design": "frozen_exact_benchmark_states",
        "parameter_derivative": "conditional_on_fixed_collocation_design",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "z_0": z_0,
        "k_0": k_0,
        "test_z_0": ic_draw.test_z_center,
        "test_k_0": ic_draw.test_k_center,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_values["euler_residual"].dtype),
        "homothetic_representation": kernel_set.homothetic_representation,
        "policy": (
            "delta_k_unit_softplus"
            if kernel_set.homothetic_representation
            else "delta_k0_unit_softplus"
        ),
        "state_coordinate": (
            "log(k / z^(1/(1-a)))"
            if kernel_set.homothetic_representation
            else "(log z, log k)"
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
        "ridge": stop_set.ridge,
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
        **tvc_fields,
        "min_test_c": float(jnp.min(test_values["c"])),
        "test_path_ok": test_path_ok,
        **error_fields,
        "steps": steps,
        "stopping_reason": stopping_reason,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
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
        "test_log_z_variance": ic_draw.test_log_z_variance,
        "test_log_k_variance": ic_draw.test_log_k_variance,
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

    policy_df = policy_diagnostics_frame(
        policy,
        train_kappa,
        test_df["kappa"].to_numpy(),
        train_kappa_0,
        test_kappa_0,
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
        p,
        data_set,
        dtype,
        "kernel",
    )
    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
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
