from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt

from models.kernels import (
    STATE_KERNEL_NAMES,
    rkhs_norm_squared,
)
from models.neoclassical_growth import (
    RecursiveParameters,
    euler_residuals,
    log_tvc_residual,
    simulate,
    trajectory_values,
)
from models.neoclassical_growth_benchmark import benchmark_df
from models.policies import (
    KernelInvestmentPolicy,
    StateKernelSettings,
    build_residual_args,
    state_lengthscales,
)
from models.utilities import (
    DOMAIN_EPS,
    STATUS_TO_REASON,
    InitialConditionSettings,
    OptimizerSettings,
    growth_initial_conditions,
    long_df,
)
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
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


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 100
    # float64-tuned; float32 callers pass looser values explicitly.
    residual_atol: float = 1e-5
    gtol: float = 1e-7
    ridge: float = 1e-12
    log_tvc_threshold: float = 0.0


# Frozen collocation design: the exact benchmark paths of the training
# initial conditions on the training date grid.
def fixed_collocation_data(
    a, delta, rho_hat, sigma_crra, g, initial_conditions, data_set, *, dtype
):
    train_times = np.linspace(0.0, data_set.train_T, data_set.train_points)
    benchmark = benchmark_df(
        initial_conditions, train_times, a, delta, rho_hat, sigma_crra, g
    )
    k = benchmark["k"].to_numpy().reshape(len(initial_conditions), -1)
    z = np.asarray(initial_conditions)[:, :1] * np.exp(
        (1.0 - a) * g * train_times[None, :]
    )
    log_states = np.stack([np.log(z), np.log(k)], axis=-1)
    return (
        jnp.asarray(np.broadcast_to(train_times, k.shape), dtype=dtype),
        jnp.asarray(log_states, dtype=dtype),
    )


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
    # Model and collocation design setup.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    if kernel_set.state_kernel not in STATE_KERNEL_NAMES:
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
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        k_thr=float("inf"),
    )

    fit_start = time.perf_counter()
    train_times, train_log_states = fixed_collocation_data(
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
    n_centers = args.centers.shape[0]
    shifted_gram = args.gram + kernel_set.metric_epsilon * jnp.eye(
        n_centers, dtype=dtype
    )
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(shifted_gram, upper=True),
        repeats=1,
    )
    theta_0 = (jnp.zeros(n_centers, dtype=dtype),)
    if kernel_set.free_constant:
        theta_0 = theta_0 + (jnp.zeros((), dtype=dtype),)
    if kernel_set.linear_mean:
        theta_0 = theta_0 + (jnp.zeros(2, dtype=dtype),)

    # Ridge Levenberg-Marquardt solve.
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

    # Diagnostics on the frozen centers and held-out rollouts.
    train_values = trajectory_values(policy, train_log_states, p)
    train_initial = np.asarray(train_initial_conditions)
    train_df = long_df(
        {"t": train_times, **train_values},
        ("trajectory", "step"),
        scalars={"z_0": train_initial[:, 0], "k_0": train_initial[:, 1]},
    )

    test_solution = simulate(
        policy, jnp.log(test_initial_conditions), test_T, n_test_steps, p
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
    test_initial = np.asarray(test_initial_conditions)
    test_df = long_df(
        {"t": test_solution.ts, **test_values},
        ("trajectory", "step"),
        scalars={"z_0": test_initial[:, 0], "k_0": test_initial[:, 1]},
    )
    benchmark = benchmark_df(
        test_initial,
        np.asarray(test_solution.ts[0]),
        a,
        delta,
        rho_hat,
        sigma_crra,
        g,
    )
    test_df = test_df.merge(
        benchmark[["trajectory", "t", "k", "c"]],
        on=["trajectory", "t"],
        suffixes=("", "_benchmark"),
    )
    test_df["k_rel_error"] = (test_df["k"] - test_df["k_benchmark"]) / test_df[
        "k_benchmark"
    ].abs().clip(lower=DOMAIN_EPS)
    test_df["c_rel_error"] = (test_df["c"] - test_df["c_benchmark"]) / test_df[
        "c_benchmark"
    ].abs().clip(lower=DOMAIN_EPS)
    abs_errors = test_df[["k_rel_error", "c_rel_error"]].abs()
    trajectory_mean = abs_errors.groupby(test_df["trajectory"]).mean()
    error_fields = {
        "test_mean_abs_k_rel_error": float(abs_errors["k_rel_error"].mean()),
        "test_mean_abs_c_rel_error": float(abs_errors["c_rel_error"].mean()),
        "test_max_abs_k_rel_error": float(abs_errors["k_rel_error"].max()),
        "test_median_trajectory_mean_abs_k_rel_error": float(
            trajectory_mean["k_rel_error"].median()
        ),
        "test_median_trajectory_mean_abs_c_rel_error": float(
            trajectory_mean["c_rel_error"].median()
        ),
    }

    # Terminal transversality certificate on the training initial conditions.
    train_tvc = log_tvc_residual(
        policy,
        jnp.log(train_initial_conditions),
        data_set.train_tvc_T,
        data_set.tvc_steps,
        p,
    )
    max_log_tvc_residual = float(jnp.max(train_tvc))
    transversality_ok = bool(max_log_tvc_residual < stop_set.log_tvc_threshold)

    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)
    gram_eigenvalues = jnp.linalg.eigvalsh(args.gram)

    # Published results.
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
        "penalized_parameter_count": n_centers,
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
        "max_log_tvc_residual": max_log_tvc_residual,
        "transversality_ok": transversality_ok,
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
        "num_centers": n_centers,
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
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_recursive)
