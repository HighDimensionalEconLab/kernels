from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from flax import struct
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt

from models.kernels import (
    rbf_state_features,
    rbf_state_gram,
    rkhs_norm_squared,
    state_features,
)
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    InitialConditionSettings,
    StructuralParameters,
    euler_residuals,
    growth_initial_conditions,
    log_tvc_residual,
    simulate,
    trajectory_values,
)
from models.utilities import (
    STATUS_TO_REASON,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
    long_df,
    unit_softplus,
)


@dataclass(frozen=True)
class DataSettings:
    train_T: float = CONSTANTS.train_T
    # Fixed RKHS centers on the exact path: a representation size, not a
    # simulation grid (nothing is integrated during training).
    train_points: int = 65
    # Held-out paths are fixed-step RK4 on the integrator's own states.
    test_points: int = 41
    tvc_steps: int = 256
    train_tvc_T: float = CONSTANTS.transversality_T
    test_T: float = CONSTANTS.test_T


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
class ToleranceSettings:
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
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)


def state_lengthscales(kernel_set):
    ell_z = kernel_set.log_z_lengthscale
    ell_k = kernel_set.log_k_lengthscale
    return (
        kernel_set.lengthscale if ell_z is None else ell_z,
        kernel_set.lengthscale if ell_k is None else ell_k,
    )


def build_residual_args(train_log_states, kernel_set, p, z_0=1.0, k_0=1.0):
    flat_states = train_log_states.reshape(-1, 2)
    dtype = flat_states.dtype
    state_offset = jnp.log(jnp.asarray([z_0, k_0], dtype))
    z_detrend = jnp.asarray(1.0 / (1.0 - p.a), dtype)
    if kernel_set.homothetic_representation:
        centers = flat_states[:, 1] - z_detrend * flat_states[:, 0]
        gram = rbf_state_gram(
            centers,
            jnp.asarray(kernel_set.amplitude, dtype),
            jnp.asarray(kernel_set.lengthscale, dtype),
        )
    else:
        centers = flat_states - state_offset
        gram = state_features(
            kernel_set.state_kernel,
            centers,
            centers,
            kernel_set.amplitude,
            state_lengthscales(kernel_set),
        )
    gram = (gram + gram.T) / 2.0
    return ResidualArgs(
        centers=centers,
        train_log_states=flat_states,
        gram=gram,
        state_offset=state_offset,
        k_scale=jnp.asarray(k_0, dtype),
        z_detrend=z_detrend,
        kernel_set=kernel_set,
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
        features = state_features(
            kernel_set.state_kernel,
            x,
            args.centers,
            kernel_set.amplitude,
            state_lengthscales(kernel_set),
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
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)

    def __call__(self, log_state, p):
        return investment(self.theta, self.args, log_state, p, self.kernel_set)


# Frozen collocation design: the exact benchmark paths of the training
# initial conditions on the training date grid.
def fixed_collocation_data(p, initial_conditions, data_set, *, dtype):
    train_times = np.linspace(0.0, data_set.train_T, data_set.train_points)
    k = np.stack(
        [
            neoclassical_growth_benchmark(p.replace(z_0=z_0, k_0=k_0))(train_times)[0]
            for z_0, k_0 in np.asarray(initial_conditions, dtype=float)
        ]
    )
    z = np.asarray(initial_conditions)[:, :1] * np.exp(
        (1.0 - float(p.a)) * float(p.g) * train_times[None, :]
    )
    log_states = np.stack([np.log(z), np.log(k)], axis=-1)
    return (
        jnp.asarray(np.broadcast_to(train_times, k.shape), dtype=dtype),
        jnp.asarray(log_states, dtype=dtype),
    )


def residual_fn(theta, args, p):
    policy = KernelInvestmentPolicy(theta, args, args.kernel_set)
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
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-6,
        damping_decrease=0.25,
        damping_increase=4.0,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
        max_steps=100,
        atol=1e-5,
        gtol=1e-7,
    ),
    ridge_set: RidgeSettings = RidgeSettings(ridge=1e-12),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    n_test_steps = data_set.test_points - 1
    test_T = data_set.test_T

    ic_draw = growth_initial_conditions(z_0, k_0, ic_set, dtype)
    train_initial_conditions = ic_draw.train_initial_conditions
    test_initial_conditions = ic_draw.test_initial_conditions
    p = StructuralParameters.create(
        dtype=dtype,
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
    p_bench = StructuralParameters.create(
        dtype=jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=1.0,
        k_0=1.0,
        g=g,
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        k_thr=float("inf"),
    )

    fit_timer = Stopwatch()
    train_times, train_log_states = fixed_collocation_data(
        p_bench, train_initial_conditions, data_set, dtype=dtype
    )
    args = build_residual_args(train_log_states, kernel_set, p, z_0, k_0)
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
        residual_fn,
        metric=metric,
        ridge=ridge_set.ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )

    solve_timer = Stopwatch()
    setup_time = fit_timer.elapsed()
    result = solver.solve(
        theta_0,
        args,
        p=p,
        max_steps=lm_set.max_steps,
        atol=lm_set.atol,
        gtol=lm_set.gtol,
        max_steps_is_success=False,
    )
    solve_time = solve_timer.elapsed(result)
    total_fit_time = fit_timer.elapsed()
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
    test_initial = np.asarray(test_initial_conditions)
    test_df = long_df(
        {"t": test_solution.ts, **test_values},
        ("trajectory", "step"),
        scalars={"z_0": test_initial[:, 0], "k_0": test_initial[:, 1]},
    )
    # long_df orders rows (trajectory, step) on the shared grid ts[0], so the
    # per-trajectory benchmark paths concatenate into aligned columns.
    test_times = np.asarray(test_solution.ts[0])
    benchmark_paths = [
        neoclassical_growth_benchmark(p_bench.replace(z_0=z_0, k_0=k_0))(test_times)
        for z_0, k_0 in test_initial
    ]
    test_df["k_benchmark"] = np.concatenate([k for k, _ in benchmark_paths])
    test_df["c_benchmark"] = np.concatenate([c for _, c in benchmark_paths])
    test_df["k_rel_error"] = (test_df["k"] - test_df["k_benchmark"]) / test_df[
        "k_benchmark"
    ].abs()
    test_df["c_rel_error"] = (test_df["c"] - test_df["c_benchmark"]) / test_df[
        "c_benchmark"
    ].abs()
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
    transversality_ok = bool(max_log_tvc_residual < tolerance_set.log_tvc_threshold)

    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    rejection_reasons = []
    if not solver_converged:
        rejection_reasons.append("solver")
    results = {
        "parameters": p.to_dict(),
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
        "tolerance_set": asdict(tolerance_set),
        "test_z_0": ic_draw.test_z_center,
        "test_k_0": ic_draw.test_k_center,
        "kernel": (
            "rbf" if kernel_set.homothetic_representation else kernel_set.state_kernel
        ),
        "h_const": h_const,
        "rkhs_norm": float(rkhs_norm_squared(alpha, args.gram)),
        "shifted_metric_norm": float(rkhs_norm_squared(alpha, shifted_gram)),
        "penalized_parameter_count": n_centers,
        "unpenalized_parameter_count": (1 if kernel_set.free_constant else 0)
        + (2 if kernel_set.linear_mean else 0),
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
        **error_fields,
        "steps": steps,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        **acceptance_fields(rejection_reasons),
        "num_centers": n_centers,
        "test_log_z_variance": ic_draw.test_log_z_variance,
        "test_log_k_variance": ic_draw.test_log_k_variance,
    }

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")
    return results, {
        "policy": policy,
        "theta": theta,
        "residual_args": args,
        "parameters": p,
        "train_initial_conditions": train_initial_conditions,
        "test_initial_conditions": test_initial_conditions,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_kernel_recursive)
