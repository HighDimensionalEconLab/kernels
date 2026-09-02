from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from flax import struct
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt

from models.kernels import (
    ard_rbf_state_features,
    rbf_state_features,
    rbf_state_gram,
)
from models.neural_networks import (
    load_model,
    unit_softplus,
)
from models.stochastic_growth_shared import (
    STOCHASTIC_GROWTH_CONSTANTS as CONSTANTS,
)
from models.stochastic_growth_shared import (
    DataSettings,
    InitialConditionSettings,
    StructuralParameters,
    add_reference_path_diagnostics,
    euler_residuals,
    growth_initial_conditions,
    log_tvc_residual,
    make_investment_policy,
    problem_cache_config,
    simulate,
    simulate_train,
    trajectory_values,
)
from models.utilities import (
    SDE_SOLVERS,
    STATUS_TO_REASON,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
    cache_key,
    long_df,
    sample_trajectory_noise,
)


@dataclass(frozen=True)
class StateKernelSettings:
    # Additive kernel: a 1-D RBF on the ray coordinate
    # q = log k - ray_slope * log z plus a 2-D ARD RBF on (log z, log k). The
    # penalized solve finds the minimum-norm split between the two components,
    # so the amplitude ratio sets how much of the policy the ray explains;
    # ray_amplitude = 0 recovers the pure 2-D representation. ray_slope is a
    # tuned hyperparameter like the lengthscales -- the structural detrending
    # 1/(1-a) must never enter the approximation class.
    ray_slope: float = 2.0
    ray_amplitude: float = 1.0
    ray_lengthscale: float = 0.6
    amplitude: float = 1.0
    log_z_lengthscale: float = 0.4
    log_k_lengthscale: float = 0.45
    metric_epsilon: float = 3e-5


# Ex-post acceptance: gate_train requires the ridge LM to have converged,
# gate_test the held-out Euler errors under the bars.
@dataclass(frozen=True)
class ToleranceSettings:
    gate_train: bool = True
    gate_test: bool = True
    test_euler_mean_abs: float = 2e-4
    test_euler_max: float = 1e-2


@struct.dataclass
class ResidualArgs:
    # train_log_states doubles as the 2-D kernel centers: the representer
    # nodes are the collocation states.
    train_log_states: jax.Array
    q_centers: jax.Array
    gram: jax.Array
    k_scale: jax.Array
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)


def build_residual_args(train_log_states, kernel_set, k_0=1.0):
    flat_states = train_log_states.reshape(-1, 2)
    dtype = flat_states.dtype
    q_centers = flat_states[:, 1] - kernel_set.ray_slope * flat_states[:, 0]
    gram = ard_rbf_state_features(
        flat_states,
        flat_states,
        jnp.asarray(kernel_set.amplitude, dtype),
        jnp.asarray(
            (kernel_set.log_z_lengthscale, kernel_set.log_k_lengthscale), dtype
        ),
    ) + rbf_state_gram(
        q_centers,
        jnp.asarray(kernel_set.ray_amplitude, dtype),
        jnp.asarray(kernel_set.ray_lengthscale, dtype),
    )
    return ResidualArgs(
        train_log_states=flat_states,
        q_centers=q_centers,
        gram=gram,
        k_scale=jnp.asarray(k_0, dtype),
        kernel_set=kernel_set,
    )


# theta = (alpha, h_const, linear_coeffs); only alpha is penalized.
def raw_policy(theta, args, log_state):
    kernel_set = args.kernel_set
    alpha, h_const, linear_coeffs = theta
    q = log_state[1] - kernel_set.ray_slope * log_state[0]
    features = ard_rbf_state_features(
        log_state,
        args.train_log_states,
        jnp.asarray(kernel_set.amplitude, log_state.dtype),
        jnp.asarray(
            (kernel_set.log_z_lengthscale, kernel_set.log_k_lengthscale),
            log_state.dtype,
        ),
    ) + rbf_state_features(
        q,
        args.q_centers,
        jnp.asarray(kernel_set.ray_amplitude, q.dtype),
        jnp.asarray(kernel_set.ray_lengthscale, q.dtype),
    )
    return features @ alpha + log_state @ linear_coeffs + h_const


# Same output map as the recursive NN: unit softplus of the raw policy
# anchored at the constant delta * k_0.
def investment(theta, args, log_state, p):
    raw = raw_policy(theta, args, log_state)
    return p.delta * args.k_scale * unit_softplus(raw)


@struct.dataclass
class KernelInvestmentPolicy:
    theta: tuple
    args: ResidualArgs
    p: StructuralParameters

    def __call__(self, log_state):
        return jnp.atleast_1d(investment(self.theta, self.args, log_state, self.p))


# Locate the reference NN policy for this exact problem (structural
# parameters, dtype, and training initial-condition design) by its problem
# hash; among matching runs, take the accepted artifact with the smallest
# held-out residual. There is deliberately no fallback: solve and cache the
# NN first via stochastic_growth_nn_recursive --policy_cache <dir>.
def load_reference_policy(policy_dir, problem_config):
    problem_key = cache_key(problem_config)
    pattern = f"stochastic_growth_nn_recursive_policy_{problem_key}-*.npz"
    candidates = []
    for path in sorted(pathlib.Path(policy_dir).glob(pattern)):
        meta = json.loads(str(np.load(path)["__meta__"]))
        if meta.get("solution_accepted", False):
            candidates.append((meta.get("test_euler_mean_abs", np.inf), path))
    if not candidates:
        raise FileNotFoundError(
            f"no accepted NN policy for problem {problem_key} under {policy_dir}; "
            "run stochastic_growth_nn_recursive with --policy_cache first"
        )
    _, best = min(candidates, key=lambda pair: pair[0])
    policy, meta = load_model(best, make_investment_policy)
    return policy, meta, best


def residual_fn(theta, args, p):
    policy = KernelInvestmentPolicy(theta, args, p)
    residual = euler_residuals(policy, args.train_log_states, p)
    return residual / jnp.sqrt(residual.size)


# The kernel representation of the same stochastic recursive model as
# stochastic_growth_nn_recursive: identical policy output map, residuals, and
# diagnostics, but the collocation states are a frozen design -- simulated
# once under the saved reference NN policy -- rather than re-integrated by
# differentiating through the solver's own simulation.
def stochastic_growth_kernel_recursive(
    a: float = CONSTANTS["a"],
    delta: float = CONSTANTS["delta"],
    rho_hat: float = CONSTANTS["rho_hat"],
    sigma_crra: float = CONSTANTS["sigma_crra"],
    eta: float = CONSTANTS["eta"],
    sigma_z: float = CONSTANTS["sigma_z"],
    z_bar: float = 1.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    policy_dir: str = ".results",
    # The frozen design needs coverage, not integration accuracy.
    data_set: DataSettings = DataSettings(train_points=33),
    ic_set: InitialConditionSettings = InitialConditionSettings(
        num_train_trajectories=16,
        train_log_z_variance=CONSTANTS["z_ergodic_variance"],
        train_log_k_variance=0.01,
    ),
    kernel_set: StateKernelSettings = StateKernelSettings(),
    # atol sits just above this representation's float64 residual floor.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-6,
        damping_decrease=0.25,
        damping_increase=4.0,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
        max_steps=150,
        atol=7e-5,
        gtol=1e-7,
    ),
    ridge_set: RidgeSettings = RidgeSettings(ridge=1e-12),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    sde = SDE_SOLVERS[sde_solver]
    n_train_steps = data_set.train_points - 1
    n_test_steps = data_set.test_points - 1
    train_dt = data_set.train_T / n_train_steps
    test_T = data_set.test_T

    ic_draw = growth_initial_conditions(z_0, k_0, ic_set, dtype)
    train_ic_key = ic_draw.train_ic_key
    test_ic_key = ic_draw.test_ic_key
    train_initial_conditions = ic_draw.train_initial_conditions
    test_initial_conditions = ic_draw.test_initial_conditions
    p = StructuralParameters.create(
        dtype=dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        eta=eta,
        sigma_z=sigma_z,
        log_z_bar=float(np.log(z_bar)),
        z_0=train_initial_conditions[:, 0],
        k_0=train_initial_conditions[:, 1],
    )

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
    reference_policy, reference_meta, reference_path = load_reference_policy(
        policy_dir, problem_config
    )
    # A scale mismatch means the artifact's problem differs from its hash.
    saved = float(reference_policy.output_scale[...][0])
    recomputed = delta * k_0
    if not np.isclose(saved, recomputed):
        raise RuntimeError(
            "reference policy scale disagrees with the requested problem: "
            f"saved output_scale = {saved}, recomputed {recomputed}"
        )
    if verbose:
        print(
            f"reference NN policy {reference_path} "
            f"(test_euler_mean_abs {reference_meta['test_euler_mean_abs']:.3e})"
        )

    fit_timer = Stopwatch()
    # Frozen collocation design: the reference NN policy's own simulation on the
    # training noise stream; the kernel never differentiates trajectories.
    train_noise = sample_trajectory_noise(
        sde,
        jax.random.fold_in(train_ic_key, 3),
        ic_set.num_train_trajectories,
        n_train_steps,
        train_dt,
        dtype,
        2,
    )
    train_solution = simulate_train(
        sde,
        reference_policy,
        jnp.log(train_initial_conditions),
        train_noise,
        p,
        data_set,
    )
    if not bool(jnp.all(train_solution.ok)):
        raise RuntimeError("reference-policy collocation simulation failed")
    train_log_states = train_solution.xs
    args = build_residual_args(train_log_states, kernel_set, k_0)
    n_train = args.train_log_states.shape[0]
    shifted_gram = args.gram + kernel_set.metric_epsilon * jnp.eye(n_train, dtype=dtype)
    metric = RepeatedFactorMetric(
        jnp.linalg.cholesky(shifted_gram, upper=True),
        repeats=1,
    )
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros((), dtype=dtype),
        jnp.zeros(2, dtype=dtype),
    )
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
    # Component attribution: the representer solution splits as
    # h_ray = sum_i alpha_i K_ray(., q_i) plus the 2-D remainder, with squared
    # RKHS norms alpha' G_ray alpha and alpha' G_2d alpha.
    ray_gram = rbf_state_gram(
        args.q_centers,
        jnp.asarray(kernel_set.ray_amplitude, dtype),
        jnp.asarray(kernel_set.ray_lengthscale, dtype),
    )
    alpha = theta[0]
    rkhs_norm_ray = float(alpha @ ray_gram @ alpha)
    rkhs_norm_2d = float(alpha @ (args.gram - ray_gram) @ alpha)
    policy = KernelInvestmentPolicy(theta, args, p)
    train_values = trajectory_values(policy, train_log_states, p)
    train_initial = np.asarray(train_initial_conditions)
    train_df = long_df(
        {"t": train_solution.ts, **train_values},
        ("trajectory", "step"),
        scalars={"z_0": train_initial[:, 0], "k_0": train_initial[:, 1]},
    )

    test_initial_log_states = jnp.log(test_initial_conditions)
    test_noise = sample_trajectory_noise(
        sde,
        jax.random.fold_in(test_ic_key, 1),
        ic_set.num_test_trajectories,
        n_test_steps,
        test_T / n_test_steps,
        dtype,
        2,
    )
    test_solution = simulate(
        sde,
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
    test_initial = np.asarray(test_initial_conditions)
    test_df = long_df(
        {"t": test_solution.ts, **test_values},
        ("trajectory", "step"),
        scalars={"z_0": test_initial[:, 0], "k_0": test_initial[:, 1]},
    )

    # Common-noise pathwise comparison against the reference NN policy: the
    # NN's held-out residuals are orders of magnitude below the FD reference,
    # so it is the tighter yardstick for the kernel solution.
    reference_solution = simulate(
        sde,
        reference_policy,
        test_initial_log_states,
        test_noise,
        test_T,
        n_test_steps,
        p,
    )
    reference_values = trajectory_values(reference_policy, reference_solution.xs, p)
    k_error, c_error = add_reference_path_diagnostics(
        test_df, test_values, reference_values
    )

    # Terminal transversality certificate on the training initial conditions.
    tvc_noise = sample_trajectory_noise(
        sde,
        jax.random.fold_in(train_ic_key, 4),
        ic_set.num_train_trajectories,
        data_set.tvc_steps,
        data_set.train_tvc_T / data_set.tvc_steps,
        dtype,
        2,
    )
    train_tvc = log_tvc_residual(
        sde,
        policy,
        jnp.log(train_initial_conditions),
        tvc_noise,
        data_set.train_tvc_T,
        data_set.tvc_steps,
        p,
    )
    max_log_tvc = float(jnp.max(train_tvc))

    max_train_euler = float(jnp.max(jnp.abs(train_values["euler_residual"])))
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    # Policy-function comparison on the realized state box: log k grid over
    # the observed range, z rows at the observed log z quantiles.
    observed_log_z = np.asarray(train_log_states[..., 0]).reshape(-1)
    observed_log_k = np.asarray(train_log_states[..., 1]).reshape(-1)
    log_k_grid = np.linspace(
        observed_log_k.min(), observed_log_k.max(), data_set.policy_grid_size
    )
    z_rows = np.quantile(observed_log_z, (0.05, 0.5, 0.95))
    grid_log_states = jnp.asarray(
        np.stack(
            [
                np.stack([np.full_like(log_k_grid, z_row), log_k_grid], axis=-1)
                for z_row in z_rows
            ]
        ),
        dtype=dtype,
    )
    grid_values = trajectory_values(policy, grid_log_states, p)
    reference_grid_values = trajectory_values(reference_policy, grid_log_states, p)
    c_grid_error = np.asarray(
        (grid_values["c"] - reference_grid_values["c"]) / reference_grid_values["c"]
    )

    test_euler_mean_abs = float(jnp.mean(jnp.abs(test_values["euler_residual"])))
    rejection_reasons = []
    if tolerance_set.gate_train and not solver_converged:
        rejection_reasons.append("train")
    if tolerance_set.gate_test and not (
        test_euler_mean_abs <= tolerance_set.test_euler_mean_abs
        and max_test_euler <= tolerance_set.test_euler_max
    ):
        rejection_reasons.append("test")
    results = {
        "parameters": p.to_dict(),
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": {
            "init_damping": lm_set.init_damping,
            "damping_decrease": lm_set.damping_decrease,
            "damping_increase": lm_set.damping_increase,
            "geodesic_acceleration": lm_set.geodesic_acceleration,
            "geodesic_acceptance_ratio": lm_set.geodesic_acceptance_ratio,
            "max_steps": lm_set.max_steps,
            "atol": lm_set.atol,
            "gtol": lm_set.gtol,
        },
        "ridge_set": {"ridge": ridge_set.ridge},
        "tolerance_set": asdict(tolerance_set),
        "reference_test_euler_mean_abs": float(reference_meta["test_euler_mean_abs"]),
        "sde_solver": sde_solver,
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
        "max_abs_diffusion_term": float(
            jnp.max(jnp.abs(test_values["diffusion_term"]))
        ),
        "max_log_tvc_residual": max_log_tvc,
        "min_test_c": float(jnp.min(test_values["c"])),
        "steps": steps,
        **acceptance_fields(rejection_reasons),
        "setup_time": setup_time,
        "solve_time": solve_time,
        "total_fit_time": total_fit_time,
        "num_centers": n_train,
        "rkhs_norm_ray": rkhs_norm_ray,
        "rkhs_norm_2d": rkhs_norm_2d,
        "test_mean_abs_k_rel_error": float(np.mean(np.abs(k_error))),
        "test_mean_abs_c_rel_error": float(np.mean(np.abs(c_error))),
        "test_max_abs_k_rel_error": float(np.max(np.abs(k_error))),
        "test_max_abs_c_rel_error": float(np.max(np.abs(c_error))),
        "policy_mean_abs_c_rel_error": float(np.mean(np.abs(c_grid_error))),
        "policy_max_abs_c_rel_error": float(np.max(np.abs(c_grid_error))),
        "observed_log_z_min": float(observed_log_z.min()),
        "observed_log_z_max": float(observed_log_z.max()),
        "observed_log_k_min": float(observed_log_k.min()),
        "observed_log_k_max": float(observed_log_k.max()),
    }

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")
    extras = {
        "policy": policy,
        "theta": theta,
        "residual_args": args,
        "parameters": p,
        "train_initial_conditions": train_initial_conditions,
        "test_initial_conditions": test_initial_conditions,
        "train_df": train_df,
        "test_df": test_df,
    }
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(stochastic_growth_kernel_recursive)
