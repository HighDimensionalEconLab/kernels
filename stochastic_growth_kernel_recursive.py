from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt

from neoclassical_growth_kernel_recursive import (
    StateKernelSettings,
    StoppingCriteria,
    build_residual_args,
    make_policy,
    state_lengthscales,
)
from neoclassical_growth_nn_recursive import (
    InitialConditionSettings,
    collocation_step_diagnostics,
    draw_initial_conditions,
    load_investment_policy,
    policy_cache_key,
    trajectory_frame,
)
from stochastic_growth_nn_recursive import (
    SDE_SOLVERS,
    DataSettings,
    StructuralParameters,
    add_fd_path_diagnostics,
    collocation_rollout,
    euler_residuals,
    log_tvc_residual,
    problem_cache_config,
    rollout_steps,
    sample_trajectory_noise,
    trajectory_values,
)
from utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from utilities import (
    STATUS_TO_REASON,
    OptimizerSettings,
)


# Locate the reference NN policy for this exact problem (structural
# parameters, dtype, and training initial-condition design) by its problem
# hash; among matching runs, take the accepted artifact with the smallest
# held-out residual. There is deliberately no fallback: solve and cache the
# NN first via stochastic_growth_nn_recursive --policy_cache <dir>.
def load_reference_policy(policy_dir, problem_config):
    problem_key = policy_cache_key(problem_config)
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
    policy, meta = load_investment_policy(best)
    return policy, meta, best


def residual_fn(theta, args, p, kernel_set):
    policy = make_policy(theta, args, p, kernel_set)
    residual = euler_residuals(policy, args.train_log_states, p)
    return residual / jnp.sqrt(residual.size)


# The kernel representation of the same stochastic recursive model as
# stochastic_growth_nn_recursive: identical policy output map, residuals, and
# diagnostics, but the collocation states are a frozen design -- simulated
# once under the saved reference NN policy -- rather than re-integrated by
# differentiating through the solver's own rollout.
def stochastic_growth_kernel_recursive(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    eta: float = CONSTANTS.eta,
    sigma_z: float = CONSTANTS.sigma_z_high,
    z_bar: float = 1.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    policy_dir: str = ".results",
    # Half the NN's collocation density: the frozen design needs coverage,
    # not integration accuracy, and 16 x 33 states solve ~3x faster at
    # near-identical held-out quality.
    data_set: DataSettings = DataSettings(train_points=33),
    ic_set: InitialConditionSettings = InitialConditionSettings(
        num_train_trajectories=16,
        train_log_z_variance=CONSTANTS.z_ergodic_variance_high,
        train_log_k_variance=0.01,
    ),
    kernel_set: StateKernelSettings = StateKernelSettings(
        homothetic_representation=False,
        state_kernel="ard_rbf",
        linear_mean=True,
        log_z_lengthscale=0.6,
        log_k_lengthscale=0.65,
    ),
    opt_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-6,
        damping_decrease=0.25,
        damping_increase=4.0,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ),
    # The float64 ridge-LM residual floor of this representation on the 2-D
    # stochastic state box sits at ~1.3e-5 RMS, slightly above the
    # deterministic variant's floor.
    stop_set: StoppingCriteria = StoppingCriteria(max_steps=150, residual_atol=2e-5),
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
    if sde_solver not in SDE_SOLVERS:
        raise ValueError(f"sde_solver must be one of {sorted(SDE_SOLVERS)}")
    # Mean reversion in z fixes a scale, so the homothetic representation is
    # misspecified by construction for this model.
    if kernel_set.homothetic_representation:
        raise ValueError(
            "the stochastic model requires homothetic_representation=False"
        )
    if stop_set.residual_atol <= 0.0 or stop_set.gtol <= 0.0:
        raise ValueError("residual_atol and gtol must be positive")
    if stop_set.ridge <= 0.0:
        raise ValueError("ridge must be positive")
    if not all(ell > 0.0 for ell in state_lengthscales(kernel_set)):
        raise ValueError("kernel lengthscales must be positive")

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
    if verbose:
        print(
            f"reference NN policy {reference_path} "
            f"(test_euler_mean_abs {reference_meta['test_euler_mean_abs']:.3e})"
        )

    fit_start = time.perf_counter()
    # Frozen collocation design: the reference NN policy's own rollout on the
    # training noise stream; the kernel never differentiates trajectories.
    train_noise = sample_trajectory_noise(
        solver,
        jax.random.fold_in(train_ic_key, 3),
        ic_set.num_train_trajectories,
        n_train_steps,
        train_dt,
        dtype,
    )
    train_solution = collocation_rollout(
        solver,
        reference_policy,
        jnp.log(train_initial_conditions),
        train_noise,
        p,
        data_set,
    )
    if not bool(jnp.all(train_solution.ok)):
        raise RuntimeError("reference-policy collocation rollout failed")
    train_log_states = train_solution.xs
    args = build_residual_args(train_log_states, kernel_set, z_bar, k_0)
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
    lm_solver = RidgeLevenbergMarquardt(
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
    result = lm_solver.solve(
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
    policy = make_policy(theta, args, p, kernel_set)
    train_values = trajectory_values(policy, train_log_states, p)
    train_df = trajectory_frame(
        train_values,
        train_solution.ts,
        train_initial_conditions,
    )
    train_df["fixed_reference_center"] = True
    collocation_diagnostics = collocation_step_diagnostics(train_solution.ts)

    test_initial_log_states = jnp.log(test_initial_conditions)
    test_noise = sample_trajectory_noise(
        solver,
        jax.random.fold_in(test_ic_key, 1),
        ic_set.num_test_trajectories,
        n_test_steps,
        test_T / n_test_steps,
        dtype,
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

    # Common-noise pathwise comparison against the reference NN policy: the
    # NN's held-out residuals are orders of magnitude below the FD reference,
    # so it is the tighter yardstick for the kernel solution.
    reference_solution = rollout_steps(
        solver,
        reference_policy,
        test_initial_log_states,
        test_noise,
        test_T,
        n_test_steps,
        p,
    )
    reference_values = trajectory_values(reference_policy, reference_solution.xs, p)
    k_error, c_error, test_summary_df = add_fd_path_diagnostics(
        test_df, test_values, reference_values, test_initial_conditions
    )

    tvc_noise = sample_trajectory_noise(
        solver,
        jax.random.fold_in(train_ic_key, 4),
        ic_set.num_train_trajectories,
        data_set.tvc_steps,
        data_set.train_tvc_T / data_set.tvc_steps,
        dtype,
    )
    max_log_tvc, train_terminal_log_tvc, tvc_ok = log_tvc_residual(
        solver,
        policy,
        jnp.log(train_initial_conditions),
        tvc_noise,
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
        jax.random.fold_in(test_ic_key, 2),
        ic_set.num_test_trajectories,
        data_set.tvc_steps,
        data_set.train_tvc_T / data_set.tvc_steps,
        dtype,
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

    gram_eigenvalues = jnp.linalg.eigvalsh(args.gram)
    results = {
        "variant": "fixed_recursive_kernel_stochastic",
        "collocation_design": "frozen_reference_nn_rollout",
        "parameter_derivative": "conditional_on_fixed_collocation_design",
        "reference_policy": str(reference_path),
        "reference_test_euler_mean_abs": float(reference_meta["test_euler_mean_abs"]),
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "z_0": z_0,
        "k_0": k_0,
        "test_z_0": test_z_center,
        "test_k_0": test_k_center,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_values["euler_residual"].dtype),
        "sde_solver": sde_solver,
        "state_coordinate": "(log z, log k)",
        "policy": "delta_k0_unit_softplus",
        "kernel": kernel_set.state_kernel,
        "amplitude": kernel_set.amplitude,
        "log_z_lengthscale": state_lengthscales(kernel_set)[0],
        "log_k_lengthscale": state_lengthscales(kernel_set)[1],
        "metric_epsilon": kernel_set.metric_epsilon,
        "metric": "shifted_rkhs_gram",
        "ridge": stop_set.ridge,
        "free_constant": kernel_set.free_constant,
        "linear_mean": kernel_set.linear_mean,
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
        "mean_log_tvc_residual": float(jnp.mean(train_terminal_log_tvc)),
        "test_mean_log_tvc_residual": float(jnp.mean(test_terminal_log_tvc)),
        "test_max_log_tvc_residual": test_max_log_tvc,
        "transversality_ok": transversality_ok,
        "test_transversality_ok": test_transversality_ok,
        "min_test_c": float(jnp.min(test_values["c"])),
        "test_path_ok": test_path_ok,
        "steps": steps,
        "stopping_reason": stopping_reason,
        "solver_converged": solver_converged,
        "solution_accepted": bool(
            solver_converged and test_path_ok and transversality_ok
        ),
        "setup_time": setup_time,
        "solve_time": solve_time,
        "total_fit_time": total_fit_time,
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        **collocation_diagnostics,
        "rkhs_coefficients": n_train,
        "gram_min_eigenvalue": float(gram_eigenvalues[0]),
        "gram_max_eigenvalue": float(gram_eigenvalues[-1]),
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
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
        "test_points": data_set.test_points,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

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
        "hyperparameters": hyperparameters,
        "test_summary_df": test_summary_df,
    }
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(stochastic_growth_kernel_recursive)
