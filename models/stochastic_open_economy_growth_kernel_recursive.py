from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from flax import struct
from nlls_gram import (
    AnnealRidge,
    LMStatus,
    RepeatedFactorMetric,
    RidgeLevenbergMarquardt,
)

from models.kernels import state_features
from models.stochastic_open_economy_growth import (
    PROCESS_CONSTANTS,
    DataSettings,
    InitialConditionSettings,
    ModelParameters,
    StructuralParameters,
    euler_residual_terms,
    make_sequential_policy,
    normalize_state,
    policy_outputs,
    problem_cache_config,
    production,
    simulate,
    simulate_train,
    trajectory_frame,
    trajectory_values,
    transversality_expectation_frame,
    transversality_statistics,
    transversality_trajectory_values,
)
from models.utilities import (
    SDE_SOLVERS,
    STATUS_TO_REASON,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    cache_key,
    load_model,
    sample_trajectory_noise,
    standardized_initial_states,
)


# Acceptance gates on the held-out batch, plus the ridge LM budget; the
# float32 replication driver passes its own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    test_euler_mean_abs: float = 5e-3
    test_euler_max: float = 3e-1
    max_steps: int = 100
    residual_atol: float = 1e-5
    gtol: float = 1e-7
    ridge: float = 1e-12
    # ridge_init=None solves at the fixed ridge above; a value enables ridge
    # continuation annealed from ridge_init down to `ridge` (the floor) by
    # factors of `ridge_decrease` on inner stationarity. The metric's
    # factorization never involves the ridge, so the anneal composes
    # unchanged.
    ridge_init: float | None = None
    ridge_decrease: float = 0.1
    # Relative stationarity threshold that triggers the next ridge level:
    # looser anneals faster (fewer steps per level), tighter solves each
    # level out more completely before descending.
    ridge_grad_rtol: float = 1e-2


# Kernel geometry lives in the NN's normalized coordinates, so one
# lengthscale is meaningful on every axis and the bounded asset squash never
# forces a retune when dispersion widens. Per-axis overrides default to
# `lengthscale`.
@dataclass(frozen=True)
class StateKernelSettings:
    # At this lengthscale the kernel carries levels and tilts, so the affine
    # mean defaults off; retune the geometry jointly, not per field.
    amplitude: float = 1.86
    lengthscale: float = 1.69
    metric_epsilon: float = 7.7e-5
    free_constant: bool = False
    linear_mean: bool = False
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
    asset_lengthscale: float | None = None
    rate_lengthscale: float | None = None


@struct.dataclass
class ResidualArgs:
    centers: jax.Array
    train_states: jax.Array
    gram: jax.Array
    state_center: jax.Array
    state_scale: jax.Array
    investment_anchor: jax.Array
    current_account_anchor: jax.Array
    output_scale: jax.Array
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)


def state_lengthscales(kernel_set):
    per_axis = (
        kernel_set.log_z_lengthscale,
        kernel_set.log_k_lengthscale,
        kernel_set.asset_lengthscale,
        kernel_set.rate_lengthscale,
    )
    return tuple(kernel_set.lengthscale if ell is None else ell for ell in per_axis)


def build_residual_args(train_states, kernel_set, reference_policy):
    flat_states = train_states.reshape(-1, 4)
    state_center = reference_policy.state_center[...]
    state_scale = reference_policy.state_scale[...]
    centers = jax.vmap(lambda state: normalize_state(state, state_center, state_scale))(
        flat_states
    )
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
        train_states=flat_states,
        gram=gram,
        state_center=state_center,
        state_scale=state_scale,
        investment_anchor=reference_policy.investment_anchor[...],
        current_account_anchor=reference_policy.current_account_anchor[...],
        output_scale=reference_policy.output_scale[...],
        kernel_set=kernel_set,
    )


# Two RKHS heads through the NN's imported normalization/output maps: zero
# theta is exactly the zero-initialized NN policy. theta = (alpha_x[n],
# alpha_ca[n], [const(2)], [linear_mean(4, 2)]); the alpha blocks must stay
# first -- RepeatedFactorMetric(repeats=2) penalizes the leading 2n coords.
@struct.dataclass
class KernelSequentialPolicy:
    theta: tuple
    args: ResidualArgs
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)

    def __call__(self, state):
        x = normalize_state(state, self.args.state_center, self.args.state_scale)
        features = state_features(
            self.kernel_set.state_kernel,
            x,
            self.args.centers,
            self.kernel_set.amplitude,
            state_lengthscales(self.kernel_set),
        )
        raw = jnp.stack([features @ self.theta[0], features @ self.theta[1]])
        index = 2
        if self.kernel_set.free_constant:
            raw = raw + self.theta[index]
            index += 1
        if self.kernel_set.linear_mean:
            raw = raw + x @ self.theta[index]
        return policy_outputs(
            raw,
            self.args.investment_anchor,
            self.args.current_account_anchor,
            self.args.output_scale,
        )


def residual_fn(theta, args, p):
    policy = KernelSequentialPolicy(theta=theta, args=args, kernel_set=args.kernel_set)
    residuals = euler_residual_terms(policy, args.train_states, p)[0]
    count = jnp.asarray(residuals.size, dtype=residuals.dtype)
    return residuals.reshape(-1) / jnp.sqrt(count)


# Best accepted artifact for this problem hash by held-out residual; no
# fallback -- cache the NN first with --policy_cache.
def load_reference_policy(policy_dir, problem_config):
    problem_key = cache_key(problem_config)
    pattern = f"stochastic_open_economy_growth_nn_recursive_policy_{problem_key}-*.npz"
    candidates = []
    for path in sorted(pathlib.Path(policy_dir).glob(pattern)):
        meta = json.loads(str(np.load(path)["__meta__"]))
        if meta.get("solution_accepted", False):
            candidates.append((meta.get("test_euler_mean_abs", np.inf), path))
    if not candidates:
        raise FileNotFoundError(
            f"no accepted NN policy for problem {problem_key} under {policy_dir}; "
            "run stochastic_open_economy_growth_nn_recursive with "
            "--policy_cache first"
        )
    best = min(candidates, key=lambda pair: pair[0])[1]
    policy, meta = load_model(best, make_sequential_policy)
    return policy, meta, best


# Same model and residuals as the NN solver, but the collocation states are
# a frozen design: simulated once under the saved reference NN policy.
def stochastic_open_economy_growth_kernel_recursive(
    k_0: float = 1.0,
    a_0: float = -0.8,
    z_0: float = 1.0,
    rate_gap_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    kappa_z: float = PROCESS_CONSTANTS.kappa_z,
    sigma_z: float = PROCESS_CONSTANTS.sigma_z,
    z_bar: float = 1.0,
    kappa_r: float = PROCESS_CONSTANTS.kappa_r,
    sigma_r: float = PROCESS_CONSTANTS.sigma_r,
    corr_zr: float = 0.0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    policy_dir: str = ".results",
    # The NN's horizons at half its density: the frozen design needs
    # coverage, not resolution, and the shared train/test protocol keeps the
    # two solvers' held-out euler errors comparable.
    data_set: DataSettings = DataSettings(train_points=49),
    # Matches the NN driver's default problem, so a module-default NN run
    # with --policy_cache feeds a module-default kernel run.
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: StateKernelSettings = StateKernelSettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=4.9e-4,
        damping_decrease=0.16,
        damping_increase=8.5,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.75,
    ),
    # residual_atol must clear the seed-dependent ridge-floor residual
    # (conjunctive stopping; observed floors reach ~5e-5). The held-out max
    # bar guards against simulation divergence, not NN accuracy: the
    # frozen-design kernel is coarser than the NN it consumes and blow-ups
    # land at 1e3+.
    tolerance_set: ToleranceSettings = ToleranceSettings(
        max_steps=300,
        residual_atol=1e-4,
        gtol=7.4e-7,
        ridge=5.4e-11,
    ),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32

    train_variances = (
        ic_set.train_log_z_variance,
        ic_set.train_log_k_variance,
        ic_set.train_asset_variance,
        ic_set.train_rate_variance,
    )
    test_variances = tuple(
        train if test is None else test
        for train, test in zip(
            train_variances,
            (
                ic_set.test_log_z_variance,
                ic_set.test_log_k_variance,
                ic_set.test_asset_variance,
                ic_set.test_rate_variance,
            ),
            strict=True,
        )
    )
    output_scale = float(production(k_0, z_0, parameters))
    initial_center = jnp.asarray(
        [math.log(z_0), math.log(k_0), a_0 / output_scale, rate_gap_0],
        dtype=dtype,
    )
    train_scales = jnp.sqrt(jnp.asarray(train_variances, dtype=dtype))
    test_scales = jnp.sqrt(jnp.asarray(test_variances, dtype=dtype))
    p = StructuralParameters.create(
        dtype,
        alpha=parameters.alpha,
        gamma=parameters.gamma,
        delta=parameters.delta,
        preference_rate=parameters.r,
        investment_eta=parameters.eta,
        interest_rate_mean=parameters.r_star,
        chi=parameters.delta ** (1.0 - parameters.eta),
        kappa_z=kappa_z,
        sigma_z=sigma_z,
        log_z_bar=math.log(z_bar),
        kappa_r=kappa_r,
        sigma_r=sigma_r,
        corr_zr=corr_zr,
        output_scale=output_scale,
    )
    sde = SDE_SOLVERS[sde_solver]
    n_train_steps = data_set.train_points - 1
    n_test_steps = data_set.test_points - 1
    n_tvc_steps = data_set.tvc_points - 1
    train_dt = data_set.train_T / n_train_steps
    test_dt = data_set.test_T / n_test_steps
    tvc_dt = data_set.tvc_T / n_tvc_steps

    data_key = jax.random.key(ic_set.data_seed)
    train_ic_key, test_ic_key = jax.random.split(data_key)
    train_initial_states = standardized_initial_states(
        train_ic_key,
        initial_center,
        train_scales,
        ic_set.num_train_trajectories,
    )
    test_initial_states = standardized_initial_states(
        test_ic_key,
        initial_center,
        test_scales,
        ic_set.num_test_trajectories,
    )

    problem_config = problem_cache_config(
        parameters,
        kappa_z,
        sigma_z,
        z_bar,
        kappa_r,
        sigma_r,
        corr_zr,
        k_0,
        a_0,
        z_0,
        rate_gap_0,
        use_float64,
        ic_set.num_train_trajectories,
        train_variances,
    )
    reference_policy, reference_meta, reference_path = load_reference_policy(
        policy_dir, problem_config
    )
    # A scale mismatch means the artifact's problem differs from its hash.
    saved_settings = reference_meta["settings"]
    investment_anchor = parameters.delta * k_0
    if not (
        math.isclose(saved_settings["output_scale"], output_scale)
        and math.isclose(saved_settings["investment_anchor"], investment_anchor)
    ):
        raise RuntimeError(
            "reference policy scales disagree with the requested problem: "
            f"saved (output_scale, investment_anchor) = "
            f"({saved_settings['output_scale']}, "
            f"{saved_settings['investment_anchor']}), "
            f"recomputed ({output_scale}, {investment_anchor})"
        )
    if verbose:
        print(
            f"reference NN policy {reference_path} "
            f"(test_euler_mean_abs {reference_meta['test_euler_mean_abs']:.3e})"
        )

    fit_timer = Stopwatch()
    # Frozen design at full shock scale on the NN driver's training noise
    # key (fold_in index 1); the discretization is deliberately coarser.
    train_noise = sample_trajectory_noise(
        sde,
        jax.random.fold_in(train_ic_key, 1),
        ic_set.num_train_trajectories,
        n_train_steps,
        train_dt,
        dtype,
        4,
    )
    train_solution = simulate_train(
        sde,
        reference_policy,
        train_initial_states,
        train_noise,
        p,
        data_set,
    )
    # Fixed-step solve_sde reports ok even on divergence: gate on finiteness.
    if not bool(jnp.all(jnp.isfinite(train_solution.xs))):
        raise RuntimeError("reference-policy collocation simulation failed")
    train_states = train_solution.xs
    args = build_residual_args(train_states, kernel_set, reference_policy)
    n_train = args.centers.shape[0]
    shifted_gram = args.gram + kernel_set.metric_epsilon * jnp.eye(n_train, dtype=dtype)
    factor = jnp.linalg.cholesky(shifted_gram, upper=True)
    if not bool(jnp.all(jnp.isfinite(factor))):
        raise RuntimeError("shifted gram factorization failed")
    metric = RepeatedFactorMetric(factor, repeats=2)
    theta_0 = (
        jnp.zeros(n_train, dtype=dtype),
        jnp.zeros(n_train, dtype=dtype),
    )
    if kernel_set.free_constant:
        theta_0 = theta_0 + (jnp.zeros(2, dtype=dtype),)
    if kernel_set.linear_mean:
        theta_0 = theta_0 + (jnp.zeros((4, 2), dtype=dtype),)
    if tolerance_set.ridge_init is None:
        initial_ridge = tolerance_set.ridge
        continuation_options = {}
    else:
        initial_ridge = tolerance_set.ridge_init
        callback = AnnealRidge(
            ridge_floor=tolerance_set.ridge,
            decrease=tolerance_set.ridge_decrease,
            grad_rtol=tolerance_set.ridge_grad_rtol,
        )
        continuation_options = {
            "callback": callback,
            "user_state": callback.init_state(dtype),
        }
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=initial_ridge,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )

    solve_timer = Stopwatch()
    setup_time = fit_timer.elapsed(factor)
    result = solver.solve(
        theta_0,
        args,
        p=p,
        max_steps=tolerance_set.max_steps,
        atol=tolerance_set.residual_atol,
        gtol=tolerance_set.gtol,
        max_steps_is_success=False,
        **continuation_options,
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
    policy = KernelSequentialPolicy(theta=theta, args=args, kernel_set=kernel_set)
    train_values = trajectory_values(policy, train_states, p)
    train_df = trajectory_frame(train_values, train_solution.ts)
    train_df["fixed_reference_center"] = True

    test_noise = sample_trajectory_noise(
        sde,
        jax.random.fold_in(test_ic_key, 1),
        ic_set.num_test_trajectories,
        n_test_steps,
        test_dt,
        dtype,
        4,
    )
    test_solution = simulate(
        sde,
        policy,
        test_initial_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    jax.block_until_ready(test_solution)
    test_states = jnp.where(test_solution.ok[:, None, None], test_solution.xs, jnp.nan)
    test_values = trajectory_values(policy, test_states, p)
    test_euler = jnp.stack([test_values["c_euler"], test_values["q_euler"]], axis=-1)
    test_path_ok = bool(
        jnp.all(test_solution.ok)
        and jnp.all(jnp.isfinite(test_states))
        and jnp.all(jnp.isfinite(test_euler))
        and jnp.min(test_values["k"]) > 0.0
        and jnp.min(test_values["q"]) > 0.0
        and jnp.min(test_values["c"]) > 0.0
        and jnp.min(test_values["x"]) > 0.0
    )
    test_df = trajectory_frame(test_values, test_solution.ts)

    # Same TVC noise stream (fold_in index 3) as the NN driver.
    tvc_initial_states = jnp.broadcast_to(
        initial_center,
        (ic_set.num_tvc_trajectories, initial_center.size),
    )
    tvc_noise = sample_trajectory_noise(
        sde,
        jax.random.fold_in(test_ic_key, 3),
        ic_set.num_tvc_trajectories,
        n_tvc_steps,
        tvc_dt,
        dtype,
        4,
    )
    tvc_solution = simulate(
        sde,
        policy,
        tvc_initial_states,
        tvc_noise,
        data_set.tvc_T,
        n_tvc_steps,
        p,
    )
    tvc_states = jnp.where(tvc_solution.ok[:, None, None], tvc_solution.xs, jnp.nan)
    tvc_values = transversality_trajectory_values(
        policy,
        tvc_states,
        tvc_solution.ts,
        p,
    )
    tvc_statistics = transversality_statistics(tvc_values)
    tvc_df = trajectory_frame(tvc_values, tvc_solution.ts)
    expected_tvc_df = transversality_expectation_frame(tvc_values, tvc_solution.ts)

    # Common-noise pathwise comparison against the reference NN policy.
    reference_solution = simulate(
        sde,
        reference_policy,
        test_initial_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    reference_path_ok = bool(
        jnp.all(reference_solution.ok) and jnp.all(jnp.isfinite(reference_solution.xs))
    )
    reference_values = trajectory_values(reference_policy, reference_solution.xs, p)
    k_error = (test_values["k"] - reference_values["k"]) / jnp.abs(
        reference_values["k"]
    )
    c_error = (test_values["c"] - reference_values["c"]) / jnp.abs(
        reference_values["c"]
    )
    asset_error = (test_values["a"] - reference_values["a"]) / output_scale
    test_df["k_benchmark"] = np.asarray(reference_values["k"]).reshape(-1)
    test_df["c_benchmark"] = np.asarray(reference_values["c"]).reshape(-1)
    test_df["a_benchmark"] = np.asarray(reference_values["a"]).reshape(-1)
    test_df["k_rel_error"] = np.asarray(k_error).reshape(-1)
    test_df["c_rel_error"] = np.asarray(c_error).reshape(-1)
    test_df["a_scaled_error"] = np.asarray(asset_error).reshape(-1)

    train_euler = jnp.stack([train_values["c_euler"], train_values["q_euler"]], axis=-1)
    test_mean_abs = jnp.mean(jnp.abs(test_euler), axis=(0, 1))
    test_euler_max_abs = float(jnp.max(jnp.abs(test_euler)))

    rejection_reasons = []
    if not solver_converged:
        rejection_reasons.append("solver")
    if not test_path_ok:
        rejection_reasons.append("test_path")
    if not float(jnp.mean(jnp.abs(test_euler))) <= tolerance_set.test_euler_mean_abs:
        rejection_reasons.append("test_euler_mean")
    if test_euler_max_abs > tolerance_set.test_euler_max:
        rejection_reasons.append("test_euler_max")
    if not reference_path_ok:
        rejection_reasons.append("reference_path")
    acceptance = acceptance_fields(rejection_reasons)

    gram_eigenvalues = jnp.linalg.eigvalsh(args.gram)
    lengthscales = state_lengthscales(kernel_set)
    free_parameter_count = (2 if kernel_set.free_constant else 0) + (
        8 if kernel_set.linear_mean else 0
    )
    results = {
        "reference_test_euler_mean_abs": float(reference_meta["test_euler_mean_abs"]),
        **asdict(parameters),
        "k_0": k_0,
        "a_0": a_0,
        "z_0": z_0,
        "rate_gap_0": rate_gap_0,
        "kappa_z": kappa_z,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "kappa_r": kappa_r,
        "sigma_r": sigma_r,
        "corr_zr": corr_zr,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_values["c_euler"].dtype),
        "sde_solver": sde_solver,
        "kernel": kernel_set.state_kernel,
        "amplitude": kernel_set.amplitude,
        "log_z_lengthscale": lengthscales[0],
        "log_k_lengthscale": lengthscales[1],
        "asset_lengthscale": lengthscales[2],
        "rate_lengthscale": lengthscales[3],
        "metric_epsilon": kernel_set.metric_epsilon,
        "ridge": tolerance_set.ridge,
        "ridge_init": tolerance_set.ridge_init,
        "ridge_decrease": tolerance_set.ridge_decrease,
        "ridge_grad_rtol": tolerance_set.ridge_grad_rtol,
        "train_loss": train_loss,
        "train_euler_mean_abs": float(jnp.mean(jnp.abs(train_euler))),
        "train_euler_max_abs": float(jnp.max(jnp.abs(train_euler))),
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_euler))),
        # NaN when rejected: the HPO sweeps sort on this.
        "test_euler_mean_abs_success": (
            float(jnp.mean(jnp.abs(test_euler)))
            if acceptance["solution_accepted"]
            else np.nan
        ),
        "test_euler_max_abs": test_euler_max_abs,
        "test_c_euler_mean_abs": float(test_mean_abs[0]),
        "test_q_euler_mean_abs": float(test_mean_abs[1]),
        "min_test_consumption": float(jnp.min(test_values["c"])),
        "test_mean_abs_k_rel_error": float(jnp.mean(jnp.abs(k_error))),
        "test_max_abs_k_rel_error": float(jnp.max(jnp.abs(k_error))),
        "test_mean_abs_c_rel_error": float(jnp.mean(jnp.abs(c_error))),
        "test_max_abs_c_rel_error": float(jnp.max(jnp.abs(c_error))),
        "test_mean_abs_asset_scaled_error": float(jnp.mean(jnp.abs(asset_error))),
        "test_max_abs_asset_scaled_error": float(jnp.max(jnp.abs(asset_error))),
        "steps": steps,
        "setup_time": setup_time,
        "solve_time": solve_time,
        "total_fit_time": total_fit_time,
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        "rkhs_coefficients": 2 * n_train,
        "free_parameter_count": free_parameter_count,
        "gram_min_eigenvalue": float(gram_eigenvalues[0]),
        "gram_max_eigenvalue": float(gram_eigenvalues[-1]),
        "observed_asset_center_min": float(jnp.min(args.centers[:, 2])),
        "observed_asset_center_max": float(jnp.max(args.centers[:, 2])),
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
        "test_points": data_set.test_points,
        "test_T": data_set.test_T,
        "tvc_T": data_set.tvc_T,
        "tvc_points": data_set.tvc_points,
        "num_tvc_trajectories": ic_set.num_tvc_trajectories,
        "ex_post_expected_absolute_asset_tvc": float(
            tvc_statistics["expected_absolute_asset_tvc"]
        ),
        "ex_post_expected_capital_tvc": float(tvc_statistics["expected_capital_tvc"]),
        "ex_post_signed_expected_asset_tvc": float(
            tvc_statistics["signed_expected_asset_tvc"]
        ),
        "ex_post_absolute_asset_tvc_mc_se": float(
            tvc_statistics["absolute_asset_tvc_mc_se"]
        ),
        "ex_post_signed_asset_tvc_mc_se": float(
            tvc_statistics["signed_asset_tvc_mc_se"]
        ),
        "ex_post_capital_tvc_mc_se": float(tvc_statistics["capital_tvc_mc_se"]),
        **acceptance,
    }

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "parameters": asdict(parameters),
        "shocks": {
            "kappa_z": kappa_z,
            "sigma_z": sigma_z,
            "z_bar": z_bar,
            "kappa_r": kappa_r,
            "sigma_r": sigma_r,
            "corr_zr": corr_zr,
        },
        "initial_conditions": {
            "k_0": k_0,
            "a_0": a_0,
            "z_0": z_0,
            "rate_gap_0": rate_gap_0,
        },
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
        "train_initial_conditions": train_initial_states,
        "test_initial_conditions": test_initial_states,
        "train_df": train_df,
        "test_df": test_df,
        "tvc_df": tvc_df,
        "expected_tvc_df": expected_tvc_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(stochastic_open_economy_growth_kernel_recursive)
