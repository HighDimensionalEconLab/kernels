from __future__ import annotations

import math
import pathlib
from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import LevenbergMarquardt, LMAction, LMStatus, MultiStart

from models.stochastic_open_economy_growth import (
    WORLD_RATE_KAPPA,
    WORLD_RATE_SIGMA,
    DataSettings,
    DrawSequentialPolicy,
    InitialConditionSettings,
    ModelParameters,
    SequentialPolicyNN,
    SequentialPolicySettings,
    StructuralParameters,
    ToleranceSettings,
    collocation_residuals,
    make_sequential_policy,
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
    ACTIVATIONS,
    DOMAIN_EPS,
    NEOCLASSICAL_GROWTH_CONSTANTS,
    SDE_SOLVERS,
    STATUS_TO_REASON,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    cache_key,
    load_model,
    sample_trajectory_noise,
    save_model,
    standardized_initial_states,
)

# Recursive stochastic small open economy on (log z, log k, a/output_scale,
# rate_gap). No benchmark or steady-state information enters the residual or
# initialization; the deterministic benchmark is ex-post diagnostics only.

# The residual differentiates the policy twice, so no kinked activation.
SMOOTH_ACTIVATIONS = frozenset(ACTIVATIONS) - {"relu"}


@dataclass(frozen=True)
class StoppingCriteria:
    # max_steps is the per-epoch LM budget; noise (and dispersed ICs) redraw
    # at epoch boundaries. The shock ladder rises geometrically to 1.0 over
    # continuation_epochs; convergence is only declared at scale 1.0.
    max_steps: int = 250
    train_loss_threshold: float = 1e-10
    num_attempts: int = 2
    num_epochs: int = 5
    epoch_early_stopping: bool = True
    shock_continuation: bool = True
    initial_shock_scale: float = 0.238
    continuation_epochs: int = 2


# ICs, noise, keys, and ladder position live in args, so the implicit
# derivative is conditional on the carried batch.
@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    initial_states: jax.Array
    initial_center: jax.Array
    initial_scales: jax.Array
    noise: object
    epoch_key: jax.Array
    restart_key: jax.Array
    epoch: jax.Array
    shock_scales: jax.Array
    shock_scale: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    solver: object = struct.field(pytree_node=False)
    data_set: DataSettings = struct.field(pytree_node=False)


# Interior epoch boundaries redraw the batch, step the ladder, and stop on a
# nonfinite or (full-scale) converged fresh-batch loss.
@dataclass(frozen=True)
class EpochCallback:
    num_train_trajectories: int
    n_steps: int
    dt: float
    epoch_early_stopping: bool
    resample_initial_states: bool

    def next_epoch(self, ctx):
        epoch_key, draw_key, noise_key = jax.random.split(ctx.args.epoch_key, 3)
        dtype = ctx.args.initial_states.dtype
        epoch = ctx.args.epoch + 1
        shock_scale = ctx.args.shock_scales[epoch]
        if self.resample_initial_states:
            # The stochastic axes' IC spread rides the shock ladder: partial-
            # scale epochs draw z / rate starts with proportionally shrunk
            # dispersion (inert at the zero-variance default).
            scale_mask = jnp.asarray([shock_scale, 1.0, 1.0, shock_scale], dtype=dtype)
            initial_states = standardized_initial_states(
                draw_key,
                ctx.args.initial_center,
                ctx.args.initial_scales * scale_mask,
                self.num_train_trajectories,
            )
        else:
            initial_states = ctx.args.initial_states
        fresh = ctx.args.replace(
            initial_states=initial_states,
            noise=sample_trajectory_noise(
                ctx.args.solver,
                noise_key,
                self.num_train_trajectories,
                self.n_steps,
                self.dt,
                dtype,
                4,
            ),
            epoch_key=epoch_key,
            epoch=epoch,
            shock_scale=shock_scale,
        )
        residual = residual_fn(ctx.x, fresh, ctx.p)
        loss = jnp.sum(residual * residual)
        nonfinite = ~jnp.isfinite(loss)
        if self.epoch_early_stopping:
            converged = (shock_scale == 1.0) & (loss < fresh.train_loss_threshold)
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


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.nondiff)
    # The ladder scales both sigmas through p (simulation and residual agree);
    # corr_zr is scale-free.
    stage_parameters = p.replace(
        sigma_z=p.sigma_z * args.shock_scale,
        sigma_r=p.sigma_r * args.shock_scale,
    )
    return collocation_residuals(
        args.solver,
        policy,
        args.initial_states,
        args.noise,
        stage_parameters,
        args.data_set,
    ).reshape(-1)


def accept_training(_, result):
    residual = residual_fn(result.x, result.args, result.p)
    loss = jnp.sum(residual * residual)
    return (
        (result.args.shock_scale == 1.0)
        & jnp.isfinite(loss)
        & (loss < result.args.train_loss_threshold)
    )


# kappa_z / sigma_z reuse the neoclassical family's calibration, so every
# stochastic recursive variant faces the identical z process.
def stochastic_open_economy_growth_nn_recursive(
    k_0: float = 1.0,
    # Debtor default: net debt ~ half of steady-state output (cf. Mendoza &
    # Villalvazo 2020).
    a_0: float = -0.8,
    z_0: float = 1.0,
    rate_gap_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    kappa_z: float = NEOCLASSICAL_GROWTH_CONSTANTS.eta,
    sigma_z: float = NEOCLASSICAL_GROWTH_CONSTANTS.sigma_z_high,
    z_bar: float = 1.0,
    kappa_r: float = WORLD_RATE_KAPPA,
    sigma_r: float = WORLD_RATE_SIGMA,
    corr_zr: float = 0.0,
    seed: int = 0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    nn_set: SequentialPolicySettings = SequentialPolicySettings(),
    opt_set: OptimizerSettings = OptimizerSettings(
        init_damping=5.1e-4,
        damping_decrease=0.195,
        damping_increase=3.22,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.782,
    ),
    stop_set: StoppingCriteria = StoppingCriteria(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    asset_shift_scale: float = 0.5,
    # Content-addressed artifact dir: an existing artifact for this exact
    # configuration loads instead of training; a fresh solve saves there.
    policy_cache: str | None = None,
    save_policy: str | None = None,
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    if nn_set.hidden_activation not in SMOOTH_ACTIVATIONS:
        raise ValueError(
            f"hidden_activation must be one of {sorted(SMOOTH_ACTIVATIONS)}"
        )

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
    if ic_set.num_train_trajectories == 1 and any(
        value > 0.0 for value in train_variances
    ):
        raise ValueError(
            "multiple training trajectories are required for nonzero spread"
        )
    if ic_set.num_test_trajectories == 1 and any(
        value > 0.0 for value in test_variances
    ):
        raise ValueError("multiple test trajectories are required for nonzero spread")

    z_ergodic_variance = sigma_z**2 / (2.0 * kappa_z)
    rate_ergodic_variance = sigma_r**2 / (2.0 * kappa_r)
    output_scale = float(production(k_0, z_0, parameters))
    initial_center = jnp.asarray(
        [math.log(z_0), math.log(k_0), a_0 / output_scale, rate_gap_0],
        dtype=dtype,
    )
    train_scales = jnp.sqrt(jnp.asarray(train_variances, dtype=dtype))
    test_scales = jnp.sqrt(jnp.asarray(test_variances, dtype=dtype))
    investment_anchor = max(parameters.delta * k_0, DOMAIN_EPS)
    current_account_anchor = 0.0
    nn_set = replace(
        nn_set,
        log_z_center=math.log(z_bar),
        log_z_scale=max(
            math.sqrt(z_ergodic_variance),
            math.sqrt(train_variances[0]),
            DOMAIN_EPS,
        ),
        log_k_center=math.log(k_0),
        asset_center=a_0 / output_scale,
        # NFA travels ~an output unit per decade, so the squash scale floors
        # at 0.1 * train_T; a unit floor saturates the tanh mid-transition.
        asset_scale=max(math.sqrt(train_variances[2]), 0.1 * data_set.train_T, 1.0),
        rate_scale=max(
            math.sqrt(rate_ergodic_variance),
            math.sqrt(train_variances[3]),
            DOMAIN_EPS,
        ),
        output_scale=output_scale,
        investment_anchor=investment_anchor,
        current_account_anchor=current_account_anchor,
    )
    p = StructuralParameters.create(
        dtype,
        alpha=parameters.alpha,
        gamma=parameters.gamma,
        delta=parameters.delta,
        preference_rate=parameters.r,
        investment_eta=parameters.eta,
        interest_rate_mean=parameters.interest_rate,
        chi=parameters.chi,
        kappa_z=kappa_z,
        sigma_z=sigma_z,
        log_z_bar=math.log(z_bar),
        kappa_r=kappa_r,
        sigma_r=sigma_r,
        corr_zr=corr_zr,
        output_scale=output_scale,
    )
    solver = SDE_SOLVERS[sde_solver]
    n_train_steps = data_set.train_points - 1
    n_test_steps = data_set.test_points - 1
    n_tvc_steps = data_set.tvc_points - 1
    train_dt = data_set.train_T / n_train_steps
    test_dt = data_set.test_T / n_test_steps
    tvc_dt = data_set.tvc_T / n_tvc_steps

    if stop_set.shock_continuation:
        continuation_scales = (
            np.ones(1)
            if stop_set.continuation_epochs == 1
            else np.geomspace(
                stop_set.initial_shock_scale,
                1.0,
                stop_set.continuation_epochs,
            )
        )
        shock_scales = jnp.asarray(
            np.concatenate(
                [
                    continuation_scales,
                    np.ones(stop_set.num_epochs - stop_set.continuation_epochs),
                ]
            ),
            dtype=dtype,
        )
    else:
        shock_scales = jnp.ones(stop_set.num_epochs, dtype=dtype)
    initial_scale_mask = jnp.asarray(
        [shock_scales[0], 1.0, 1.0, shock_scales[0]], dtype=dtype
    )

    data_key = jax.random.key(ic_set.data_seed)
    train_ic_key, test_ic_key = jax.random.split(data_key)
    train_initial_states = standardized_initial_states(
        train_ic_key,
        initial_center,
        train_scales * initial_scale_mask,
        ic_set.num_train_trajectories,
    )
    test_initial_states = standardized_initial_states(
        test_ic_key,
        initial_center,
        test_scales,
        ic_set.num_test_trajectories,
    )

    fit_timer = Stopwatch()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    initial_model = SequentialPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key))
    graphdef, theta_0, nondiff = nnx.split(initial_model, nnx.Param, ...)
    # fold_in(train_ic_key, 1) is the training noise stream; the kernel
    # consumer regenerates this exact stream to rebuild the frozen design.
    train_noise_key = jax.random.fold_in(train_ic_key, 1)
    test_noise_key = jax.random.fold_in(test_ic_key, 1)
    tvc_noise_key = jax.random.fold_in(test_ic_key, 3)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        nondiff=nondiff,
        initial_states=train_initial_states,
        initial_center=initial_center,
        initial_scales=train_scales,
        noise=sample_trajectory_noise(
            solver,
            train_noise_key,
            ic_set.num_train_trajectories,
            n_train_steps,
            train_dt,
            dtype,
            4,
        ),
        epoch_key=jax.random.fold_in(train_ic_key, 2),
        restart_key=jax.random.fold_in(train_ic_key, 4),
        epoch=jnp.asarray(0, dtype=jnp.int32),
        shock_scales=shock_scales,
        shock_scale=shock_scales[0],
        train_loss_threshold=jnp.asarray(stop_set.train_loss_threshold, dtype=dtype),
        epoch_steps=jnp.asarray(stop_set.max_steps, dtype=jnp.int32),
        total_steps=jnp.asarray(
            stop_set.num_epochs * stop_set.max_steps, dtype=jnp.int32
        ),
        solver=solver,
        data_set=data_set,
    )
    # <problem>-<run> key: consumers glob *_policy_<problem>-*.npz; the run
    # hash covers training minutiae only.
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
    run_config = {
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "sde_solver": sde_solver,
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
        "nn_set": asdict(nn_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
    }
    problem_key = cache_key(problem_config)
    run_key = cache_key(run_config)
    policy_key = f"{problem_key}-{run_key}"
    cache_path = (
        None
        if policy_cache is None
        else pathlib.Path(policy_cache)
        / f"stochastic_open_economy_growth_nn_recursive_policy_{policy_key}.npz"
    )
    cache_hit = cache_path is not None and cache_path.exists()
    if cache_hit:
        policy, cached_meta = load_model(cache_path, make_sequential_policy)
        theta_final = nnx.split(policy, nnx.Param, ...)[1]
        # The saved policy is the end of the ladder: diagnostics must run at
        # full shock scale, never the fresh args' fractional scale.
        solve_args = residual_args.replace(
            shock_scale=jnp.asarray(1.0, dtype=dtype),
            epoch=jnp.asarray(stop_set.num_epochs - 1, dtype=jnp.int32),
        )
        steps = 0
        epochs_run = 0
        attempt = 0
        stopping_reason = "cached"
        final_damping = float("nan")
        solver_accepted = bool(cached_meta["solver_accepted"])
        solve_time = 0.0
        total_fit_time = fit_timer.elapsed()
        setup_time = total_fit_time
        train_residual = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(train_residual * train_residual))
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
        callback = EpochCallback(
            num_train_trajectories=ic_set.num_train_trajectories,
            n_steps=n_train_steps,
            dt=train_dt,
            epoch_early_stopping=stop_set.epoch_early_stopping,
            resample_initial_states=any(value > 0.0 for value in train_variances),
        )
        # Per-step atol is sound only for a never-changing batch; epochs are
        # judged at the boundaries and by the acceptance test.
        atol = (
            math.sqrt(stop_set.train_loss_threshold)
            if stop_set.num_epochs == 1
            else 0.0
        )
        solve_timer = Stopwatch()
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
                draw=DrawSequentialPolicy(nn_set, dtype),
                accept=accept_training,
            ),
            max_steps_is_success=False,
        )
        solve_time = solve_timer.elapsed(result)
        total_fit_time = fit_timer.elapsed()
        setup_time = total_fit_time - solve_time
        theta_final, solve_args = result.x, result.args
        train_residual = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(train_residual * train_residual))
        steps = int(result.steps)
        epochs_run = int(solve_args.epoch) + 1
        attempt = int(result.multi_start.attempt) + 1
        stopping_reason = STATUS_TO_REASON.get(int(result.status), "unknown")
        final_damping = float(result.lm_state.damping)
        solver_accepted = bool(result.multi_start.accepted)
        if verbose:
            print(
                f"attempt {attempt}: {stopping_reason} after {steps} steps "
                f"(epoch {epochs_run}/{stop_set.num_epochs}, shock scale "
                f"{float(solve_args.shock_scale):.3f}), "
                f"train_loss={train_loss:.3e} ({solve_time:.2f}s)"
            )
        policy = nnx.merge(graphdef, theta_final, nondiff)

    # Training diagnostics use the batch the solution was returned on.
    train_stage_parameters = p.replace(
        sigma_z=p.sigma_z * solve_args.shock_scale,
        sigma_r=p.sigma_r * solve_args.shock_scale,
    )
    train_solution = simulate_train(
        solver,
        policy,
        solve_args.initial_states,
        solve_args.noise,
        train_stage_parameters,
        data_set,
    )
    train_values = trajectory_values(policy, train_solution.xs, train_stage_parameters)
    test_noise = sample_trajectory_noise(
        solver,
        test_noise_key,
        ic_set.num_test_trajectories,
        n_test_steps,
        test_dt,
        dtype,
        4,
    )
    test_solution = simulate(
        solver,
        policy,
        test_initial_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    jax.block_until_ready((train_solution, train_values, test_solution))
    test_states = jnp.where(test_solution.ok[:, None, None], test_solution.xs, jnp.nan)
    test_values = trajectory_values(policy, test_states, p)

    tvc_initial_states = jnp.broadcast_to(
        initial_center,
        (ic_set.num_tvc_trajectories, initial_center.size),
    )
    tvc_noise = sample_trajectory_noise(
        solver,
        tvc_noise_key,
        ic_set.num_tvc_trajectories,
        n_tvc_steps,
        tvc_dt,
        dtype,
        4,
    )
    tvc_solution = simulate(
        solver,
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
    jax.block_until_ready((tvc_solution, tvc_values, tvc_statistics))

    train_df = trajectory_frame(train_values, train_solution.ts)
    test_df = trajectory_frame(test_values, test_solution.ts)
    tvc_df = trajectory_frame(tvc_values, tvc_solution.ts)
    expected_tvc_df = transversality_expectation_frame(tvc_values, tvc_solution.ts)
    train_euler = jnp.stack([train_values["c_euler"], train_values["q_euler"]], axis=-1)
    test_euler = jnp.stack([test_values["c_euler"], test_values["q_euler"]], axis=-1)
    test_mean_abs = jnp.mean(jnp.abs(test_euler), axis=(0, 1))
    test_max_abs = jnp.max(jnp.abs(test_euler), axis=(0, 1))
    test_p95_abs = jnp.quantile(jnp.abs(test_euler), 0.95, axis=(0, 1))
    test_diffusion = jnp.stack(
        [
            test_values["c_diffusion_term"],
            test_values["q_diffusion_term"],
        ],
        axis=-1,
    )
    test_diffusion_max_abs = jnp.max(jnp.abs(test_diffusion), axis=(0, 1))
    train_path_ok = bool(
        jnp.all(train_solution.ok)
        and jnp.all(jnp.isfinite(train_solution.xs))
        and jnp.all(jnp.isfinite(train_euler))
        and jnp.min(train_values["k"]) > 0.0
        and jnp.min(train_values["q"]) > 0.0
        and jnp.min(train_values["c"]) > 0.0
        and jnp.min(train_values["x"]) > 0.0
    )
    test_path_ok = bool(
        jnp.all(test_solution.ok)
        and jnp.all(jnp.isfinite(test_states))
        and jnp.all(jnp.isfinite(test_euler))
        and jnp.min(test_values["k"]) > 0.0
        and jnp.min(test_values["q"]) > 0.0
        and jnp.min(test_values["c"]) > 0.0
        and jnp.min(test_values["x"]) > 0.0
    )
    tvc_feasible_observations = (
        jnp.isfinite(tvc_values["discounted_marginal_utility"])
        & jnp.isfinite(tvc_values["absolute_asset_tvc"])
        & jnp.isfinite(tvc_values["capital_tvc"])
        & (tvc_values["k"] > 0.0)
        & (tvc_values["q"] > 0.0)
        & (tvc_values["c"] > 0.0)
        & (tvc_values["x"] > 0.0)
    )
    tvc_paths_ok = bool(
        jnp.all(tvc_solution.ok)
        and jnp.all(jnp.isfinite(tvc_states))
        and jnp.all(tvc_feasible_observations)
    )

    # Paired common-noise simulations from shifted NFA positions exercise the
    # asset axis even at zero IC spread.
    paired_states = jnp.broadcast_to(
        initial_center,
        (ic_set.num_test_trajectories, initial_center.size),
    )
    asset_shift = asset_shift_scale * output_scale
    low_states = paired_states.at[:, 2].set((a_0 - asset_shift) / output_scale)
    high_states = paired_states.at[:, 2].set((a_0 + asset_shift) / output_scale)
    low_solution = simulate(
        solver,
        policy,
        low_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    high_solution = simulate(
        solver,
        policy,
        high_states,
        test_noise,
        data_set.test_T,
        n_test_steps,
        p,
    )
    low_solution_states = jnp.where(
        low_solution.ok[:, None, None], low_solution.xs, jnp.nan
    )
    high_solution_states = jnp.where(
        high_solution.ok[:, None, None], high_solution.xs, jnp.nan
    )
    low_values = trajectory_values(policy, low_solution_states, p)
    high_values = trajectory_values(policy, high_solution_states, p)
    jax.block_until_ready((low_values, high_values))

    low_euler = jnp.stack([low_values["c_euler"], low_values["q_euler"]], axis=-1)
    high_euler = jnp.stack([high_values["c_euler"], high_values["q_euler"]], axis=-1)
    hysteresis_euler = jnp.concatenate([low_euler, high_euler], axis=0)
    hysteresis_euler_mean_abs = float(jnp.mean(jnp.abs(hysteresis_euler)))
    hysteresis_euler_max_abs = float(jnp.max(jnp.abs(hysteresis_euler)))
    hysteresis_paths_ok = bool(
        jnp.all(low_solution.ok)
        and jnp.all(high_solution.ok)
        and jnp.all(jnp.isfinite(low_solution_states))
        and jnp.all(jnp.isfinite(high_solution_states))
        and jnp.all(jnp.isfinite(low_euler))
        and jnp.all(jnp.isfinite(high_euler))
        and jnp.min(low_values["k"]) > 0.0
        and jnp.min(high_values["k"]) > 0.0
        and jnp.min(low_values["q"]) > 0.0
        and jnp.min(high_values["q"]) > 0.0
        and jnp.min(low_values["c"]) > 0.0
        and jnp.min(high_values["c"]) > 0.0
        and jnp.min(low_values["x"]) > 0.0
        and jnp.min(high_values["x"]) > 0.0
    )
    hysteresis_terminal_asset_difference = float(
        jnp.mean(high_values["a"][:, -1] - low_values["a"][:, -1])
    )

    test_terminal_capital_mean = float(jnp.mean(test_values["k"][:, -1]))

    rejection_reasons = []
    # A cached policy trained on other batches: only its saved solver
    # acceptance applies, never the fresh-batch loss.
    if not solver_accepted or (
        not cache_hit and not train_loss <= stop_set.train_loss_threshold
    ):
        rejection_reasons.append("train_loss")
    if not train_path_ok:
        rejection_reasons.append("train_path")
    if not test_path_ok:
        rejection_reasons.append("test_path")
    if bool(jnp.any(test_mean_abs > tolerance_set.test_euler_mean_abs)):
        rejection_reasons.append("test_euler_mean")
    if bool(jnp.any(test_max_abs > tolerance_set.test_euler_max)):
        rejection_reasons.append("test_euler_max")
    if bool(jnp.any(test_diffusion_max_abs > tolerance_set.test_diffusion_max_abs)):
        rejection_reasons.append("test_diffusion_max")
    if not hysteresis_paths_ok:
        rejection_reasons.append("hysteresis_path")
    if hysteresis_euler_mean_abs > tolerance_set.test_euler_mean_abs:
        rejection_reasons.append("hysteresis_euler_mean")
    if hysteresis_euler_max_abs > tolerance_set.test_euler_max:
        rejection_reasons.append("hysteresis_euler_max")
    acceptance = acceptance_fields(rejection_reasons)
    solution_accepted = acceptance["solution_accepted"]

    theta_flat, _ = ravel_pytree(theta_final)
    observed_assets = np.asarray(test_values["a"]).reshape(-1)
    observed_rates = np.asarray(test_values["interest_rate"]).reshape(-1)
    test_asset_change = test_values["a"][:, -1] - test_values["a"][:, 0]
    # Near one, the squash saturates and the heads lose asset sensitivity.
    test_normalized_asset = (
        test_states[..., 2] - nn_set.asset_center
    ) / nn_set.asset_scale
    test_asset_squash_saturation = float(
        jnp.nanmax(jnp.abs(jnp.tanh(test_normalized_asset / 2.0)))
    )
    results = {
        **asdict(parameters),
        "variant": "stochastic_recursive_open_economy_nn",
        "state_coordinate": "(log z, log k, a/output_scale, rate_gap)",
        "policy": "investment_current_account",
        "initialization": "replacement_investment_zero_current_account",
        "k_0": k_0,
        "a_0": a_0,
        "z_0": z_0,
        "rate_gap_0": rate_gap_0,
        "interest_rate_mean": parameters.interest_rate,
        "kappa_z": kappa_z,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "kappa_r": kappa_r,
        "sigma_r": sigma_r,
        "corr_zr": corr_zr,
        "z_ergodic_sd": math.sqrt(z_ergodic_variance),
        "rate_ergodic_sd": math.sqrt(rate_ergodic_variance),
        "seed": seed,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_residual.dtype),
        "sde_solver": sde_solver,
        "train_loss": train_loss,
        "train_euler_mean_abs": float(jnp.mean(jnp.abs(train_euler))),
        "train_euler_max_abs": float(jnp.max(jnp.abs(train_euler))),
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_euler))),
        # NaN when rejected: the HPO sweeps sort on this.
        "test_euler_mean_abs_success": (
            float(jnp.mean(jnp.abs(test_euler))) if solution_accepted else np.nan
        ),
        "test_c_euler_mean_abs": float(test_mean_abs[0]),
        "test_q_euler_mean_abs": float(test_mean_abs[1]),
        "test_c_euler_max_abs": float(test_max_abs[0]),
        "test_q_euler_max_abs": float(test_max_abs[1]),
        "test_c_euler_p95_abs": float(test_p95_abs[0]),
        "test_q_euler_p95_abs": float(test_p95_abs[1]),
        "test_c_diffusion_max_abs": float(test_diffusion_max_abs[0]),
        "test_q_diffusion_max_abs": float(test_diffusion_max_abs[1]),
        "min_test_consumption": float(jnp.min(test_values["c"])),
        "min_test_interest_rate": float(np.min(observed_rates)),
        "test_negative_rate_fraction": float(np.mean(observed_rates < 0.0)),
        "test_asset_p025": float(np.quantile(observed_assets, 0.025)),
        "test_asset_p50": float(np.quantile(observed_assets, 0.5)),
        "test_asset_p975": float(np.quantile(observed_assets, 0.975)),
        "test_asset_squash_saturation": test_asset_squash_saturation,
        "test_terminal_asset_mean": float(jnp.mean(test_values["a"][:, -1])),
        "test_terminal_asset_sd": float(jnp.std(test_values["a"][:, -1])),
        "test_terminal_capital_mean": test_terminal_capital_mean,
        "test_terminal_capital_sd": float(jnp.std(test_values["k"][:, -1])),
        "test_min_capital": float(jnp.min(test_values["k"])),
        "test_max_capital": float(jnp.max(test_values["k"])),
        "test_asset_change_mean": float(jnp.mean(test_asset_change)),
        "test_asset_change_sd": float(jnp.std(test_asset_change)),
        "test_terminal_current_account_mean": float(
            jnp.mean(test_values["current_account"][:, -1])
        ),
        "tvc_T": data_set.tvc_T,
        "tvc_points": data_set.tvc_points,
        "num_tvc_trajectories": ic_set.num_tvc_trajectories,
        "tvc_paths_ok": tvc_paths_ok,
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
        "asset_shift": asset_shift,
        "hysteresis_terminal_asset_mean_difference": (
            hysteresis_terminal_asset_difference
        ),
        "hysteresis_asset_persistence_ratio": (
            hysteresis_terminal_asset_difference / (2.0 * asset_shift)
        ),
        "hysteresis_paths_ok": hysteresis_paths_ok,
        "hysteresis_euler_mean_abs": hysteresis_euler_mean_abs,
        "hysteresis_euler_max_abs": hysteresis_euler_max_abs,
        "steps": steps,
        "num_epochs": stop_set.num_epochs,
        "epochs_run": epochs_run,
        "shock_continuation_scales": tuple(
            float(value) for value in np.asarray(shock_scales)
        ),
        "final_training_shock_scale": float(solve_args.shock_scale),
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "final_damping": final_damping,
        "policy_cache_key": policy_key,
        "policy_cache_hit": cache_hit,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "solve_time_success": (
            solve_time if solver_accepted and not cache_hit else np.nan
        ),
        "total_params": int(theta_flat.size),
        "residual_count": int(train_residual.size),
        "parameter_to_residual_ratio": float(theta_flat.size / train_residual.size),
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        "train_log_z_variance": train_variances[0],
        "train_log_k_variance": train_variances[1],
        "train_asset_variance": train_variances[2],
        "train_rate_variance": train_variances[3],
        "train_points": data_set.train_points,
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
        "train_path_ok": train_path_ok,
        "test_path_ok": test_path_ok,
        "solver_accepted": solver_accepted,
        "solver_converged": solver_accepted,
        **acceptance,
        "lm_linear_system": "default_auto_dual_gram_when_overparameterized",
        "parameter_derivative": (
            "minimum_euclidean_norm_svd_conditional_on_carried_training_batch"
        ),
    }

    hyperparameters = {
        "parameters": asdict(parameters),
        "shocks": {
            "kappa_z": kappa_z,
            "sigma_z": sigma_z,
            "z_bar": z_bar,
            "kappa_r": kappa_r,
            "sigma_r": sigma_r,
            "corr_zr": corr_zr,
        },
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "nn_set": asdict(nn_set),
        "opt_set": asdict(opt_set),
        "stop_set": asdict(stop_set),
        "tolerance_set": asdict(tolerance_set),
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

    structural = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "preference_rate": parameters.r,
        "investment_eta": parameters.eta,
        "interest_rate_mean": parameters.interest_rate,
        "chi": parameters.chi,
        "kappa_z": kappa_z,
        "sigma_z": sigma_z,
        "log_z_bar": math.log(z_bar),
        "kappa_r": kappa_r,
        "sigma_r": sigma_r,
        "corr_zr": corr_zr,
        "output_scale": output_scale,
    }
    initial_state = {"k_0": k_0, "a_0": a_0, "z_0": z_0, "rate_gap_0": rate_gap_0}
    policy_extra = {
        "model": "stochastic_open_economy_growth_nn_recursive",
        "problem_key": problem_key,
        "run_key": run_key,
        "cache_key": policy_key,
        "config": {"problem": problem_config, "run": run_config},
        "solver_accepted": solver_accepted,
        "solution_accepted": solution_accepted,
        "train_loss": train_loss,
        "test_euler_mean_abs": results["test_euler_mean_abs"],
    }
    policy_meta = {
        "settings": asdict(nn_set),
        "structural": structural,
        "initial_state": initial_state,
        **policy_extra,
    }
    if cache_path is not None and not cache_hit:
        save_model(cache_path, policy, policy_meta)
    if save_policy is not None:
        save_model(save_policy, policy, policy_meta)
    return results, {
        "model": policy,
        "parameters": p,
        "train_initial_states": solve_args.initial_states,
        "test_initial_states": test_initial_states,
        "train_df": train_df,
        "test_df": test_df,
        "tvc_df": tvc_df,
        "expected_tvc_df": expected_tvc_df,
        "asset_low_df": trajectory_frame(low_values, low_solution.ts),
        "asset_high_df": trajectory_frame(high_values, high_solution.ts),
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(stochastic_open_economy_growth_nn_recursive)
