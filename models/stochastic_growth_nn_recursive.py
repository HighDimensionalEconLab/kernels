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

from models.stochastic_growth_shared import (
    STOCHASTIC_GROWTH_CONSTANTS as CONSTANTS,
)
from models.stochastic_growth_shared import (
    DataSettings,
    DrawInvestmentPolicy,
    InitialConditionSettings,
    InvestmentPolicyNN,
    InvestmentPolicySettings,
    StructuralParameters,
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
    Stopwatch,
    acceptance_fields,
    cache_key,
    load_model,
    long_df,
    sample_trajectory_noise,
    save_model,
    standardized_initial_states,
)


@dataclass(frozen=True)
class ToleranceSettings:
    # max_steps is the LM-update budget per epoch; the noise realization (and
    # the training ICs when their variance is positive) is redrawn at each
    # interior epoch boundary. An attempt can only succeed at those interior
    # boundaries, so a failed fresh-batch gate costs the full epoch budget.
    max_steps: int = 125
    # The gate must sit well above the reachable fresh-batch floor (~1e-15):
    # tighter gates make acceptance depend on the noise draw and burn whole
    # attempts, while test-Euler quality is flat below ~5e-13.
    train_loss_threshold: float = 5e-13
    log_tvc_threshold: float = 0.0
    num_epochs: int = 6
    epoch_early_stopping: bool = True


# Training ICs, the noise realization, and keys live in args (the epoch
# callback may rewrite args only), so the implicit derivative is conditional
# on the carried batch of initial conditions and shocks.
@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    initial_log_states: jax.Array
    noise: object
    epoch_key: jax.Array
    restart_key: jax.Array
    epoch: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    sde: object = struct.field(pytree_node=False)
    data_set: DataSettings = struct.field(pytree_node=False)


# At each interior epoch boundary: redraw the shock realization (and the
# training ICs when configured), stop on a nonfinite or converged fresh-batch
# loss, else continue with damping reset.
@dataclass(frozen=True)
class EpochCallback:
    num_train_trajectories: int
    log_z_0: float
    log_k_0: float
    log_z_scale: float
    log_k_scale: float
    epoch_early_stopping: bool
    resample_initial_states: bool
    n_steps: int
    dt: float

    def next_epoch(self, ctx):
        epoch_key, draw_key, noise_key = jax.random.split(ctx.args.epoch_key, 3)
        dtype = ctx.args.initial_log_states.dtype
        if self.resample_initial_states:
            initial_log_states = standardized_initial_states(
                draw_key,
                jnp.asarray([self.log_z_0, self.log_k_0], dtype),
                jnp.asarray([self.log_z_scale, self.log_k_scale], dtype),
                self.num_train_trajectories,
            )
        else:
            initial_log_states = ctx.args.initial_log_states
        fresh = ctx.args.replace(
            initial_log_states=initial_log_states,
            noise=sample_trajectory_noise(
                ctx.args.sde,
                noise_key,
                self.num_train_trajectories,
                self.n_steps,
                self.dt,
                dtype,
                2,
            ),
            epoch_key=epoch_key,
            epoch=ctx.args.epoch + 1,
        )
        r = residual_fn(ctx.x, fresh, ctx.p)
        loss = jnp.sum(r * r)
        nonfinite = ~jnp.isfinite(loss)
        if self.epoch_early_stopping:
            converged = loss < fresh.train_loss_threshold
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
    solution = simulate_train(
        args.sde, policy, args.initial_log_states, args.noise, p, args.data_set
    )
    log_states = solution.xs
    euler = euler_residuals(policy, log_states.reshape(-1, 2), p).reshape(
        log_states.shape[:2]
    )
    # Residual vector scaled by 1/sqrt(count); the caller decides how to use
    # it.
    count = jnp.asarray(euler.size, dtype=euler.dtype)
    # A nonfinite trajectory did not cover the requested economic horizon. Make
    # that a loud optimizer failure instead of fitting a prefix.
    return jnp.where(
        solution.ok[:, None],
        euler / jnp.sqrt(count),
        jnp.full_like(euler, jnp.inf),
    ).reshape(-1)


# The deterministic model plus its OU log-TFP.
def stochastic_growth_nn_recursive(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    eta: float = CONSTANTS.eta,
    sigma_z: float = CONSTANTS.sigma_z,
    z_bar: float = 1.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    sde_solver: str = "sra1",
    data_set: DataSettings = DataSettings(),
    # The initial log z spread matches the ergodic variance (sigma_z^2/(2 eta)).
    ic_set: InitialConditionSettings = InitialConditionSettings(
        num_train_trajectories=16,
        train_log_z_variance=CONSTANTS.z_ergodic_variance,
        train_log_k_variance=0.01,
    ),
    nn_set: InvestmentPolicySettings = InvestmentPolicySettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=3.26e-5,
        damping_decrease=0.482,
        damping_increase=2.71,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.541,
        num_attempts=4,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    # Directory for content-addressed policy artifacts (e.g. ".results"): an
    # existing artifact for this exact configuration is loaded instead of
    # training, and a fresh solve is saved there. save_policy writes one
    # explicit extra path.
    policy_cache: str | None = None,
    save_policy: str | None = None,
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
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        eta=eta,
        sigma_z=sigma_z,
        log_z_bar=math.log(z_bar),
        z_0=train_initial_conditions[:, 0],
        k_0=train_initial_conditions[:, 1],
    )

    fit_timer = Stopwatch()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    graphdef, _, nondiff = nnx.split(
        InvestmentPolicyNN(nn_set, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(0)),
        nnx.Param,
        ...,
    )
    # The initial training draws consume train_ic_key itself; the epoch,
    # restart, and noise streams chain from independent fold-ins of the same
    # key, and the test noise stream chains from test_ic_key.
    train_noise_key = jax.random.fold_in(train_ic_key, 3)
    tvc_noise_key = jax.random.fold_in(train_ic_key, 4)
    test_noise_key = jax.random.fold_in(test_ic_key, 1)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        nondiff=nondiff,
        initial_log_states=jnp.log(train_initial_conditions),
        noise=sample_trajectory_noise(
            sde,
            train_noise_key,
            ic_set.num_train_trajectories,
            n_train_steps,
            train_dt,
            dtype,
            2,
        ),
        epoch_key=jax.random.fold_in(train_ic_key, 1),
        restart_key=jax.random.fold_in(train_ic_key, 2),
        epoch=jnp.asarray(0, dtype=jnp.int32),
        train_loss_threshold=jnp.asarray(
            tolerance_set.train_loss_threshold, dtype=dtype
        ),
        epoch_steps=jnp.asarray(tolerance_set.max_steps, dtype=jnp.int32),
        total_steps=jnp.asarray(
            tolerance_set.num_epochs * tolerance_set.max_steps, dtype=jnp.int32
        ),
        sde=sde,
        data_set=data_set,
    )
    # Two-part artifact key <problem>-<run>: the problem hash covers the deep
    # structural parameters, dtype, and the training initial-condition design
    # shared by any solver of the same problem, so a consumer can glob
    # *_policy_<problem>-*.npz for a reference solution; the run hash covers
    # the training minutiae. Test/benchmark settings enter neither, so
    # changing them still reuses a cached policy.
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
    run_config = {
        "seed": seed,
        "data_seed": ic_set.data_seed,
        "sde_solver": sde_solver,
        "train_points": data_set.train_points,
        "train_T": data_set.train_T,
        "nn_set": asdict(nn_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
    }
    problem_key = cache_key(problem_config)
    run_key = cache_key(run_config)
    policy_key = f"{problem_key}-{run_key}"
    cache_path = (
        None
        if policy_cache is None
        else pathlib.Path(policy_cache)
        / f"stochastic_growth_nn_recursive_policy_{policy_key}.npz"
    )
    cache_hit = cache_path is not None and cache_path.exists()
    if cache_hit:
        policy, cached_meta = load_model(cache_path, make_investment_policy)
        theta_final, solve_args = nnx.split(policy, nnx.Param, ...)[1], residual_args
        steps = 0
        epochs_run = 0
        attempt = 0
        stopping_reason = "cached"
        final_damping = float("nan")
        solver_accepted = bool(cached_meta["solution_accepted"])
        solve_time = 0.0
        total_fit_time = fit_timer.elapsed()
        setup_time = total_fit_time
        r = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(r * r))
        if verbose:
            print(f"loaded {cache_path}, train_loss={train_loss:.3e}")
    else:
        solver = LevenbergMarquardt(
            residual_fn,
            init_damping=lm_set.init_damping,
            damping_decrease=lm_set.damping_decrease,
            damping_increase=lm_set.damping_increase,
            geodesic_acceleration=lm_set.geodesic_acceleration,
            geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
        )
        _, theta_0, _ = nnx.split(
            InvestmentPolicyNN(
                nn_set, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(init_key)
            ),
            nnx.Param,
            ...,
        )
        callback = EpochCallback(
            num_train_trajectories=ic_set.num_train_trajectories,
            log_z_0=math.log(z_0),
            log_k_0=math.log(k_0),
            log_z_scale=math.sqrt(ic_set.train_log_z_variance),
            log_k_scale=math.sqrt(ic_set.train_log_k_variance),
            epoch_early_stopping=tolerance_set.epoch_early_stopping,
            resample_initial_states=(
                ic_set.train_log_z_variance > 0.0 or ic_set.train_log_k_variance > 0.0
            ),
            n_steps=n_train_steps,
            dt=train_dt,
        )
        draw = DrawInvestmentPolicy(nn_set, k_scale=k_0, dtype=dtype)
        # The per-step tolerance is sound only while the training batch never
        # changes; across epochs, convergence is judged at the boundaries
        # instead.
        atol = (
            math.sqrt(tolerance_set.train_loss_threshold)
            if tolerance_set.num_epochs == 1
            else 0.0
        )
        multi_start = MultiStart(
            key=multi_start_key,
            num_starts=lm_set.num_attempts,
            draw=draw,
        )
        solve_options = dict(
            max_steps=tolerance_set.num_epochs * tolerance_set.max_steps,
            atol=atol,
            callback=callback,
            max_steps_is_success=not tolerance_set.epoch_early_stopping,
        )

        setup_time = fit_timer.elapsed()
        solve_timer = Stopwatch()
        result = solver.solve(
            theta_0,
            residual_args,
            p=p,
            multi_start=multi_start,
            **solve_options,
        )
        solve_time = solve_timer.elapsed(result)
        total_fit_time = fit_timer.elapsed()
        steps = int(result.steps)
        # info.loss can describe the pre-swap batch when an epoch boundary
        # stopped the solve, so the loss is recomputed at the returned
        # (x, args).
        theta_final, solve_args = result.x, result.args
        r = residual_fn(theta_final, solve_args, p)
        train_loss = float(jnp.sum(r * r))
        epochs_run = -(-steps // tolerance_set.max_steps)
        stopping_reason = STATUS_TO_REASON.get(int(result.status), "max_steps")
        attempt = int(result.multi_start.attempt) + 1
        final_damping = float(result.lm_state.damping)
        solver_accepted = bool(result.multi_start.accepted)
        if verbose:
            print(
                f"attempt {attempt}: {stopping_reason} after {steps} steps "
                f"(epoch {epochs_run}/{tolerance_set.num_epochs}), "
                f"train_loss={train_loss:.3e} ({solve_time:.2f}s)"
            )
        policy = nnx.merge(graphdef, theta_final, nondiff)

    # Training diagnostics use the batch (ICs and noise) the solution was
    # returned on.
    train_batch_log_states = solve_args.initial_log_states
    train_batch_initial_conditions = jnp.exp(train_batch_log_states)
    train_solution = simulate_train(
        sde, policy, train_batch_log_states, solve_args.noise, p, data_set
    )
    train_log_states = train_solution.xs
    train_values = trajectory_values(policy, train_log_states, p)
    train_initial = np.asarray(train_batch_initial_conditions)
    train_df = long_df(
        {"t": train_solution.ts, **train_values},
        ("trajectory", "step"),
        scalars={"z_0": train_initial[:, 0], "k_0": train_initial[:, 1]},
    )
    num_simulation_states = int(train_solution.ts.size)

    test_initial_log_states = jnp.log(test_initial_conditions)
    test_noise = sample_trajectory_noise(
        sde,
        test_noise_key,
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

    # Terminal transversality certificate on the training batch.
    train_tvc_noise = sample_trajectory_noise(
        sde,
        tvc_noise_key,
        ic_set.num_train_trajectories,
        data_set.tvc_steps,
        data_set.train_tvc_T / data_set.tvc_steps,
        dtype,
        2,
    )
    train_tvc = log_tvc_residual(
        sde,
        policy,
        train_batch_log_states,
        train_tvc_noise,
        data_set.train_tvc_T,
        data_set.tvc_steps,
        p,
    )
    max_log_tvc = float(jnp.max(train_tvc))
    transversality_ok = bool(max_log_tvc < tolerance_set.log_tvc_threshold)
    max_train_euler = float(train_df["euler_residual"].abs().max())
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    z_ergodic_sd = sigma_z / math.sqrt(2.0 * eta)
    observed_log_z = np.concatenate(
        [
            np.asarray(train_log_states[..., 0]).reshape(-1),
            np.asarray(test_log_states[..., 0]).reshape(-1),
        ]
    )
    observed_log_k = np.concatenate(
        [
            np.asarray(train_log_states[..., 1]).reshape(-1),
            np.asarray(test_log_states[..., 1]).reshape(-1),
        ]
    )
    observed_log_z = observed_log_z[np.isfinite(observed_log_z)]
    observed_log_k = observed_log_k[np.isfinite(observed_log_k)]

    rejection_reasons = []
    if not solver_accepted:
        rejection_reasons.append("train_loss")
    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "z_ergodic_sd": z_ergodic_sd,
        "z_0": z_0,
        "k_0": k_0,
        "test_z_0": ic_draw.test_z_center,
        "test_k_0": ic_draw.test_k_center,
        "seed": seed,
        "data_seed": ic_set.data_seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(r.dtype),
        "sde_solver": sde_solver,
        "train_loss": train_loss,
        "max_train_euler": max_train_euler,
        "max_test_euler": max_test_euler,
        "final_damping": final_damping,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_values["euler_residual"]))),
        "test_trajectory_euler_mean_abs_p50": float(
            jnp.quantile(test_trajectory_euler, 0.5)
        ),
        "test_trajectory_euler_mean_abs_p90": float(
            jnp.quantile(test_trajectory_euler, 0.9)
        ),
        "test_trajectory_euler_mean_abs_max": float(jnp.max(test_trajectory_euler)),
        "max_abs_diffusion_term": float(
            jnp.max(jnp.abs(test_values["diffusion_term"]))
        ),
        "mean_abs_diffusion_term": float(
            jnp.mean(jnp.abs(test_values["diffusion_term"]))
        ),
        "max_log_tvc_residual": max_log_tvc,
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_values["c"])),
        "steps": steps,
        "num_epochs": tolerance_set.num_epochs,
        "epochs_run": epochs_run,
        "attempt": attempt,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "solve_time_success": solve_time if solver_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        **acceptance_fields(rejection_reasons),
        "num_train_trajectories": ic_set.num_train_trajectories,
        "num_test_trajectories": ic_set.num_test_trajectories,
        "num_simulation_states": num_simulation_states,
        "train_path_ok": bool(jnp.all(train_solution.ok)),
        "observed_log_z_min": float(observed_log_z.min()),
        "observed_log_z_max": float(observed_log_z.max()),
        "observed_log_z_p025": float(np.quantile(observed_log_z, 0.025)),
        "observed_log_z_p975": float(np.quantile(observed_log_z, 0.975)),
        "observed_log_k_min": float(observed_log_k.min()),
        "observed_log_k_max": float(observed_log_k.max()),
        "train_log_z_variance": ic_set.train_log_z_variance,
        "train_log_k_variance": ic_set.train_log_k_variance,
        "test_log_z_variance": ic_draw.test_log_z_variance,
        "test_log_k_variance": ic_draw.test_log_k_variance,
        "train_points": data_set.train_points,
        "train_steps": n_train_steps,
        "test_points": data_set.test_points,
        "train_T": data_set.train_T,
        "test_T": test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    hyperparameters = {
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
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
        "policy": policy,
        "parameters": p,
        "train_initial_conditions": train_initial_conditions,
        "test_initial_conditions": test_initial_conditions,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }
    # A cache hit loads the policy from disk instead of solving, so there is
    # no primal solve to describe.
    if not cache_hit:
        extras["solve"] = {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": p,
            "residual_args": residual_args,
            "multi_start": multi_start,
            "options": solve_options,
        }
        extras["result"] = result
    structural = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "log_z_bar": math.log(z_bar),
    }
    policy_extra = {
        "model": "stochastic_growth_nn_recursive",
        "problem_key": problem_key,
        "run_key": run_key,
        "cache_key": policy_key,
        "config": {"problem": problem_config, "run": run_config},
        "solution_accepted": solver_accepted,
        "train_loss": train_loss,
        "test_euler_mean_abs": results["test_euler_mean_abs"],
    }
    policy_meta = {"settings": asdict(nn_set), "structural": structural, **policy_extra}
    if cache_path is not None and not cache_hit:
        save_model(cache_path, policy, policy_meta)
    if save_policy is not None:
        save_model(save_policy, policy, policy_meta)
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(stochastic_growth_nn_recursive)
