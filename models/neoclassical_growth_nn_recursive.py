from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import DrawNNXModule, LevenbergMarquardt, LMAction, LMStatus, MultiStart
from tinydiffeq import RK4, ConstantStepSize, IController, SaveAt, Tsit5, solve_ode

from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    InitialConditionSettings,
    StructuralParameters,
    euler_residuals,
    growth_initial_conditions,
    log_state_drift,
    log_tvc,
    simulate,
    trajectory_values,
)
from models.neural_networks import (
    ACTIVATIONS,
    PolicySettings,
    StateMLP,
    unit_softplus,
)
from models.utilities import (
    STATUS_TO_REASON,
    OptimizerSettings,
    ResetEpochStream,
    Stopwatch,
    acceptance_fields,
    long_df,
    standardized_initial_states,
)


@dataclass(frozen=True)
class DataSettings:
    # Training residuals on the integrator's own states: n points from
    # n - 1 steps including the initial condition, nothing interpolated.
    # "rk4" spends the budget on a fixed grid; "tsit5" spends it adaptively
    # and enforces residuals on the accepted steps only. rtol/atol/initial_dt
    # configure the adaptive controller and are ignored under rk4; test/tvc
    # simulations stay fixed-step.
    train_points: int = 61
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = 60.0
    train_tvc_T: float = CONSTANTS["transversality_T"]
    test_T: float = CONSTANTS["test_T"]
    ode_solver: str = "rk4"
    rtol: float = 1e-6
    atol: float = 1e-8
    initial_dt: float | None = None


# Ex-post acceptance: gate_train requires the training loss on the returned
# batch under train_loss_threshold, gate_test the held-out Euler errors under
# the bars. The optimizer stops on its own criteria (atol, or the epoch
# callback's fresh-batch loss test) and its status is reported, never gated.
@dataclass(frozen=True)
class ToleranceSettings:
    # max_steps is the LM-update budget per epoch; training ICs are redrawn at
    # each interior epoch boundary, so num_epochs = 1 never resamples.
    max_steps: int = 125
    train_loss_threshold: float = 2.15e-14
    num_epochs: int = 1
    epoch_early_stopping: bool = True
    gate_train: bool = True
    gate_test: bool = True
    test_euler_mean_abs: float = 3e-6
    test_euler_max: float = 9e-5


# Training ICs and keys live in args (the epoch callback may rewrite args
# only), so the implicit derivative is conditional on the carried batch.
@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    initial_log_states: jax.Array
    epoch_key: jax.Array
    restart_key: jax.Array
    epoch: jax.Array
    train_loss_threshold: jax.Array
    epoch_steps: jax.Array
    total_steps: jax.Array
    data_set: DataSettings = struct.field(pytree_node=False)


# At each interior epoch boundary: redraw the training ICs, stop on a
# nonfinite fresh-batch loss or (with epoch_early_stopping) a fresh-batch
# loss under train_loss_threshold, else continue with damping reset.
@dataclass(frozen=True)
class EpochCallback:
    num_train_trajectories: int
    log_z_0: float
    log_k_0: float
    log_z_scale: float
    log_k_scale: float
    epoch_early_stopping: bool

    def next_epoch(self, ctx):
        epoch_key, draw_key = jax.random.split(ctx.args.epoch_key)
        dtype = ctx.args.initial_log_states.dtype
        fresh = ctx.args.replace(
            initial_log_states=standardized_initial_states(
                draw_key,
                jnp.asarray([self.log_z_0, self.log_k_0], dtype),
                jnp.asarray([self.log_z_scale, self.log_k_scale], dtype),
                self.num_train_trajectories,
            ),
            epoch_key=epoch_key,
            epoch=ctx.args.epoch + 1,
        )
        loss, train_ok = train_accepted(ctx.x, fresh, ctx.p)
        nonfinite = ~jnp.isfinite(loss)
        converged = train_ok if self.epoch_early_stopping else jnp.asarray(False)
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


# The training criterion on the batch carried in args: (loss, train_ok) with
# train_ok a finite loss under train_loss_threshold. A NaN loss fails the
# comparison, so a diverged batch never passes.
def train_accepted(theta, args, p):
    residual = residual_fn(theta, args, p)
    loss = jnp.sum(residual * residual)
    return loss, jnp.isfinite(loss) & (loss < args.train_loss_threshold)


# Training simulation on the integrator's own states.
@partial(jax.jit, static_argnames=("data_set",))
def simulate_train(policy, initial_log_states, p, data_set):
    budget = data_set.train_points - 1
    if data_set.ode_solver == "rk4":
        solver = RK4()
        controller = ConstantStepSize()
        dt_0 = data_set.train_T / budget
    else:
        solver = Tsit5()
        controller = IController(rtol=data_set.rtol, atol=data_set.atol)
        dt_0 = (
            data_set.train_T / budget
            if data_set.initial_dt is None
            else data_set.initial_dt
        )

    def solve_one(initial_log_state):
        return solve_ode(
            lambda log_state: log_state_drift(policy, log_state, p),
            solver,
            0.0,
            data_set.train_T,
            initial_log_state,
            dt_0=dt_0,
            controller=controller,
            max_steps=budget,
            save_at=SaveAt(steps=True),
        )

    return jax.vmap(solve_one)(initial_log_states)


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.nondiff)
    solution = simulate_train(policy, args.initial_log_states, p, args.data_set)
    log_states = solution.xs
    euler = euler_residuals(policy, log_states.reshape(-1, 2), p).reshape(
        log_states.shape[:2]
    )
    # Residual vector scaled by 1/sqrt(count); the caller decides how to use
    # it. An adaptive simulation only fills the accepted prefix of the step
    # buffer, so both the residuals and the count are restricted to that
    # prefix.
    if args.data_set.ode_solver == "rk4":
        count = jnp.asarray(euler.size, dtype=euler.dtype)
        normalized = euler / jnp.sqrt(count)
    else:
        active = solution.accepted
        count = jnp.maximum(jnp.sum(active), 1).astype(euler.dtype)
        normalized = jnp.where(active, euler, 0.0) / jnp.sqrt(count)
    # A nonfinite trajectory did not cover the requested economic horizon. Make
    # that a loud optimizer failure instead of fitting a prefix.
    return jnp.where(
        solution.ok[:, None],
        normalized,
        jnp.full_like(euler, jnp.inf),
    ).reshape(-1)


# Deterministic recursive neoclassical growth: a general StateMLP on
# (log z, log k), the policy form of the stochastic driver
# (stochastic_growth_nn_recursive), with the structural constants only in
# the output anchor delta * k_0. No benchmark or steady-state information
# enters the residual or initialization; the closed form is ex-post
# diagnostics only.
def neoclassical_growth_nn_recursive(
    a: float = CONSTANTS["a"],
    delta: float = CONSTANTS["delta"],
    rho_hat: float = CONSTANTS["rho_hat"],
    sigma_crra: float = CONSTANTS["sigma_crra"],
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    nn_set: PolicySettings = PolicySettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=4.74e-4,
        damping_decrease=0.511,
        damping_increase=2.82,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.892,
        num_attempts=2,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    n_test_steps = data_set.test_points - 1
    test_T = data_set.test_T
    ic_draw = growth_initial_conditions(z_0, k_0, ic_set, dtype)
    train_ic_key = ic_draw.train_ic_key
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

    fit_timer = Stopwatch()
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    # Unit input scales: log z is pinned at g = 0 and drifts O(0.1) per decade
    # otherwise, and a network that insensitive to z at the draw trains best.
    policy_spec = dict(
        width=nn_set.width,
        depth=nn_set.depth,
        activation=ACTIVATIONS[nn_set.hidden_activation],
        head_kernel_std=nn_set.head_init_scale,
        final_activation=unit_softplus,
        center=(math.log(z_0), math.log(k_0)),
        output_scale=(delta * k_0,),
        dtype=dtype,
    )
    graphdef, _, nondiff = nnx.split(
        StateMLP(2, 1, rngs=nnx.Rngs(0), **policy_spec),
        nnx.Param,
        ...,
    )
    # The initial training draw consumes train_ic_key itself; the epoch and
    # restart streams chain from independent fold-ins of the same key.
    residual_args = ResidualArgs(
        graphdef=graphdef,
        nondiff=nondiff,
        initial_log_states=jnp.log(train_initial_conditions),
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
        data_set=data_set,
    )

    # Multi-start Levenberg-Marquardt training.
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    _, theta_0, _ = nnx.split(
        StateMLP(2, 1, rngs=nnx.Rngs(init_key), **policy_spec),
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
    )
    # The per-step tolerance is sound only while the training batch never
    # changes; across epochs the callback judges the fresh batch at the
    # boundaries instead.
    atol = (
        math.sqrt(tolerance_set.train_loss_threshold)
        if tolerance_set.num_epochs == 1 and tolerance_set.epoch_early_stopping
        else 0.0
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=lm_set.num_attempts,
        draw=DrawNNXModule(StateMLP, 2, 1, **policy_spec),
        reset_args=ResetEpochStream(),
    )
    solve_options = dict(
        max_steps=tolerance_set.num_epochs * tolerance_set.max_steps,
        atol=atol,
        callback=callback,
        # A completed multi-epoch schedule is a normal stop; a single-epoch
        # solve must reach atol.
        max_steps_is_success=tolerance_set.num_epochs > 1,
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
    loss, train_ok = train_accepted(theta_final, solve_args, p)
    train_loss = float(loss)
    epochs_run = int(solve_args.epoch) + 1
    stopping_reason = STATUS_TO_REASON.get(int(result.status), "unknown")
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

    # Training diagnostics use the batch the solution was returned on.
    train_batch_log_states = solve_args.initial_log_states
    train_batch_initial_conditions = jnp.exp(train_batch_log_states)
    train_solution = simulate_train(policy, train_batch_log_states, p, data_set)
    train_values = trajectory_values(policy, train_solution.xs, p)
    train_initial = np.asarray(train_batch_initial_conditions)
    train_df = long_df(
        {"t": train_solution.ts, **train_values},
        ("trajectory", "step"),
        scalars={"z_0": train_initial[:, 0], "k_0": train_initial[:, 1]},
    )
    # Adaptive simulations pad the step buffer past the accepted prefix; those
    # rows are not training states and leave the frame here.
    train_step_mask = (
        None if data_set.ode_solver == "rk4" else np.asarray(train_solution.accepted)
    )
    if train_step_mask is not None:
        train_df = train_df[train_step_mask.reshape(-1)].reset_index(drop=True)
    num_simulation_states = int(
        train_solution.ts.size if train_step_mask is None else train_step_mask.sum()
    )

    # Held-out simulations and benchmark errors.
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

    # Terminal transversality certificate on the training batch.
    train_tvc = log_tvc(
        policy, train_batch_log_states, data_set.train_tvc_T, data_set.tvc_steps, p
    )
    max_log_tvc = float(jnp.max(train_tvc))

    max_train_euler = float(train_df["euler_residual"].abs().max())
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_euler_mean_abs = float(jnp.mean(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    rejection_reasons = []
    if tolerance_set.gate_train and not bool(train_ok):
        rejection_reasons.append("train")
    if tolerance_set.gate_test and not (
        test_euler_mean_abs <= tolerance_set.test_euler_mean_abs
        and max_test_euler <= tolerance_set.test_euler_max
    ):
        rejection_reasons.append("test")
    acceptance = acceptance_fields(rejection_reasons)
    solution_accepted = acceptance["solution_accepted"]

    results = {
        "parameters": p.to_dict(),
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": {
            "init_damping": lm_set.init_damping,
            "damping_decrease": lm_set.damping_decrease,
            "damping_increase": lm_set.damping_increase,
            "geodesic_acceleration": lm_set.geodesic_acceleration,
            "geodesic_acceptance_ratio": lm_set.geodesic_acceptance_ratio,
            "num_attempts": lm_set.num_attempts,
        },
        "tolerance_set": asdict(tolerance_set),
        "seed": seed,
        "train_loss": train_loss,
        "max_train_euler": max_train_euler,
        "max_test_euler": max_test_euler,
        "final_damping": final_damping,
        "test_euler_mean_abs": test_euler_mean_abs,
        "test_euler_mean_abs_success": (
            test_euler_mean_abs if solution_accepted else np.nan
        ),
        "test_trajectory_euler_mean_abs_p50": float(
            jnp.quantile(test_trajectory_euler, 0.5)
        ),
        "test_trajectory_euler_mean_abs_p90": float(
            jnp.quantile(test_trajectory_euler, 0.9)
        ),
        "test_trajectory_euler_mean_abs_max": float(jnp.max(test_trajectory_euler)),
        "max_log_tvc": max_log_tvc,
        "min_test_c": float(jnp.min(test_values["c"])),
        "steps": steps,
        "solver_status": stopping_reason,
        "epochs_run": epochs_run,
        "attempt": attempt,
        "attempts_run": int(result.multi_start.attempts_run),
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        "solver_accepted": solver_accepted,
        **acceptance,
        "num_simulation_states": num_simulation_states,
        "train_path_ok": bool(jnp.all(train_solution.ok)),
        **error_fields,
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
    }
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
    return results, extras


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn_recursive)
