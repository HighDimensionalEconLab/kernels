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
    RecursiveParameters,
    StructuralParameters,
    euler_residuals,
    growth_initial_conditions,
    log_state_drift_field,
    log_tvc_residual,
    simulate,
    trajectory_values,
)
from models.utilities import (
    ACTIVATIONS,
    STATUS_TO_REASON,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    long_df,
    standardized_initial_states,
    unit_softplus,
)


@dataclass(frozen=True)
class DataSettings:
    # Training residuals on the integrator's own states: n points from
    # n - 1 steps including the initial condition, nothing interpolated.
    # "rk4" spends the budget on a fixed grid; "tsit5" spends it adaptively
    # and enforces residuals on the accepted steps only. rtol/atol/initial_dt
    # configure the adaptive controller and are ignored under rk4; test/tvc
    # simulations stay fixed-step.
    train_points: int = 65
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = CONSTANTS.train_T
    train_tvc_T: float = CONSTANTS.transversality_T
    test_T: float = CONSTANTS.test_T
    ode_solver: str = "rk4"
    rtol: float = 1e-6
    atol: float = 1e-8
    initial_dt: float | None = None


@dataclass(frozen=True)
class ToleranceSettings:
    # max_steps is the LM-update budget per epoch; training ICs are redrawn at
    # each interior epoch boundary, so num_epochs = 1 never resamples.
    max_steps: int = 250
    train_loss_threshold: float = 1e-14
    log_tvc_threshold: float = 0.0
    num_epochs: int = 1
    epoch_early_stopping: bool = True


@dataclass(frozen=True)
class InvestmentPolicySettings:
    mlp_width: int = 32
    mlp_depth: int = 3
    hidden_activation: str = "gelu"
    head_init_scale: float = 1e-3
    homothetic_representation: bool = True


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


# i = anchor * U(h), U the unit softplus with near-zero head: homothetic is
# h(log kappa) with kappa = k / z^{1/(1-a)} the detrended capital, anchored at
# delta * k; non-homothetic h(centered log z, log k) anchored at the constant
# delta * k_scale; consumption Y - i stays unclamped.
class InvestmentPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: InvestmentPolicySettings,
        *,
        z_scale=1.0,
        k_scale=1.0,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.homothetic = nnx.static(settings.homothetic_representation)
        self.log_z_scale = nnx.Variable(jnp.asarray(math.log(z_scale), dtype=dtype))
        self.log_k_scale = nnx.Variable(jnp.asarray(math.log(k_scale), dtype=dtype))
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        in_features = 1 if self.homothetic else 2
        self.layers = nnx.List(
            [
                nnx.Linear(
                    in_features,
                    settings.mlp_width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(settings.mlp_depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.mlp_width,
                    settings.mlp_width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.head = nnx.Linear(
            settings.mlp_width,
            1,
            kernel_init=nnx.initializers.normal(settings.head_init_scale),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, log_state, p):
        log_z, log_k = log_state
        if self.homothetic:
            h = jnp.atleast_1d(log_k - log_z / (1.0 - p.a))
            anchor = p.delta * jnp.exp(log_k)
        else:
            h = jnp.stack(
                [log_z - self.log_z_scale[...], log_k - self.log_k_scale[...]]
            )
            anchor = p.delta * self.k_scale[...]
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return anchor * unit_softplus(self.head(h)[0])


# The load template only needs the architecture from meta: the scale
# Variables ride in the artifact's nondiff block and overwrite the defaults.
def make_investment_policy(meta, dtype):
    return InvestmentPolicyNN(
        InvestmentPolicySettings(**meta["settings"]), dtype=dtype, rngs=nnx.Rngs(0)
    )


# At each interior epoch boundary: redraw the training ICs, stop on a
# nonfinite or converged fresh-batch loss, else continue with damping reset.
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


# DrawNNXModule redraws theta but passes args through unchanged; multi-start
# chains each attempt's initial args from the previous draw, so re-chaining
# restart_key/epoch_key here (and resetting epoch to 0) is what keeps each
# retry's epoch-boundary resampling independent of the others.
class DrawInvestmentPolicy:
    def __init__(self, settings, *, z_scale, k_scale, dtype):
        self.draw_theta = DrawNNXModule(
            InvestmentPolicyNN, settings, z_scale=z_scale, k_scale=k_scale, dtype=dtype
        )

    def __call__(self, key, _, args):
        theta, _ = self.draw_theta(key, _, args)
        restart_key, epoch_key = jax.random.split(args.restart_key)
        return theta, args.replace(
            epoch_key=epoch_key,
            restart_key=restart_key,
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def __hash__(self):
        return hash((DrawInvestmentPolicy, self.draw_theta))

    def __eq__(self, other):
        return (
            isinstance(other, DrawInvestmentPolicy)
            and self.draw_theta == other.draw_theta
        )


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
            log_state_drift_field,
            solver,
            0.0,
            data_set.train_T,
            initial_log_state,
            dt_0=dt_0,
            controller=controller,
            max_steps=budget,
            save_at=SaveAt(steps=True),
            p=(policy, p),
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


# The stochastic variant (stochastic_growth_nn_recursive) solves this same
# model with stochastic TFP.
def neoclassical_growth_nn_recursive(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    sigma_crra: float = CONSTANTS.sigma_crra,
    g: float = 0.0,
    z_0: float = 1.0,
    k_0: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    ic_set: InitialConditionSettings = InitialConditionSettings(),
    nn_set: InvestmentPolicySettings = InvestmentPolicySettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=3.26e-5,
        damping_decrease=0.482,
        damping_increase=2.71,
        geodesic_acceleration=True,
        geodesic_acceptance_ratio=0.541,
        num_attempts=5,
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
    p = RecursiveParameters.create(
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
    graphdef, _, nondiff = nnx.split(
        InvestmentPolicyNN(
            nn_set, z_scale=z_0, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(0)
        ),
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
        InvestmentPolicyNN(
            nn_set, z_scale=z_0, k_scale=k_0, dtype=dtype, rngs=nnx.Rngs(init_key)
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
    )
    draw = DrawInvestmentPolicy(nn_set, z_scale=z_0, k_scale=k_0, dtype=dtype)
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
    train_accepted = (
        None if data_set.ode_solver == "rk4" else np.asarray(train_solution.accepted)
    )
    if train_accepted is not None:
        train_df = train_df[train_accepted.reshape(-1)].reset_index(drop=True)
    num_simulation_states = int(
        train_solution.ts.size if train_accepted is None else train_accepted.sum()
    )

    # Held-out simulations and benchmark errors.
    test_initial_log_states = jnp.log(test_initial_conditions)
    test_solution = simulate(policy, test_initial_log_states, test_T, n_test_steps, p)
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
    train_tvc = log_tvc_residual(
        policy, train_batch_log_states, data_set.train_tvc_T, data_set.tvc_steps, p
    )
    max_log_tvc_residual = float(jnp.max(train_tvc))
    transversality_ok = bool(max_log_tvc_residual < tolerance_set.log_tvc_threshold)

    max_train_euler = float(train_df["euler_residual"].abs().max())
    max_test_euler = float(jnp.max(jnp.abs(test_values["euler_residual"])))
    test_trajectory_euler = jnp.mean(jnp.abs(test_values["euler_residual"]), axis=1)

    rejection_reasons = []
    if not solver_accepted:
        rejection_reasons.append("train_loss")
    results = {
        "parameters": p.to_dict(),
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "test_z_0": ic_draw.test_z_center,
        "test_k_0": ic_draw.test_k_center,
        "seed": seed,
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
        "max_log_tvc_residual": max_log_tvc_residual,
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_values["c"])),
        "steps": steps,
        "epochs_run": epochs_run,
        "attempt": attempt,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "total_fit_time": total_fit_time,
        "solve_time_success": solve_time if solver_accepted else np.nan,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        **acceptance_fields(rejection_reasons),
        "num_simulation_states": num_simulation_states,
        "train_path_ok": bool(jnp.all(train_solution.ok)),
        "test_log_z_variance": ic_draw.test_log_z_variance,
        "test_log_k_variance": ic_draw.test_log_k_variance,
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
