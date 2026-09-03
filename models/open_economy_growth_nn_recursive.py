from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import DrawNNXModule, LevenbergMarquardt, MultiStart
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_ode

from models.neural_networks import ACTIVATIONS, StateMLP, UnitSoftplusWhere
from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.open_economy_growth_shared import (
    OPEN_ECONOMY_GROWTH_CONSTANTS as CONSTANTS,
)
from models.open_economy_growth_shared import (
    StructuralParameters,
    marginal_product,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.utilities import (
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    value_and_grad_fwd,
)

# Sequential/state-space formulation. The neural policy supplies positive
# investment expenditure x(k) and the signed current account CA(k); capital and
# assets are rolled forward, while q and c are recovered algebraically. No
# terminal, TVC, no-Ponzi, steady-state, or benchmark information enters the
# residual or initialization.


# Acceptance is one predicate, evaluated by the multi-start accept hook and
# ex post: gate_train requires the max Euler residual on the training path
# under train_residual, gate_test the max Euler residual on the diagnostic
# path (T = diagnostic_T) under test_euler_max.
@dataclass(frozen=True)
class ToleranceSettings:
    gate_train: bool = True
    gate_test: bool = True
    train_residual: float = 1e-5
    test_euler_max: float = 5e-6


@dataclass(frozen=True)
class DataSettings:
    # Every simulation is fixed-step RK4 evaluated on the integrator's own states:
    # ``n`` points come from ``n - 1`` steps of ``T / (n - 1)``, including the
    # initial condition. Training places a residual on every knot; the held-out
    # and diagnostic paths read the same states, so nothing is interpolated.
    train_points: int = 65
    train_T: float = 40.0
    test_T: float = 50.0
    test_points: int = 51
    diagnostic_T: float = 160.0
    diagnostic_points: int = 161
    policy_grid_size: int = 100


@dataclass(frozen=True)
class SequentialPolicySettings:
    width: int = 16
    depth: int = 2
    hidden_activation: str = "silu"
    head_init_scale: float = 1e-3


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    train_residual: jax.Array
    test_euler_max: jax.Array
    data_set: DataSettings = struct.field(pytree_node=False)
    gate_train: bool = struct.field(pytree_node=False)
    gate_test: bool = struct.field(pytree_node=False)


def policy_values(policy, k, a, p):
    flows, flow_log_derivatives = value_and_grad_fwd(policy)(jnp.log(k))
    investment, current_account = flows
    # The policy maps log k; level derivatives divide the log-derivatives by k.
    investment_derivative, current_account_derivative = flow_log_derivatives / k
    investment_rate = investment / k
    eta = p.eta
    chi = p.delta ** (1.0 - eta)
    q = investment_rate ** (1.0 - eta) / (chi * eta)
    output = production(k, 1.0, p)
    mpk = marginal_product(k, 1.0, p)
    capital_drift = k * (chi * investment_rate**eta - p.delta)
    log_q_derivative = (1.0 - eta) * (investment_derivative / investment - 1.0 / k)
    log_q_drift = log_q_derivative * capital_drift
    q_equation = p.r_star + p.delta - (1.0 - eta) * chi * investment_rate**eta - mpk / q
    consumption = p.r_star * a + output - investment - current_account
    consumption_drift = (
        mpk - investment_derivative - current_account_derivative
    ) * capital_drift + p.r_star * current_account
    consumption_growth = (p.r_star - p.r) / p.gamma
    return {
        "k": k,
        "a": a,
        "q": q,
        "c": consumption,
        "x": investment,
        "output": output,
        "current_account": current_account,
        "capital_drift": capital_drift,
        "q_euler": log_q_drift - q_equation,
        "c_euler": consumption_drift - consumption_growth * consumption,
    }


def path_values(policy, states, p):
    return jax.vmap(lambda state: policy_values(policy, state[0], state[1], p))(states)


def simulate(policy, T, budget, k_0, a_0, p, *, save_at):
    log_k_0 = jnp.log(k_0)

    def drift(state):
        k = k_0 * jnp.exp(state[0])
        investment, current_account = policy(log_k_0 + state[0])
        investment_rate = investment / k
        chi = p.delta ** (1.0 - p.eta)
        capital_drift = k * (chi * investment_rate**p.eta - p.delta)
        return jnp.stack([capital_drift / k, current_account])

    dtype = jnp.asarray(k_0).dtype
    return solve_ode(
        drift,
        RK4(),
        0.0,
        T,
        jnp.stack([jnp.zeros((), dtype=dtype), jnp.asarray(a_0, dtype=dtype)]),
        dt_0=T / budget,
        controller=ConstantStepSize(),
        max_steps=budget,
        save_at=save_at,
    )


def states_from_solution(solution, k_0):
    k = k_0 * jnp.exp(solution.xs[:, 0])
    states = jnp.stack([k, solution.xs[:, 1]], axis=1)
    return jnp.where(solution.ok, states, jnp.full_like(states, jnp.inf))


@partial(jax.jit, static_argnames=("data_set",))
def simulate_train(policy, initial_k, initial_a, p, data_set):
    def solve_one(k_0, a_0):
        return simulate(
            policy,
            data_set.train_T,
            data_set.train_points - 1,
            k_0,
            a_0,
            p,
            save_at=SaveAt(steps=True),
        )

    return jax.vmap(solve_one)(initial_k, initial_a)


# Collocation states, the active-state mask, and the raw solution. Every step
# of the fixed-step mesh is a collocation point, so the returned mask is full;
# it is kept so the caller can assert that.
def collocation_states(policy, p, data_set):
    initial_k = jnp.reshape(p.k_0, (1,))
    initial_a = jnp.reshape(p.a_0, (1,))
    solution = simulate_train(policy, initial_k, initial_a, p, data_set)
    states = jax.vmap(states_from_solution)(solution, initial_k)
    safe_states = jnp.where(
        solution.accepted[..., None],
        states,
        jnp.stack([initial_k, initial_a], axis=1)[:, None, :],
    )
    return safe_states, solution.accepted, solution


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.nondiff)
    states, active, solution = collocation_states(policy, p, args.data_set)
    values = path_values(policy, states.reshape(-1, 2), p)
    euler = jnp.stack([values["q_euler"], values["c_euler"]], axis=-1)
    euler = euler.reshape(active.shape + (2,))
    active_count = jnp.asarray(jnp.maximum(jnp.sum(active), 1), dtype=euler.dtype)
    residual = jnp.where(active[..., None], euler, jnp.zeros_like(euler)) / jnp.sqrt(
        2.0 * active_count
    )
    # A budget-exhausted trajectory did not cover the requested economic
    # horizon. Make that a loud optimizer failure instead of fitting a prefix.
    return jnp.where(
        solution.ok[:, None, None], residual, jnp.full_like(residual, jnp.inf)
    ).reshape(-1)


@partial(jax.jit, static_argnames=("budget",))
def evaluate_path_jit(policy, T, budget, p):
    solution = simulate(policy, T, budget, p.k_0, p.a_0, p, save_at=SaveAt(steps=True))
    states = states_from_solution(solution, p.k_0)
    return {"t": solution.ts, **path_values(policy, states, p)}


# Evaluates along the integrator's own states: n_points from n - 1 steps.
def evaluate_path(policy, T, n_points, p):
    return evaluate_path_jit(policy, T, n_points - 1, p)


def max_euler_residual(values):
    return jnp.maximum(
        jnp.max(jnp.abs(values["q_euler"])), jnp.max(jnp.abs(values["c_euler"]))
    )


# The acceptance predicate: (ok, train residual, test residual, train_ok,
# test_ok). NaN residuals fail their comparison, so a diverged path is never
# accepted.
def accepted(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.nondiff)
    data_set = args.data_set
    train = max_euler_residual(
        evaluate_path(policy, data_set.train_T, data_set.train_points, p)
    )
    test = max_euler_residual(
        evaluate_path(policy, data_set.diagnostic_T, data_set.diagnostic_points, p)
    )
    train_ok = train <= args.train_residual
    test_ok = test <= args.test_euler_max
    ok = (train_ok | (not args.gate_train)) & (test_ok | (not args.gate_test))
    return ok, train, test, train_ok, test_ok


def accept_training(_, result):
    return accepted(result.x, result.args, result.p)[0]


def open_economy_growth_nn_recursive(
    alpha: float = CONSTANTS["alpha"],
    gamma: float = CONSTANTS["gamma"],
    delta: float = CONSTANTS["delta"],
    r: float = CONSTANTS["r"],
    eta: float = CONSTANTS["eta"],
    r_star: float = CONSTANTS["r_star"],
    k_0: float = 1.0,
    a_0: float = 0.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: SequentialPolicySettings = SequentialPolicySettings(),
    # atol is on the residual norm (the loss stops at atol^2).
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=3.26e-5,
        damping_decrease=0.482,
        damping_increase=2.71,
        geodesic_acceptance_ratio=0.541,
        max_steps=250,
        atol=1e-07,
        num_attempts=5,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    p = StructuralParameters.create(
        dtype=dtype,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        r=r,
        eta=eta,
        r_star=r_star,
        k_0=k_0,
        a_0=a_0,
    )
    initial_output = float(production(p.k_0, 1.0, p))
    policy_spec = dict(
        width=nn_set.width,
        depth=nn_set.depth,
        activation=ACTIVATIONS[nn_set.hidden_activation],
        head_kernel_std=nn_set.head_init_scale,
        # Investment starts at replacement investment delta * k_0; the
        # local_slope gives both outputs the same O(initial_output) Jacobian.
        final_activation=UnitSoftplusWhere(
            local_slope=(initial_output / (delta * k_0), None)
        ),
        center=(math.log(k_0),),
        output_scale=(delta * k_0, initial_output),
        dtype=dtype,
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = StateMLP(1, 2, rngs=nnx.Rngs(init_key), **policy_spec)
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        nondiff=nondiff,
        train_residual=jnp.asarray(tolerance_set.train_residual, dtype=dtype),
        test_euler_max=jnp.asarray(tolerance_set.test_euler_max, dtype=dtype),
        data_set=data_set,
        gate_train=tolerance_set.gate_train,
        gate_test=tolerance_set.gate_test,
    )

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=lm_set.num_attempts,
        draw=DrawNNXModule(StateMLP, 1, 2, **policy_spec),
        accept=accept_training,
    )
    solve_options = dict(
        max_steps=lm_set.max_steps,
        atol=lm_set.atol if tolerance_set.gate_train else 0.0,
        max_steps_is_success=False,
    )
    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        residual_args,
        p=p,
        multi_start=multi_start,
        **solve_options,
    )
    solve_time = timer.elapsed(result)
    attempt = int(result.multi_start.attempt) + 1
    solver_accepted = bool(result.multi_start.accepted)
    _, train_residual, test_residual, train_ok, test_ok = accepted(
        result.x, residual_args, p
    )
    max_train_residual = float(train_residual)
    max_validation_residual = float(test_residual)
    rejection_reasons = []
    if tolerance_set.gate_train and not bool(train_ok):
        rejection_reasons.append("train")
    if tolerance_set.gate_test and not bool(test_ok):
        rejection_reasons.append("test")
    acceptance = acceptance_fields(rejection_reasons)
    solution_accepted = acceptance["solution_accepted"]
    policy = nnx.merge(graphdef, result.x, nondiff)

    train_eval = evaluate_path(policy, data_set.train_T, data_set.train_points, p)
    test_eval = evaluate_path(policy, data_set.test_T, data_set.test_points, p)
    diagnostic_eval = evaluate_path(
        policy, data_set.diagnostic_T, data_set.diagnostic_points, p
    )
    train_t = train_eval["t"]
    test_t = test_eval["t"]
    jax.block_until_ready((train_eval, test_eval, diagnostic_eval))

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(p)
    benchmark_values = benchmark(np.asarray(test_t))
    benchmark_test_values = {
        name: jnp.asarray(benchmark_values[name], dtype=dtype)
        for name in ("k", "q", "c", "a")
    }
    output_scale = jnp.asarray(steady_state(p)["output"], dtype=dtype)
    k_relative_error = (
        jnp.abs(test_eval["k"] - benchmark_test_values["k"])
        / benchmark_test_values["k"]
    )
    q_relative_error = (
        jnp.abs(test_eval["q"] - benchmark_test_values["q"])
        / benchmark_test_values["q"]
    )
    c_relative_error = (
        jnp.abs(test_eval["c"] - benchmark_test_values["c"])
        / benchmark_test_values["c"]
    )
    a_scaled_error = jnp.abs(test_eval["a"] - benchmark_test_values["a"]) / output_scale
    max_benchmark_error = max(
        float(jnp.max(k_relative_error)),
        float(jnp.max(q_relative_error)),
        float(jnp.max(c_relative_error)),
        float(jnp.max(a_scaled_error)),
    )

    terminal = {name: diagnostic_eval[name][-1] for name in diagnostic_eval}
    tail = tail_diagnostics(
        terminal["k"],
        terminal["a"],
        terminal["q"],
        terminal["c"],
        p,
    )
    tvc = tvc_diagnostics(
        terminal["k"],
        terminal["a"],
        terminal["q"],
        terminal["c"],
        terminal["t"],
        p,
    )
    pv_gap = present_value_gap(
        diagnostic_eval["t"],
        diagnostic_eval["k"],
        diagnostic_eval["q"],
        diagnostic_eval["c"],
        p,
    )
    theta_initial, _ = ravel_pytree(theta_0)
    theta_final, _ = ravel_pytree(result.x)

    steady = steady_state(p)
    grid_lo = min(k_0, float(steady["k"]))
    grid_hi = max(k_0, float(steady["k"]))
    margin = 0.05 * (grid_hi - grid_lo) + 1e-6
    # Floor at a positive value: the grid feeds the policy in log k.
    policy_k = jnp.linspace(
        max(grid_lo - margin, 1e-8),
        grid_hi + margin,
        data_set.policy_grid_size,
        dtype=dtype,
    )
    policy_flows = jax.vmap(policy)(jnp.log(policy_k))
    jax.block_until_ready(policy_flows)
    policy_x = policy_flows[:, 0]
    policy_ca = policy_flows[:, 1]
    policy_s = policy_x / policy_k
    policy_chi = delta ** (1.0 - eta)
    policy_q = policy_s ** (1.0 - eta) / (policy_chi * eta)

    results = {
        "parameters": p.to_dict(),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": {
            "init_damping": lm_set.init_damping,
            "damping_decrease": lm_set.damping_decrease,
            "damping_increase": lm_set.damping_increase,
            "geodesic_acceleration": lm_set.geodesic_acceleration,
            "geodesic_acceptance_ratio": lm_set.geodesic_acceptance_ratio,
            "max_steps": lm_set.max_steps,
            "atol": lm_set.atol,
            "num_attempts": lm_set.num_attempts,
        },
        "tolerance_set": asdict(tolerance_set),
        "output_ss": float(output_scale),
        "seed": seed,
        "dae_residual_count": 2 * int(train_t.size),
        "domain_residual_count": int(train_t.size),
        "residual_count": 3 * int(train_t.size),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_validation_residual_success": (
            max_validation_residual if solution_accepted else np.nan
        ),
        "max_benchmark_error": max_benchmark_error,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "max_consumption_variation": float(
            jnp.max(jnp.abs(diagnostic_eval["c"] - diagnostic_eval["c"][0]))
        ),
        "ex_post_present_value_gap": float(jnp.abs(pv_gap)),
        "ex_post_current_account": float(jnp.abs(terminal["current_account"])),
        "ex_post_stable_gap": abs(float(tail["stable"])),
        "ex_post_present_value_tail_gap": abs(float(tail["present_value"])),
        "ex_post_discounted_assets": float(tvc["discounted_assets"]),
        "ex_post_asset_tvc": float(tvc["asset_tvc"]),
        "ex_post_capital_tvc": float(tvc["capital_tvc"]),
        "benchmark_present_value_residual": abs(
            float(benchmark_diagnostics["present_value_residual"])
        ),
        "steps": int(result.steps),
        "attempt": attempt,
        "attempts_run": int(result.multi_start.attempts_run),
        "solve_time": solve_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "solver_accepted": solver_accepted,
        "total_params": int(theta_final.size),
        "unknown_count": int(theta_final.size),
        "num_training_trajectories": 1,
        "ode_solver": "rk4",
        **acceptance,
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    frame_names = (
        "t",
        "k",
        "a",
        "q",
        "c",
        "x",
        "output",
        "current_account",
        "capital_drift",
        "q_euler",
        "c_euler",
    )
    train_df = pd.DataFrame(
        {name: np.asarray(train_eval[name]) for name in frame_names}
    )
    test_df = pd.DataFrame({name: np.asarray(test_eval[name]) for name in frame_names})
    diagnostic_df = pd.DataFrame(
        {name: np.asarray(diagnostic_eval[name]) for name in frame_names}
    )
    for name in ("k", "a", "q", "c", "x", "output", "current_account"):
        test_df[f"{name}_benchmark"] = np.asarray(benchmark_values[name])
    test_df["k_relative_error"] = np.asarray(k_relative_error)
    test_df["q_relative_error"] = np.asarray(q_relative_error)
    test_df["c_relative_error"] = np.asarray(c_relative_error)
    test_df["a_scaled_error"] = np.asarray(a_scaled_error)
    policy_df = pd.DataFrame(
        {
            "k": np.asarray(policy_k),
            "x": np.asarray(policy_x),
            "current_account": np.asarray(policy_ca),
            "q": np.asarray(policy_q),
        }
    )
    jax.block_until_ready(nnx.state(policy))
    return results, {
        "policy": policy,
        "train_df": train_df,
        "test_df": test_df,
        "diagnostic_df": diagnostic_df,
        "policy_df": policy_df,
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": p,
            "residual_args": residual_args,
            "multi_start": multi_start,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_nn_recursive)
