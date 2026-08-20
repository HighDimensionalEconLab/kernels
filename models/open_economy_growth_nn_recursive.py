from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
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

from models.open_economy_growth import (
    ModelParameters,
    marginal_product,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from models.open_economy_growth_benchmark import open_economy_growth_benchmark
from models.utilities import (
    ACTIVATIONS,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    unit_softplus,
)

# Sequential/state-space formulation. The neural policy supplies positive
# investment expenditure x(k) and the signed current account CA(k). Capital and
# assets are rolled forward, while q and c are recovered algebraically. This is
# deliberate: a free consumption head has no useful bias toward the annuity
# root, whereas a smooth bounded CA(k) cannot represent the singular explosive
# asset branch as capital converges. No terminal, TVC, no-Ponzi, steady-state,
# or benchmark information enters the residual or initialization.


# Acceptance gates (CLI-reachable so a float32 caller can pass its own bar;
# the float32 replication driver does) plus the solve budget.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    validation_residual: float = 5e-3
    benchmark_match: float = 1e-3
    max_steps: int = 250
    train_loss_threshold: float = 1e-14
    # frac_accepted measures acceptance within the restart budget, not first-try
    # robustness: a seed that stalls short of the loss threshold is redrawn.
    num_attempts: int = 5


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
    width: int = 32
    depth: int = 3
    hidden_activation: str = "gelu"
    flow_scale: float = 1.0
    investment_anchor: float = 1.0
    k_0_ref: float = 1.0


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    data_set: DataSettings = struct.field(pytree_node=False)


@struct.dataclass
class DynamicModelParameters:
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    r: jax.Array
    eta: jax.Array
    r_star: jax.Array
    chi: jax.Array

    @classmethod
    def from_model(cls, p, dtype):
        return cls(
            alpha=jnp.asarray(p.alpha, dtype=dtype),
            gamma=jnp.asarray(p.gamma, dtype=dtype),
            delta=jnp.asarray(p.delta, dtype=dtype),
            r=jnp.asarray(p.r, dtype=dtype),
            eta=jnp.asarray(p.eta, dtype=dtype),
            r_star=jnp.asarray(p.r_star, dtype=dtype),
            chi=jnp.asarray(p.delta ** (1.0 - p.eta), dtype=dtype),
        )


@struct.dataclass
class ResidualParameters:
    k_0: jax.Array
    a_0: jax.Array
    model: DynamicModelParameters


class SequentialPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: SequentialPolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        self.flow_scale = nnx.Variable(jnp.asarray(settings.flow_scale, dtype=dtype))
        self.investment_anchor = nnx.Variable(
            jnp.asarray(settings.investment_anchor, dtype=dtype)
        )
        self.k_0_ref = nnx.Variable(jnp.asarray(settings.k_0_ref, dtype=dtype))
        self.layers = nnx.List(
            [nnx.Linear(1, settings.width, dtype=dtype, param_dtype=dtype, rngs=rngs)]
        )
        for _ in range(settings.depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.width,
                    settings.width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.head = nnx.Linear(
            settings.width,
            2,
            kernel_init=nnx.initializers.normal(1e-3),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, k):
        h = jnp.atleast_1d(jnp.log(k / self.k_0_ref[...]))
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        raw = self.head(h)
        investment = self.investment_anchor[...] * unit_softplus(
            (self.flow_scale[...] / self.investment_anchor[...]) * raw[0]
        )
        current_account = self.flow_scale[...] * raw[1]
        return jnp.stack([investment, current_account])


def policy_values(policy, k, a, p):
    flows, flow_derivatives = jax.jvp(policy, (k,), (jnp.ones_like(k),))
    investment, current_account = flows
    investment_derivative, current_account_derivative = flow_derivatives
    investment_rate = investment / k
    eta = p.eta
    q = investment_rate ** (1.0 - eta) / (p.chi * eta)
    output = production(k, 1.0, p)
    mpk = marginal_product(k, 1.0, p)
    capital_drift = k * (p.chi * investment_rate**eta - p.delta)
    log_q_derivative = (1.0 - eta) * (investment_derivative / investment - 1.0 / k)
    log_q_drift = log_q_derivative * capital_drift
    q_equation = (
        p.r_star + p.delta - (1.0 - eta) * p.chi * investment_rate**eta - mpk / q
    )
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


def drift_field(state, _, args, p):
    policy, k_0 = p
    k = k_0 * jnp.exp(state[0])
    investment, current_account = policy(k)
    investment_rate = investment / k
    capital_drift = k * (args.chi * investment_rate**args.eta - args.delta)
    return jnp.stack(
        [
            capital_drift / k,
            current_account / policy.flow_scale[...],
        ]
    )


def simulate(policy, T, budget, k_0, args, data_set, *, save_at):
    return solve_ode(
        drift_field,
        RK4(),
        0.0,
        T,
        jnp.zeros(2, dtype=jnp.asarray(k_0).dtype),
        dt_0=T / budget,
        controller=ConstantStepSize(),
        max_steps=budget,
        save_at=save_at,
        args=args,
        p=(policy, k_0),
    )


def states_from_solution(solution, k_0, a_0, flow_scale):
    k = k_0 * jnp.exp(solution.xs[:, 0])
    a = a_0 + flow_scale * solution.xs[:, 1]
    states = jnp.stack([k, a], axis=1)
    return jnp.where(solution.ok, states, jnp.full_like(states, jnp.inf))


@partial(jax.jit, static_argnames=("data_set",))
def simulate_train(policy, initial_k, p, data_set):
    def solve_one(k_0):
        return simulate(
            policy,
            data_set.train_T,
            data_set.train_points - 1,
            k_0,
            p,
            data_set,
            save_at=SaveAt(steps=True),
        )

    return jax.vmap(solve_one)(initial_k)


def collocation_states(policy, k_0, a_0, p, data_set):
    """Return collocation states, the active-state mask, and the raw solution.

    Every step of the fixed-step mesh is a collocation point, so the returned
    mask is full; it is kept so the caller can assert that.
    """
    initial_k = jnp.reshape(k_0, (1,))
    initial_a = jnp.reshape(a_0, (1,))
    solution = simulate_train(policy, initial_k, p, data_set)
    states = jax.vmap(states_from_solution, in_axes=(0, 0, 0, None))(
        solution,
        initial_k,
        initial_a,
        policy.flow_scale[...],
    )
    safe_states = jnp.where(
        solution.accepted[..., None],
        states,
        jnp.stack([initial_k, initial_a], axis=1)[:, None, :],
    )
    return safe_states, solution.accepted, solution


def residual_fn(theta, args, p):
    policy = nnx.merge(args.graphdef, theta, args.nondiff)
    states, active, solution = collocation_states(
        policy,
        p.k_0,
        p.a_0,
        p.model,
        args.data_set,
    )
    values = path_values(policy, states.reshape(-1, 2), p.model)
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
def evaluate_path_jit(policy, T, budget, k_0, a_0, args):
    solution = solve_ode(
        drift_field,
        RK4(),
        0.0,
        T,
        jnp.zeros(2, dtype=jnp.asarray(k_0).dtype),
        dt_0=T / budget,
        max_steps=budget,
        save_at=SaveAt(steps=True),
        args=args,
        p=(policy, k_0),
    )
    states = states_from_solution(solution, k_0, a_0, policy.flow_scale[...])
    return {"t": solution.ts, **path_values(policy, states, args)}


def evaluate_path(policy, T, n_points, k_0, a_0, p):
    """Evaluate along the integrator's own states: ``n_points`` from n-1 steps."""
    dtype = jnp.asarray(k_0).dtype
    dynamic_parameters = DynamicModelParameters.from_model(p, dtype)
    return evaluate_path_jit(
        policy,
        T,
        n_points - 1,
        k_0,
        a_0,
        dynamic_parameters,
    )


def open_economy_growth_nn_recursive(
    k_0: float = 1.0,
    a_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: SequentialPolicySettings = SequentialPolicySettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=3.26e-5,
        damping_decrease=0.482,
        damping_increase=2.71,
        geodesic_acceptance_ratio=0.541,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    k_0_jax = jnp.asarray(k_0, dtype=dtype)
    a_0_jax = jnp.asarray(a_0, dtype=dtype)
    dynamic_parameters = DynamicModelParameters.from_model(parameters, dtype)
    initial_output = float(production(k_0_jax, 1.0, parameters))
    nn_set = replace(
        nn_set,
        flow_scale=initial_output,
        investment_anchor=parameters.delta * k_0,
        k_0_ref=k_0,
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = SequentialPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key))
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        nondiff=nondiff,
        data_set=data_set,
    )
    residual_parameters = ResidualParameters(
        k_0=k_0_jax,
        a_0=a_0_jax,
        model=dynamic_parameters,
    )

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    atol = math.sqrt(tolerance_set.train_loss_threshold)
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=tolerance_set.num_attempts,
        draw=DrawNNXModule(SequentialPolicyNN, nn_set, dtype=dtype),
    )
    solve_options = dict(
        max_steps=tolerance_set.max_steps,
        atol=atol,
        max_steps_is_success=False,
    )
    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        residual_args,
        p=residual_parameters,
        multi_start=multi_start,
        **solve_options,
    )
    solve_time = timer.elapsed(result)
    attempt = int(result.multi_start.attempt) + 1
    policy = nnx.merge(graphdef, result.x, nondiff)

    train_eval = evaluate_path(
        policy,
        data_set.train_T,
        data_set.train_points,
        k_0_jax,
        a_0_jax,
        parameters,
    )
    test_eval = evaluate_path(
        policy,
        data_set.test_T,
        data_set.test_points,
        k_0_jax,
        a_0_jax,
        parameters,
    )
    diagnostic_eval = evaluate_path(
        policy,
        data_set.diagnostic_T,
        data_set.diagnostic_points,
        k_0_jax,
        a_0_jax,
        parameters,
    )
    train_t = train_eval["t"]
    test_t = test_eval["t"]
    jax.block_until_ready((train_eval, test_eval, diagnostic_eval))

    residual_names = ("q_euler", "c_euler")
    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(diagnostic_eval[name]))) for name in residual_names
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, test_eval, diagnostic_eval)
            for name in ("k", "q", "c", "x")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, test_eval, diagnostic_eval)
            for name in ("a", "current_account", "q_euler", "c_euler")
        )
    )
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    if max_validation_residual > tolerance_set.validation_residual:
        rejection_reasons.append("validation_residual")
    acceptance = acceptance_fields(rejection_reasons)
    solution_accepted = acceptance["solution_accepted"]

    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(
        k_0, a_0, parameters=parameters
    )
    benchmark_values = benchmark(np.asarray(test_t))
    benchmark_test_values = {
        name: jnp.asarray(benchmark_values[name], dtype=dtype)
        for name in ("k", "q", "c", "a")
    }
    output_scale = jnp.asarray(steady_state(parameters)["output"], dtype=dtype)
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
        parameters,
    )
    tvc = tvc_diagnostics(
        terminal["k"],
        terminal["a"],
        terminal["q"],
        terminal["c"],
        terminal["t"],
        parameters,
    )
    pv_gap = present_value_gap(
        diagnostic_eval["t"],
        diagnostic_eval["k"],
        diagnostic_eval["q"],
        diagnostic_eval["c"],
        a_0_jax,
        parameters,
    )
    theta_initial, _ = ravel_pytree(theta_0)
    theta_final, _ = ravel_pytree(result.x)

    steady = steady_state(parameters)
    grid_lo = min(k_0, float(steady["k"]))
    grid_hi = max(k_0, float(steady["k"]))
    margin = 0.05 * (grid_hi - grid_lo) + 1e-6
    # Floor at a positive value: the policy evaluates log k.
    policy_k = jnp.linspace(
        max(grid_lo - margin, 1e-8),
        grid_hi + margin,
        data_set.policy_grid_size,
        dtype=dtype,
    )
    policy_flows = jax.vmap(policy)(policy_k)
    jax.block_until_ready(policy_flows)
    policy_x = policy_flows[:, 0]
    policy_ca = policy_flows[:, 1]
    policy_s = policy_x / policy_k
    policy_chi = parameters.delta ** (1.0 - parameters.eta)
    policy_q = policy_s ** (1.0 - parameters.eta) / (policy_chi * parameters.eta)

    results = {
        **asdict(parameters),
        "r_star_resolved": parameters.r_star,
        "k_0": k_0,
        "a_0": a_0,
        "seed": seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["q_euler"].dtype),
        "terminal_residual_count": 0,
        "dae_residual_count": 2 * int(train_t.size),
        "domain_residual_count": int(train_t.size),
        "residual_count": 3 * int(train_t.size),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_match_tolerance": tolerance_set.benchmark_match,
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
        "solve_time": solve_time,
        "solve_time_success": solve_time if solution_accepted else np.nan,
        "total_params": int(theta_final.size),
        "unknown_count": int(theta_final.size),
        "parameter_norm_initial": float(jnp.linalg.norm(theta_initial)),
        "parameter_norm_final": float(jnp.linalg.norm(theta_final)),
        "parameter_step_norm": float(jnp.linalg.norm(theta_final - theta_initial)),
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
        "train_T": data_set.train_T,
        "num_train_points": int(train_t.size),
        "num_training_trajectories": 1,
        "test_T": data_set.test_T,
        "diagnostic_T": data_set.diagnostic_T,
        "ode_solver": "rk4",
        "train_steps": data_set.train_points - 1,
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
    hyperparameters = {
        "parameters": asdict(parameters),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
    }
    jax.block_until_ready(nnx.state(policy))
    return results, {
        "model": policy,
        "train_df": train_df,
        "test_df": test_df,
        "diagnostic_df": diagnostic_df,
        "policy_df": policy_df,
        "hyperparameters": hyperparameters,
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": residual_parameters,
            "residual_args": residual_args,
            "multi_start": multi_start,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_nn_recursive)
