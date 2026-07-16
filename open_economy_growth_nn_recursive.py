from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from jax import config
from jax.flatten_util import ravel_pytree
from nlls_gram import LMStatus, UnderdeterminedLevenbergMarquardt
from tinydiffeq import (
    RK4,
    ConstantStepSize,
    Euler,
    IController,
    Rodas5P,
    SaveAt,
    Tsit5,
    solve_ode,
)

from neural_networks import ACTIVATIONS, unit_softplus
from open_economy_growth import (
    ModelParameters,
    marginal_product,
    present_value_gap,
    production,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
)
from open_economy_growth_benchmark import open_economy_growth_benchmark

# Sequential/state-space formulation. The neural policy supplies positive
# investment expenditure x(k) and the signed current account CA(k). Capital and
# assets are rolled forward, while q and c are recovered algebraically. This is
# deliberate: a free consumption head has no useful bias toward the annuity
# root, whereas a smooth bounded CA(k) cannot represent the singular explosive
# asset branch as capital converges. No terminal, TVC, no-Ponzi, steady-state,
# or benchmark information enters the residual or initialization.
config.update("jax_enable_x64", True)

DOMAIN_EPS = 1e-8
TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
BENCHMARK_MATCH_TOL = 1e-3
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class DataSettings:
    dt: float = 1.0
    # RK4 is seed-robust at 32 internal steps. A 48-step Euler rollout is
    # faster for seed 0 but misses the independent RK4 residual gate for seeds
    # 1 and 2; adaptive Tsit5 is accurate but materially slower here.
    ode_solver: Literal["tsit5", "rk4", "euler", "rodas5p"] = "rk4"
    rollout_steps: int = 32
    diagnostic_steps: int = 160
    rtol: float = 6.3e-7
    atol: float = 2.2e-7
    train_T: float = 40.0
    train_points: int = 65
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 51
    diagnostic_T: float = 160.0
    diagnostic_points: int = 161
    benchmark_T: float = 200.0
    benchmark_points: int = 241
    policy_grid_size: int = 100


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 250
    train_loss_threshold: float = 1e-12


@dataclass(frozen=True)
class NLLSGramSettings:
    init_damping: float = 3.26e-5
    damping_decrease: float = 0.482
    damping_increase: float = 2.71
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.541


@dataclass(frozen=True)
class SequentialPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "gelu"
    flow_scale: float = 1.0
    investment_anchor: float = 1.0
    k_0_ref: float = 1.0


class SequentialPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: SequentialPolicySettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.flow_scale = settings.flow_scale
        self.investment_anchor = settings.investment_anchor
        self.k_0_ref = settings.k_0_ref
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
        h = jnp.atleast_1d(jnp.log(k / self.k_0_ref))
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        raw = self.head(h)
        investment = self.investment_anchor * unit_softplus(
            (self.flow_scale / self.investment_anchor) * raw[0]
        )
        current_account = self.flow_scale * raw[1]
        return jnp.stack([investment, current_account])


def make_train_t_grid(data_set, dtype):
    if data_set.train_points_list is None:
        values = np.linspace(
            0.0,
            data_set.train_T,
            data_set.train_points,
            dtype=np.float64,
        ).tolist()
    else:
        values = list(data_set.train_points_list)
    values = sorted({0.0, *(float(value) for value in values)})
    if values[0] < 0.0 or values[-1] > data_set.train_T:
        raise ValueError("training times must lie in [0, train_T]")
    return jnp.asarray(values, dtype=dtype)


def policy_values(policy, k, a, parameters):
    flows, flow_derivatives = jax.jvp(policy, (k,), (jnp.ones_like(k),))
    investment, current_account = flows
    investment_derivative, current_account_derivative = flow_derivatives
    investment_rate = investment / k
    installation_elasticity = parameters.installation_elasticity
    tobin_q = investment_rate ** (1.0 - installation_elasticity) / (
        parameters.installation_efficiency * installation_elasticity
    )
    output = production(k, 1.0, parameters)
    mpk = marginal_product(k, 1.0, parameters)
    capital_drift = k * (
        parameters.installation_efficiency
        * investment_rate**installation_elasticity
        - parameters.delta
    )
    log_q_derivative = (1.0 - installation_elasticity) * (
        investment_derivative / investment - 1.0 / k
    )
    log_q_drift = log_q_derivative * capital_drift
    q_equation = (
        parameters.interest_rate
        + parameters.delta
        - (1.0 - installation_elasticity)
        * parameters.installation_efficiency
        * investment_rate**installation_elasticity
        - mpk / tobin_q
    )
    consumption = (
        parameters.interest_rate * a
        + output
        - investment
        - current_account
    )
    consumption_drift = (
        (mpk - investment_derivative - current_account_derivative) * capital_drift
        + parameters.interest_rate * current_account
    )
    return {
        "k": k,
        "a": a,
        "q": tobin_q,
        "c": consumption,
        "x": investment,
        "output": output,
        "current_account": current_account,
        "capital_drift": capital_drift,
        "q_euler": log_q_drift - q_equation,
        "c_euler": consumption_drift / policy.flow_scale,
    }


def path_values(policy, states, parameters):
    return jax.vmap(lambda state: policy_values(policy, state[0], state[1], parameters))(
        states
    )


def rollout_field(policy, k_0, parameters):
    def state_drift(state):
        k = k_0 * jnp.exp(state[0])
        investment, current_account = policy(k)
        investment_rate = investment / k
        capital_drift = k * (
            parameters.installation_efficiency
            * investment_rate**parameters.installation_elasticity
            - parameters.delta
        )
        return jnp.stack(
            [
                capital_drift / k,
                current_account / policy.flow_scale,
            ]
        )

    return state_drift


def rollout(
    policy,
    T,
    budget,
    k_0,
    a_0,
    parameters,
    data_set,
    *,
    save_at,
    ode_solver_name=None,
):
    ode_solver_name = ode_solver_name or data_set.ode_solver
    if ode_solver_name == "tsit5":
        ode_solver = Tsit5()
        controller = IController(
            rtol=data_set.rtol,
            atol=data_set.atol,
            dt_min=1e-10,
        )
        dt_0 = data_set.dt
    elif ode_solver_name == "rodas5p":
        ode_solver = Rodas5P()
        controller = IController(
            rtol=data_set.rtol,
            atol=data_set.atol,
            dt_min=1e-10,
        )
        dt_0 = data_set.dt
    elif ode_solver_name == "rk4":
        ode_solver = RK4()
        controller = ConstantStepSize()
        dt_0 = T / budget
    elif ode_solver_name == "euler":
        ode_solver = Euler()
        controller = ConstantStepSize()
        dt_0 = T / budget
    else:
        raise ValueError(f"unknown ode_solver: {ode_solver_name!r}")
    return solve_ode(
        rollout_field(policy, k_0, parameters),
        ode_solver,
        0.0,
        T,
        jnp.zeros(2, dtype=jnp.asarray(k_0).dtype),
        dt_0=dt_0,
        controller=controller,
        max_steps=budget,
        save_at=save_at,
    )


def states_from_solution(solution, k_0, a_0, flow_scale):
    k = k_0 * jnp.exp(solution.xs[:, 0])
    a = a_0 + flow_scale * solution.xs[:, 1]
    states = jnp.stack([k, a], axis=1)
    return jnp.where(solution.ok, states, jnp.inf)


def collocation_states(policy, train_t, k_0, a_0, parameters, data_set):
    solution = rollout(
        policy,
        data_set.train_T,
        data_set.rollout_steps,
        k_0,
        a_0,
        parameters,
        data_set,
        save_at=SaveAt(ts=train_t),
    )
    return states_from_solution(solution, k_0, a_0, policy.flow_scale)


def evaluate_path(policy, times, budget, k_0, a_0, parameters, data_set):
    solution = rollout(
        policy,
        float(times[-1]),
        budget,
        k_0,
        a_0,
        parameters,
        data_set,
        save_at=SaveAt(ts=times),
        ode_solver_name="rk4",
    )
    states = states_from_solution(solution, k_0, a_0, policy.flow_scale)
    return {"t": times, **path_values(policy, states, parameters)}


def open_economy_growth_nn_recursive(
    k_0: float = 1.0,
    a_0: float = 0.0,
    parameters: ModelParameters = ModelParameters(),
    seed: int = 0,
    matmul_highest: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: SequentialPolicySettings = SequentialPolicySettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
    measure_compilation: bool = True,
):
    if k_0 <= 0.0:
        raise ValueError("k_0 must be positive")
    if abs(parameters.interest_rate - parameters.rho) > 1e-12:
        raise ValueError("the recursive baseline requires world_interest_rate == rho")
    if data_set.diagnostic_T < max(data_set.train_T, data_set.test_T):
        raise ValueError("diagnostic_T must cover the train and test horizons")
    if data_set.benchmark_T < data_set.test_T:
        raise ValueError("benchmark_T must cover the test horizon")
    if stop_set.train_loss_threshold <= 0.0:
        raise ValueError("train_loss_threshold must be positive")
    if matmul_highest:
        config.update("jax_default_matmul_precision", "highest")

    dtype = jnp.float64
    k_0_jax = jnp.asarray(k_0, dtype=dtype)
    a_0_jax = jnp.asarray(a_0, dtype=dtype)
    initial_output = float(production(k_0_jax, 1.0, parameters))
    flow_scale = max(initial_output, DOMAIN_EPS)
    nn_set = replace(
        nn_set,
        flow_scale=flow_scale,
        investment_anchor=max(parameters.delta * k_0, DOMAIN_EPS),
        k_0_ref=k_0,
    )
    train_t = make_train_t_grid(data_set, dtype)
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    diagnostic_t = jnp.linspace(
        0.0,
        data_set.diagnostic_T,
        data_set.diagnostic_points,
        dtype=dtype,
    )
    init_key = jax.random.key(seed)
    model = SequentialPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key))
    graphdef, theta_0, fixed_state = nnx.split(model, nnx.Param, ...)

    def residual_fn(theta):
        policy = nnx.merge(graphdef, theta, fixed_state)
        states = collocation_states(
            policy,
            train_t,
            k_0_jax,
            a_0_jax,
            parameters,
            data_set,
        )
        values = path_values(policy, states, parameters)
        consumption_positivity = jnp.minimum(
            values["c"] / policy.flow_scale - DOMAIN_EPS,
            0.0,
        )
        residual = jnp.concatenate(
            [values["q_euler"], values["c_euler"], consumption_positivity]
        )
        return residual / jnp.sqrt(residual.size)

    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver="cholesky",
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )
    atol = math.sqrt(stop_set.train_loss_threshold)
    compile_time = np.nan
    if measure_compilation:
        compile_start = time.perf_counter()
        compile_result = solver.solve(theta_0, max_steps=1, atol=atol)
        jax.block_until_ready(compile_result.x)
        compile_time = time.perf_counter() - compile_start

    train_start = time.perf_counter()
    result = solver.solve(theta_0, max_steps=stop_set.max_steps, atol=atol)
    jax.block_until_ready(result.x)
    train_time = time.perf_counter() - train_start
    policy = nnx.merge(graphdef, result.x, fixed_state)

    train_eval = evaluate_path(
        policy,
        train_t,
        data_set.rollout_steps,
        k_0_jax,
        a_0_jax,
        parameters,
        data_set,
    )
    test_steps = max(1, round(data_set.test_T / data_set.dt))
    test_eval = evaluate_path(
        policy,
        test_t,
        test_steps,
        k_0_jax,
        a_0_jax,
        parameters,
        data_set,
    )
    diagnostic_eval = evaluate_path(
        policy,
        diagnostic_t,
        data_set.diagnostic_steps,
        k_0_jax,
        a_0_jax,
        parameters,
        data_set,
    )

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
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    solution_accepted = not rejection_reasons

    benchmark_grid = np.linspace(
        0.0,
        data_set.benchmark_T,
        data_set.benchmark_points,
    )
    benchmark, benchmark_diagnostics = open_economy_growth_benchmark(
        k_0,
        a_0,
        benchmark_grid,
        parameters=parameters,
    )
    benchmark_values = benchmark(np.asarray(test_t))
    output_scale = float(steady_state(parameters)["output"])
    k_relative_error = jnp.abs(
        test_eval["k"] - jnp.asarray(benchmark_values["k"])
    ) / jnp.asarray(benchmark_values["k"])
    q_relative_error = jnp.abs(
        test_eval["q"] - jnp.asarray(benchmark_values["q"])
    ) / jnp.asarray(benchmark_values["q"])
    c_relative_error = jnp.abs(
        test_eval["c"] - jnp.asarray(benchmark_values["c"])
    ) / jnp.asarray(benchmark_values["c"])
    a_scaled_error = (
        jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
    )
    max_benchmark_error = max(
        float(jnp.max(k_relative_error)),
        float(jnp.max(q_relative_error)),
        float(jnp.max(c_relative_error)),
        float(jnp.max(a_scaled_error)),
    )
    benchmark_match = max_benchmark_error <= BENCHMARK_MATCH_TOL

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
        diagnostic_eval["c"][0],
        a_0_jax,
        parameters,
    )
    theta_initial, _ = ravel_pytree(theta_0)
    theta_final, _ = ravel_pytree(result.x)

    steady = steady_state(parameters)
    grid_lo = min(k_0, float(steady["k"]))
    grid_hi = max(k_0, float(steady["k"]))
    margin = 0.05 * (grid_hi - grid_lo) + 1e-6
    policy_k = jnp.linspace(
        max(grid_lo - margin, DOMAIN_EPS),
        grid_hi + margin,
        data_set.policy_grid_size,
        dtype=dtype,
    )
    policy_flows = jax.vmap(policy)(policy_k)
    policy_x = policy_flows[:, 0]
    policy_ca = policy_flows[:, 1]
    policy_s = policy_x / policy_k
    policy_q = policy_s ** (1.0 - parameters.installation_elasticity) / (
        parameters.installation_efficiency * parameters.installation_elasticity
    )

    results = {
        **asdict(parameters),
        "world_interest_rate_resolved": parameters.interest_rate,
        "k_0": k_0,
        "a_0": a_0,
        "seed": seed,
        "solver": "nlls-gram",
        "ansatz": "sequential_investment_current_account_levels",
        "selection": "minimum_identity_metric_parameter_step",
        "metric": "identity_parameter_metric",
        "initialization": "replacement_investment_near_zero_current_account",
        "terminal_residual_count": 0,
        "dae_residual_count": 2 * int(train_t.size),
        "domain_residual_count": int(train_t.size),
        "residual_count": 3 * int(train_t.size),
        "train_loss": float(result.info.loss),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_benchmark_error": max_benchmark_error,
        "benchmark_match": benchmark_match,
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
        "benchmark_a_0_gap": abs(float(benchmark_diagnostics["a_0_gap"])),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "train_time": train_time,
        "train_time_success": train_time if solution_accepted else np.nan,
        "compile_time": compile_time,
        "total_params": int(theta_final.size),
        "unknown_count": int(theta_final.size),
        "parameter_norm_initial": float(jnp.linalg.norm(theta_initial)),
        "parameter_norm_final": float(jnp.linalg.norm(theta_final)),
        "parameter_step_norm": float(jnp.linalg.norm(theta_final - theta_initial)),
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
        "train_T": data_set.train_T,
        "num_train_points": int(train_t.size),
        "test_T": data_set.test_T,
        "diagnostic_T": data_set.diagnostic_T,
        "ode_solver": data_set.ode_solver,
        "evaluation_ode_solver": "rk4",
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted" if solution_accepted else ",".join(rejection_reasons)
        ),
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
        "stop_set": asdict(stop_set),
        "initial_conditions": {"k_0": k_0, "a_0": a_0},
    }
    return results, {
        "model": policy,
        "train_df": train_df,
        "test_df": test_df,
        "diagnostic_df": diagnostic_df,
        "policy_df": policy_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(open_economy_growth_nn_recursive)
