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
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    DrawNNXModule,
    LevenbergMarquardt,
    LMStatus,
    MultiStart,
)
from tinydiffeq import IController, Rodas5P, SaveAt, Tsit5, solve_ode

from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from utilities import ACTIVATIONS, unit_softplus

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

DOMAIN_EPS = 1e-8


@dataclass(frozen=True)
class StructuralParameters:
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    k_0: jax.Array
    A: jax.Array
    b_1: jax.Array
    b_2: jax.Array
    k_thr: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 3.26e-5
    damping_decrease: float = 0.482
    damping_increase: float = 2.71
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.541


@dataclass(frozen=True)
class InvestmentPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "gelu"
    i_scale: float = 1.0
    i_anchor: float = 0.0
    k_0_ref: float = 1.0


class InvestmentPolicyNN(nnx.Module):
    def __init__(self, settings: InvestmentPolicySettings, *, dtype, rngs: nnx.Rngs):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.i_scale = settings.i_scale
        self.i_anchor = settings.i_anchor
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
            1,
            kernel_init=nnx.initializers.normal(1e-3),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, k):
        h = jnp.atleast_1d(jnp.log(k / self.k_0_ref))
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        raw = self.head(h)[0]
        return self.i_anchor * unit_softplus((self.i_scale / self.i_anchor) * raw)


def production(k, p):
    base = k**p.a
    return p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


def mpk(k, p):
    base = p.A * p.a * k ** (p.a - 1.0)
    return jnp.where(k > p.k_thr, p.b_1 * base, base)


def consumption(policy, k, p):
    return production(k, p) - policy(k)


def drift(policy, k, p):
    return policy(k) - p.delta * k


def euler_residuals(policy, states, p):
    def residual_scalar(k):
        investment, investment_derivative = jax.jvp(policy, (k,), (jnp.ones_like(k),))
        c = production(k, p) - investment
        dk_dt = investment - p.delta * k
        dc_dk = mpk(k, p) - investment_derivative
        return dc_dk * dk_dt - c * (mpk(k, p) - p.delta - p.rho_hat)

    return jax.vmap(residual_scalar)(states)


def consumption_positivity(policy, states, p):
    c = jax.vmap(lambda k: consumption(policy, k, p))(states)
    return jnp.minimum(c - DOMAIN_EPS, 0.0)


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 100.0
    ode_solver: Literal["tsit5", "rodas5p"] = "tsit5"
    adaptive_steps: int | None = None
    dt: float = 0.25
    step_growth_max: float = 3.0
    rtol: float = 1e-8
    atol: float = 1e-10
    test_T: float = 100.0
    test_points: int = 401
    test_steps: int = 128
    settle_T: float = 400.0
    train_tvc_T: float = 200.0
    tvc_points: int = 129
    tvc_steps: int = 128
    benchmark_T: float = 200.0
    benchmark_points: int = 800
    policy_grid_size: int = 100


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 250
    train_loss_threshold: float = 1e-14
    num_attempts: int = 1
    log_tvc_threshold: float = 0.0


STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


def ode_components(data_set):
    controller = IController(
        rtol=data_set.rtol,
        atol=data_set.atol,
        dt_min=1e-10,
        factor_max=data_set.step_growth_max,
    )
    if data_set.ode_solver == "tsit5":
        return Tsit5(), controller
    if data_set.ode_solver == "rodas5p":
        return Rodas5P(), controller
    raise ValueError(f"unknown ode_solver: {data_set.ode_solver!r}")


def training_attempt_budget(data_set):
    if data_set.adaptive_steps is not None:
        return data_set.adaptive_steps
    return 64 if data_set.ode_solver == "tsit5" else 128


def adaptive_rollout(policy, T, budget, p, data_set, save_at):
    ode_solver, controller = ode_components(data_set)

    def log_capital_drift(log_kappa):
        k = p.k_0 * jnp.exp(log_kappa)
        return drift(policy, k, p) / k

    return solve_ode(
        log_capital_drift,
        ode_solver,
        0.0,
        T,
        jnp.zeros_like(p.k_0),
        dt_0=data_set.dt,
        controller=controller,
        max_steps=budget,
        save_at=save_at,
    )


def adaptive_collocation(policy, p, data_set):
    budget = training_attempt_budget(data_set)
    solution = adaptive_rollout(
        policy,
        data_set.train_T,
        budget,
        p,
        data_set,
        SaveAt(steps=True),
    )
    states = p.k_0 * jnp.exp(jnp.where(solution.ok, solution.xs, jnp.inf))
    return states, solution.ts, solution.accepted, solution.ok


def adaptive_residuals(policy, p, data_set):
    states, _, active, rollout_ok = adaptive_collocation(policy, p, data_set)
    euler = euler_residuals(policy, states, p)
    domain = consumption_positivity(policy, states, p)
    active_scale = jnp.sqrt(2 * jnp.maximum(jnp.sum(active), 1))
    residual = (
        jnp.concatenate(
            [
                jnp.where(active, euler, 0.0),
                jnp.where(active, domain, 0.0),
            ]
        )
        / active_scale
    )
    return jnp.where(rollout_ok, residual, jnp.inf)


def neoclassical_growth_nn_recursive_adaptive(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: InvestmentPolicySettings = InvestmentPolicySettings(),
    lm_set: SolverSettings = SolverSettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
    measure_compilation: bool = True,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    kinked_parameter_count = sum(parameter is not None for parameter in (A, b_1, b_2))
    if kinked_parameter_count not in {0, 3}:
        raise ValueError("Set all of A, b_1, and b_2, or set none of them.")
    use_kinked_production = kinked_parameter_count == 3
    A_value = 1.0 if A is None else float(A)
    b_1_value = 1.0 if b_1 is None else float(b_1)
    b_2_value = 0.0 if b_2 is None else float(b_2)
    k_thr = (
        (b_2_value / (b_1_value - 1.0)) ** (1.0 / a)
        if use_kinked_production
        else float("inf")
    )
    dtype = jnp.float64 if use_float64 else jnp.float32
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        k_0=k_0,
        A=A_value,
        b_1=b_1_value,
        b_2=b_2_value,
        k_thr=k_thr,
    )

    def branch_steady_state(branch_factor):
        return ((delta + rho_hat) / (A_value * a * branch_factor)) ** (1.0 / (a - 1.0))

    if use_kinked_production:
        k_ss_low = branch_steady_state(1.0)
        k_ss_high = branch_steady_state(b_1_value)
        steady_targets = (k_ss_low, k_ss_high)
    else:
        k_ss = branch_steady_state(1.0)
        steady_targets = (k_ss,)

    output_0 = A_value * max(k_0**a, b_1_value * k_0**a - b_2_value)
    i_scale = max(output_0 - delta * k_0, DOMAIN_EPS)
    nn_set = replace(
        nn_set,
        i_scale=i_scale,
        i_anchor=max(delta * k_0, DOMAIN_EPS),
        k_0_ref=k_0,
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    graphdef, _ = nnx.split(
        InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(0)), nnx.Param
    )

    def residual_fn(theta):
        return adaptive_residuals(nnx.merge(graphdef, theta), p, data_set)

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
        linear_solve_dtype=jnp.float64,
    )
    _, theta_0 = nnx.split(
        InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key)), nnx.Param
    )
    draw = DrawNNXModule(InvestmentPolicyNN, nn_set, dtype=dtype)
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=stop_set.num_attempts,
        draw=draw,
    )
    atol = math.sqrt(stop_set.train_loss_threshold)

    compile_time = np.nan
    if measure_compilation:
        compile_start = time.perf_counter()
        compiled = solver.solve(
            theta_0,
            max_steps=1,
            atol=atol,
            multi_start=MultiStart(
                key=jax.random.key(0),
                num_starts=stop_set.num_attempts,
                draw=draw,
            ),
            max_steps_is_success=False,
        )
        jax.block_until_ready(compiled.x)
        compile_time = time.perf_counter() - compile_start

    start_time = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=stop_set.max_steps,
        atol=atol,
        multi_start=multi_start,
        max_steps_is_success=False,
    )
    jax.block_until_ready(result.x)
    train_time = time.perf_counter() - start_time
    policy = nnx.merge(graphdef, result.x)

    collocation_states, collocation_times, active, collocation_ok = (
        adaptive_collocation(policy, p, data_set)
    )
    train_attempt_budget = training_attempt_budget(data_set)
    accepted_times = np.asarray(collocation_times)[np.asarray(active)]
    accepted_time_steps = np.diff(accepted_times)
    min_accepted_time_step = (
        float(np.min(accepted_time_steps)) if accepted_time_steps.size else np.nan
    )
    max_accepted_time_step = (
        float(np.max(accepted_time_steps)) if accepted_time_steps.size else np.nan
    )

    test_ts = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    test_solution = adaptive_rollout(
        policy,
        data_set.test_T,
        data_set.test_steps,
        p,
        data_set,
        SaveAt(ts=test_ts),
    )
    test_states = p.k_0 * jnp.exp(
        jnp.where(test_solution.ok, test_solution.xs, jnp.nan)
    )
    test_i = jax.vmap(policy)(test_states)
    test_c = jax.vmap(lambda k: consumption(policy, k, p))(test_states)
    test_dk = jax.vmap(lambda k: drift(policy, k, p))(test_states)
    test_euler = euler_residuals(policy, test_states, p)
    test_euler_mean_abs = float(jnp.mean(jnp.abs(test_euler)))
    test_path_ok = bool(
        test_solution.ok
        and jnp.all(jnp.isfinite(test_states))
        and jnp.all(jnp.isfinite(test_c))
        and jnp.min(test_states) > 0.0
        and jnp.min(test_c) > 0.0
    )

    tvc_ts = jnp.linspace(
        data_set.train_T, data_set.train_tvc_T, data_set.tvc_points, dtype=dtype
    )
    tvc_solution = adaptive_rollout(
        policy,
        data_set.train_tvc_T,
        data_set.tvc_steps,
        p,
        data_set,
        SaveAt(ts=tvc_ts),
    )
    tvc_states = p.k_0 * jnp.exp(jnp.where(tvc_solution.ok, tvc_solution.xs, jnp.nan))
    tvc_c = jax.vmap(lambda k: consumption(policy, k, p))(tvc_states)
    log_tvc = -p.rho_hat * tvc_ts - jnp.log(tvc_c) + jnp.log(tvc_states)
    max_log_tvc = float(jnp.nanmax(log_tvc))
    terminal_log_tvc = float(log_tvc[-1])
    log_tvc_tail_change = float(log_tvc[-1] - log_tvc[0])
    tvc_tail_declining = bool(log_tvc_tail_change < 0.0)
    transversality_ok = bool(
        tvc_solution.ok
        and max_log_tvc < stop_set.log_tvc_threshold
        and tvc_tail_declining
    )

    test_df = pd.DataFrame(
        {
            "t": np.asarray(test_ts),
            "k": np.asarray(test_states),
            "c": np.asarray(test_c),
            "dk": np.asarray(test_dk),
            "i": np.asarray(test_i),
            "euler_residual": np.asarray(test_euler),
        }
    )

    k_grid = jnp.linspace(
        max(min(k_0, *steady_targets) * 0.9, DOMAIN_EPS),
        max(k_0, *steady_targets) * 1.1,
        data_set.policy_grid_size,
        dtype=dtype,
    )
    policy_i = jax.vmap(policy)(k_grid)
    policy_c = jax.vmap(lambda k: consumption(policy, k, p))(k_grid)
    policy_df = pd.DataFrame(
        {
            "k": np.asarray(k_grid),
            "i_nn": np.asarray(policy_i),
            "c_nn": np.asarray(policy_c),
            "dk_nn": np.asarray(policy_i - delta * k_grid),
        }
    )

    status = int(result.status)
    solution_accepted = bool(result.multi_start.accepted)
    results = {
        "variant": "i_adaptive",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "k_0": k_0,
        "A": A_value,
        "b_1": b_1_value,
        "b_2": b_2_value,
        "use_kinked_production": use_kinked_production,
        "seed": seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "linear_solve_dtype": "float64",
        "metric": "identity_parameter_metric",
        "selection": "minimum_identity_metric_parameter_step",
        "train_loss": float(result.info.loss),
        "test_euler_mean_abs": test_euler_mean_abs,
        "min_test_c": float(jnp.min(test_c)),
        "test_path_ok": test_path_ok,
        "max_log_tvc_residual": max_log_tvc,
        "terminal_log_tvc_residual": terminal_log_tvc,
        "log_tvc_tail_change": log_tvc_tail_change,
        "tvc_tail_declining": tvc_tail_declining,
        "transversality_ok": transversality_ok,
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(status, "max_steps"),
        "solution_accepted": solution_accepted,
        "train_time": train_time,
        "train_time_success": train_time if solution_accepted else np.nan,
        "compile_time": compile_time,
        "attempt": int(result.multi_start.attempt) + 1,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        "num_collocation_states": int(jnp.sum(active)),
        "collocation_capacity": train_attempt_budget + 1,
        "integrator_attempt_budget": train_attempt_budget,
        "collocation_path_ok": bool(collocation_ok),
        "min_accepted_time_step": min_accepted_time_step,
        "max_accepted_time_step": max_accepted_time_step,
        "ode_solver": data_set.ode_solver,
        "step_growth_max": data_set.step_growth_max,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    if use_kinked_production:
        settle_ts = jnp.asarray([data_set.settle_T], dtype=dtype)
        settle_solution = adaptive_rollout(
            policy,
            data_set.settle_T,
            data_set.test_steps,
            p,
            data_set,
            SaveAt(ts=settle_ts),
        )
        k_settled = float(
            p.k_0
            * jnp.exp(jnp.where(settle_solution.ok, settle_solution.xs[-1], jnp.nan))
        )
        basin_tolerance = 0.05 * k_thr
        target = k_ss_low if k_0 < k_thr else k_ss_high
        other = k_ss_high if k_0 < k_thr else k_ss_low
        basin_correct = abs(k_settled - target) < basin_tolerance
        false_positive = not basin_correct and abs(k_settled - other) < basin_tolerance
        results.update(
            k_thr=k_thr,
            k_ss_low=k_ss_low,
            k_ss_high=k_ss_high,
            k_settled=k_settled,
            basin_correct=bool(basin_correct),
            false_positive=bool(false_positive),
            basin_failed=bool(not (basin_correct or false_positive)),
        )
    else:
        results["k_ss"] = k_ss
        benchmark_grid = jnp.linspace(
            0.0,
            data_set.benchmark_T,
            data_set.benchmark_points,
            dtype=dtype,
        )
        benchmark_solution = neoclassical_growth_benchmark(
            a, delta, rho_hat, 1.0, k_0, benchmark_grid
        )
        k_benchmark, c_benchmark = benchmark_solution(test_ts)
        k_relative_error = jnp.abs(test_states - k_benchmark) / jnp.maximum(
            jnp.abs(k_benchmark), DOMAIN_EPS
        )
        c_relative_error = jnp.abs(test_c - c_benchmark) / jnp.maximum(
            jnp.abs(c_benchmark), DOMAIN_EPS
        )
        test_df["k_benchmark"] = np.asarray(k_benchmark)
        test_df["c_benchmark"] = np.asarray(c_benchmark)
        test_df["k_rel_error"] = np.asarray(k_relative_error)
        test_df["c_rel_error"] = np.asarray(c_relative_error)
        results.update(
            test_mean_abs_k_rel_error=float(jnp.mean(k_relative_error)),
            test_mean_abs_c_rel_error=float(jnp.mean(c_relative_error)),
        )

        k_benchmark_path, c_benchmark_path = benchmark_solution(benchmark_grid)
        order = np.argsort(np.asarray(k_benchmark_path))
        c_benchmark_grid = np.interp(
            np.asarray(k_grid),
            np.asarray(k_benchmark_path)[order],
            np.asarray(c_benchmark_path)[order],
        )
        policy_df["c_benchmark"] = c_benchmark_grid
        policy_df["dk_benchmark"] = (
            np.asarray(production(k_grid, p))
            - delta * np.asarray(k_grid)
            - c_benchmark_grid
        )
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    return results, {
        "model": policy,
        "parameters": p,
        "collocation_states": collocation_states,
        "collocation_times": collocation_times,
        "collocation_active": active,
        "test_df": test_df,
        "policy_df": policy_df,
        "hyperparameters": {
            "data_set": asdict(data_set),
            "nn_set": asdict(nn_set),
            "lm_set": asdict(lm_set),
            "stop_set": asdict(stop_set),
            "parameters": {
                "a": a,
                "delta": delta,
                "rho_hat": rho_hat,
                "A": A_value,
                "b_1": b_1_value,
                "b_2": b_2_value,
            },
        },
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn_recursive_adaptive)
