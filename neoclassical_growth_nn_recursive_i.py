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
from tinydiffeq import (
    RK4,
    ConstantStepSize,
    Euler,
    SaveAt,
    solve_ode,
)

from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from neural_networks import ACTIVATIONS, unit_softplus

# Recursive (state-space) neoclassical growth solver with log utility: the NN
# approximates gross investment i(k), state = k only. The drift and
# consumption are implied by the resource constraint
#   dk(k) = i(k) - delta k,   c(k) = f(k) - i(k),
# so the investment head is a positive unit-softplus output anchored at the
# replacement level delta k_0, while c > 0 is enforced through penalty rows in
# the residual (log utility needs c > 0). The stationary Euler
# residual is
#   R(k) = c'(k) dk(k) - c(k) (f'(k) - delta - rho_hat),
# with c'(k) = f'(k) - i'(k). Collocation states come from a bounded
# tinydiffeq rollout of the learned policy from k_0, differentiated through
# by the optimizer and sampled on a fixed time grid. The input derivative
# i'(k) uses forward-mode jax.jvp inside the residual while the LM solver
# uses reverse mode over the parameters (reverse-over-forward).

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


@dataclass
class DataSettings:
    # dt is the grid spacing of the test/settle evaluation paths.
    dt: float = 1.0
    ode_solver: Literal["rk4", "euler"] = "rk4"
    # Static internal-step budgets. Fixed solvers use train_T / train_steps.
    train_steps: int = 32
    # A fixed grid keeps the residual shape and scaling constant. The capacity
    # sweep reaches its accuracy plateau at 65 points.
    train_points: int = 65
    tvc_steps: int = 64
    train_T: float = 40.0
    # Auxiliary horizon for the post-hoc transversality certificate. The
    # signal peaks at or just after train_T (capital is bounded by c > 0, so
    # the level term saturates while the discount keeps falling), so a short
    # window past the training horizon suffices.
    train_tvc_T: float = 50.0
    test_T: float = 50.0
    settle_T: float = 400.0  # basin classification horizon (kinked production)
    policy_grid_size: int = 100
    benchmark_T: float = 80.0
    benchmark_points: int = 300


@dataclass(frozen=True)
class StoppingCriteria:
    # Three-seed RK4 calibration converges in 47-62 steps.
    max_steps: int = 250
    # 1e-14 certifies k relative error around 1.4e-6 against the tight
    # benchmark within the default budget.
    train_loss_threshold: float = 1e-14
    num_attempts: int = 1  # single attempt: frac_accepted measures first-try robustness
    # Bar on the log-TVC certificate: 0.0 demands the discounted value of
    # capital in marginal-utility units stays below its O(1) scale at every
    # checked horizon (saddle: -rho_hat t + O(1) < 0; the degenerate branch
    # is far positive).
    log_tvc_threshold: float = 0.0


# Defaults from the Fir HPO sweep winner (kernels/9j88zxne; see
# sweeps/cluster_hpo.md) -- though the sweep found every damping knob flat
# across orders of magnitude; geodesic acceleration is the one that matters.
@dataclass(frozen=True)
class NLLSGramSettings:
    init_damping: float = 3.26e-5
    damping_decrease: float = 0.482
    damping_increase: float = 2.71
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.541


# Runtime i_scale, i_anchor, and k_0_ref are folded in via replace() before
# construction (computed from the initial condition only -- no steady-state or
# kink info).
# The local capacity/collocation sweep found the smooth-model error plateaus at
# width 32, depth 3, and 65 collocation points. Wider 8.5k and 33k parameter
# networks match its accuracy but cost 1.3-3.4x more warm solve time.
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


# The optimizer stops solely on its residual tolerance. The exact log-TVC
# certificate is computed once afterward and never enters training or
# acceptance.
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


def production(k, p):
    base = k**p.a
    return p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


# Marginal product with the branch selected by the (model-intrinsic) kink.
def mpk(k, p):
    base = p.A * p.a * k ** (p.a - 1.0)
    return jnp.where(k > p.k_thr, p.b_1 * base, base)


def consumption(policy, k, p):
    return production(k, p) - policy(k)


def drift(policy, k, p):
    return policy(k) - p.delta * k


def euler_residuals(policy, states, p):
    def residual_scalar(k):
        i, di_dk = jax.jvp(policy, (k,), (jnp.ones_like(k),))
        c = production(k, p) - i
        dk_dt = i - p.delta * k
        dc_dk = mpk(k, p) - di_dk
        return dc_dk * dk_dt - c * (mpk(k, p) - p.delta - p.rho_hat)

    return jax.vmap(residual_scalar)(states)


# c > 0 guard: penalty rows active only where the implied consumption dips
# below the domain floor.
def consumption_positivity(policy, states, p):
    c = jax.vmap(lambda k: consumption(policy, k, p))(states)
    return jnp.minimum(c - DOMAIN_EPS, 0.0)


# Rollouts use tinydiffeq's bounded scans. Training samples the dense solution
# on a fixed grid; the ex-post TVC retains the internal path.
def rollout_field(policy, p):
    return lambda k: drift(policy, k, p)


def log_rollout_field(policy, p):
    def log_capital_drift(q):
        k = p.k_0 * jnp.exp(q)
        return drift(policy, k, p) / k

    return log_capital_drift


def rollout(policy, T, budget, p, data_set, *, save_at=None):
    if save_at is None:
        save_at = SaveAt(steps=True)
    if data_set.ode_solver == "rk4":
        ode_solver = RK4()
        controller = ConstantStepSize()
        dt_0 = T / budget
    elif data_set.ode_solver == "euler":
        ode_solver = Euler()
        controller = ConstantStepSize()
        dt_0 = T / budget
    else:
        raise ValueError(f"unknown ode_solver: {data_set.ode_solver!r}")
    return solve_ode(
        log_rollout_field(policy, p),
        ode_solver,
        0.0,
        T,
        jnp.zeros_like(p.k_0),
        dt_0=dt_0,
        controller=controller,
        max_steps=budget,
        save_at=save_at,
    )


def collocation_states(policy, p, data_set):
    train_ts = jnp.linspace(0.0, data_set.train_T, data_set.train_points)
    sol = rollout(
        policy,
        data_set.train_T,
        data_set.train_steps,
        p,
        data_set,
        save_at=SaveAt(ts=train_ts),
    )
    # A failed rollout poisons every row so LM rejects the trial loudly.
    states = p.k_0 * jnp.exp(sol.xs)
    return jnp.where(sol.ok, states, jnp.inf)


# Post-hoc transversality certificate (log utility): the max over horizons
# t in [T_start, T_end] of
#   log(e^{-rho_hat t} u'(c_t) k_t) = -rho_hat t - log c(k_t) + log k_t.
# On the saddle path this is -rho_hat t + O(1) < 0 at every horizon past the
# training window; a capital bubble grows into the test, and the degenerate
# near-zero-consumption (over-accumulation) branch shows up as a large
# POSITIVE u'(c) k level -- the level itself is the signal, so the residual is
# deliberately NOT normalized. The flat-start parameterization is what selects
# the saddle path (the continuous-time analog of the stock-policy identity
# initialization); this certificate checks that selection ex post.
def log_tvc_residual(policy, T_start, T_end, p, data_set):
    sol = rollout(policy, T_end, data_set.tvc_steps, p, data_set)
    ts = jnp.where(sol.ok, sol.ts, jnp.inf)
    states = p.k_0 * jnp.exp(jnp.where(sol.ok, sol.xs, jnp.inf))
    c = jax.vmap(lambda k: consumption(policy, k, p))(states)
    log_tvc = -p.rho_hat * ts - jnp.log(c) + jnp.log(states)
    valid = sol.accepted & (ts >= T_start)
    return jnp.max(jnp.where(valid, log_tvc, -jnp.inf))


def neoclassical_growth_nn_recursive_i(
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
    lm_set: NLLSGramSettings = NLLSGramSettings(),
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

    # Steady states from f'(k_ss) = delta + rho_hat, per production branch --
    # EX-POST classification/reporting only; nothing enters the net or training.
    def branch_steady_state(branch_factor):
        return ((delta + rho_hat) / (A_value * a * branch_factor)) ** (1.0 / (a - 1.0))

    if use_kinked_production:
        k_ss_low = branch_steady_state(1.0)
        k_ss_high = branch_steady_state(b_1_value)
        steady_targets = [k_ss_low, k_ss_high]
    else:
        k_ss = branch_steady_state(1.0)
        steady_targets = [k_ss]

    # Anchoring and scaling from the initial condition only: the head starts
    # near zero, so the untrained policy is the flat replacement investment
    # i(k_0) = delta k_0 (zero drift); the steady-flow magnitude at k_0 sets
    # the output scale.
    i_scale = max(
        float(A_value * max(k_0**a, b_1_value * k_0**a - b_2_value)) - delta * k_0,
        DOMAIN_EPS,
    )
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
    num_collocation_states = data_set.train_points

    def residual_fn(theta):
        policy = nnx.merge(graphdef, theta)
        states = collocation_states(policy, p, data_set)
        residual = jnp.concatenate(
            [
                euler_residuals(policy, states, p),
                consumption_positivity(policy, states, p),
            ]
        )
        return residual / jnp.sqrt(2 * data_set.train_points)

    solver = LevenbergMarquardt(
        residual_fn,
        linear_solver="cholesky",
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
        dual_solve_dtype=jnp.float64,
    )

    def tvc_certificate(theta):
        return log_tvc_residual(
            nnx.merge(graphdef, theta),
            data_set.train_T,
            data_set.train_tvc_T,
            p,
            data_set,
        )

    _, theta_0 = nnx.split(
        InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(init_key)), nnx.Param
    )
    atol = math.sqrt(stop_set.train_loss_threshold)
    draw = DrawNNXModule(InvestmentPolicyNN, nn_set, dtype=dtype)

    compile_time = np.nan
    if measure_compilation:
        compile_start = time.perf_counter()
        jax.block_until_ready(
            solver.solve(
                theta_0,
                max_steps=1,
                atol=atol,
                multi_start=MultiStart(
                    key=jax.random.key(0),
                    num_starts=stop_set.num_attempts,
                    draw=draw,
                ),
            ).x
        )
        compile_time = time.perf_counter() - compile_start

    start_time = time.perf_counter()
    result = solver.solve(
        theta_0,
        max_steps=stop_set.max_steps,
        atol=atol,
        multi_start=MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=draw,
        ),
    )
    jax.block_until_ready(result.x)
    train_time = time.perf_counter() - start_time
    policy = nnx.merge(graphdef, result.x)
    steps = int(result.steps)
    attempt = int(result.multi_start.attempt) + 1
    stopping_reason = STATUS_TO_REASON.get(int(result.status), "max_steps")
    train_loss = float(result.info.loss)
    max_log_tvc = float(tvc_certificate(result.x))
    transversality_ok = bool(max_log_tvc < stop_set.log_tvc_threshold)
    accepted = bool(result.multi_start.accepted)

    # Test path: simulate the final policy to test_T and evaluate the Euler
    # residual along it.
    n_test_steps = round(data_set.test_T / data_set.dt)
    test_solution = solve_ode(
        rollout_field(policy, p),
        RK4(),
        0.0,
        n_test_steps * data_set.dt,
        p.k_0,
        dt_0=data_set.dt,
        max_steps=n_test_steps,
        save_at=SaveAt(steps=True),
    )
    test_states = jnp.where(test_solution.ok, test_solution.xs, jnp.nan)
    test_i = jax.vmap(policy)(test_states)
    test_c = jax.vmap(lambda k: consumption(policy, k, p))(test_states)
    test_dk = jax.vmap(lambda k: drift(policy, k, p))(test_states)
    test_resid = euler_residuals(policy, test_states, p)
    test_path_ok = bool(
        test_solution.ok
        and jnp.all(jnp.isfinite(test_states))
        and jnp.all(jnp.isfinite(test_c))
        and jnp.min(test_states) > 0.0
        and jnp.min(test_c) > 0.0
    )
    accepted = accepted and test_path_ok
    test_t = np.arange(n_test_steps + 1) * data_set.dt
    test_df = pd.DataFrame(
        {
            "t": test_t,
            "k": np.asarray(test_states),
            "c": np.asarray(test_c),
            "dk": np.asarray(test_dk),
            "i": np.asarray(test_i),
            "euler_residual": np.asarray(test_resid),
        }
    )

    # Policy diagnostics on a k-grid spanning the IC and steady-state targets.
    grid_lo = min([float(k_0)] + steady_targets)
    grid_hi = max([float(k_0)] + steady_targets)
    margin = 0.05 * (grid_hi - grid_lo) + 1e-6
    k_grid = np.linspace(
        max(grid_lo - margin, DOMAIN_EPS),
        grid_hi + margin,
        data_set.policy_grid_size,
    )
    k_grid_jax = jnp.asarray(k_grid, dtype=dtype)
    policy_df = pd.DataFrame(
        {
            "k": k_grid,
            "c_nn": np.asarray(
                jax.vmap(lambda k: consumption(policy, k, p))(k_grid_jax)
            ),
            "dk_nn": np.asarray(jax.vmap(lambda k: drift(policy, k, p))(k_grid_jax)),
            "i_nn": np.asarray(jax.vmap(policy)(k_grid_jax)),
        }
    )

    results = {
        "variant": "i",
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
        "dual_solve_dtype": "float64",
        "train_loss": train_loss,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_resid))),
        "max_log_tvc_residual": max_log_tvc,
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_c)),
        "test_path_ok": test_path_ok,
        "steps": steps,
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "train_time_success": train_time if accepted else np.nan,
        "compile_time": compile_time,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        "solution_accepted": bool(accepted),
        "num_collocation_states": num_collocation_states,
        "ode_solver": data_set.ode_solver,
        "train_steps": data_set.train_steps,
        "dt": data_set.dt,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    if use_kinked_production:
        # Settle the trained policy to settle_T and classify the basin it
        # reaches (ball of 5% of the kink around a steady state).
        n_settle_steps = round(data_set.settle_T / data_set.dt)
        settle_solution = solve_ode(
            rollout_field(policy, p),
            RK4(),
            0.0,
            n_settle_steps * data_set.dt,
            p.k_0,
            dt_0=data_set.dt,
            max_steps=n_settle_steps,
            save_at=SaveAt(steps=True),
        )
        settled = jnp.where(settle_solution.ok, settle_solution.xs, jnp.nan)
        k_settled = float(settled[-1])
        tol = 0.05 * k_thr
        target = k_ss_low if k_0 < k_thr else k_ss_high
        other = k_ss_high if k_0 < k_thr else k_ss_low
        per_correct = abs(k_settled - target) < tol
        per_fp = (not per_correct) and abs(k_settled - other) < tol
        results.update(
            k_thr=k_thr,
            k_ss_low=k_ss_low,
            k_ss_high=k_ss_high,
            k_settled=k_settled,
            basin_correct=bool(per_correct and accepted),
            false_positive=bool(per_fp and accepted),
            basin_failed=bool(not (per_correct or per_fp) or not accepted),
        )
    else:
        results["k_ss"] = k_ss
        # Benchmark path from solve_bvp, interpolated onto the simulated grid.
        benchmark_grid = jnp.linspace(
            0.0, data_set.benchmark_T, data_set.benchmark_points, dtype=dtype
        )
        benchmark_solution = neoclassical_growth_benchmark(
            a, delta, rho_hat, 1.0, k_0, benchmark_grid
        )
        k_benchmark, c_benchmark = benchmark_solution(jnp.asarray(test_t, dtype))
        test_df["k_benchmark"] = np.asarray(k_benchmark)
        test_df["c_benchmark"] = np.asarray(c_benchmark)
        test_df["k_rel_error"] = np.abs(test_df["k"] - test_df["k_benchmark"]) / np.abs(
            test_df["k_benchmark"]
        ).clip(lower=DOMAIN_EPS)
        test_df["c_rel_error"] = np.abs(test_df["c"] - test_df["c_benchmark"]) / np.abs(
            test_df["c_benchmark"]
        ).clip(lower=DOMAIN_EPS)
        results.update(
            test_mean_abs_k_rel_error=float(test_df["k_rel_error"].mean()),
            test_mean_abs_c_rel_error=float(test_df["c_rel_error"].mean()),
        )
        # Benchmark policy by interpolating the (k, c) benchmark path sorted in k.
        k_bench_path, c_bench_path = benchmark_solution(benchmark_grid)
        order = np.argsort(np.asarray(k_bench_path))
        c_benchmark_grid = np.interp(
            k_grid, np.asarray(k_bench_path)[order], np.asarray(c_bench_path)[order]
        )
        policy_df["c_benchmark"] = c_benchmark_grid
        policy_df["dk_benchmark"] = (
            np.asarray(production(k_grid_jax, p)) - delta * k_grid - c_benchmark_grid
        )

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    hyperparameters = {
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
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
        "initial_conditions": {"k_0": k_0},
    }
    return results, {
        "model": policy,
        "test_df": test_df,
        "policy_df": policy_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn_recursive_i)
