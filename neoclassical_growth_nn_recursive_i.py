from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx
from jax import config
from jax.flatten_util import ravel_pytree
from nlls_gram import LMSolveAction, LMStatus, UnderdeterminedLevenbergMarquardt

import integrators
from neoclassical_growth_benchmark import neoclassical_growth_benchmark

# Recursive (state-space) neoclassical growth solver with log utility: the NN
# approximates gross investment i(k), state = k only. The drift and
# consumption are implied by the resource constraint
#   dk(k) = i(k) - delta k,   c(k) = f(k) - i(k),
# so the investment head is a raw linear output anchored at the replacement
# level delta k_0 (no irreversibility constraint imposed; an i >= 0 positivity
# transform is the natural future extension) and c > 0 is enforced through
# penalty rows in the residual (log utility needs c > 0). The stationary Euler
# residual is
#   R(k) = c'(k) dk(k) - c(k) (f'(k) - delta - rho_hat),
# with c'(k) = f'(k) - i'(k). Collocation states come from a free-stepping
# adaptive Tsit5 rollout of the learned policy from k_0
# (integrators.tsit5_free, differentiated through by the optimizer): the
# solver's accepted states ARE the collocation points. The input derivative
# i'(k) uses forward-mode jax.jvp inside the residual while the LM solver
# uses reverse mode over the parameters (reverse-over-forward).

config.update("jax_enable_x64", True)

DOMAIN_EPS = 1e-8

ACTIVATIONS = {
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "softplus": jax.nn.softplus,
    "identity": lambda x: x,
}


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
    # "trajectory": collocation states from the forward-simulated rollout of the
    # current policy. Single dispatch point (collocation_states) so "fixed" /
    # "smolyak" grids can slot in later.
    state_grid: str = "trajectory"
    # dt is the rollout's initial step and the grid spacing of the test /
    # settle evaluation paths (which use fixed-grid RK4 so test_df stays
    # comparable across seeds and against the benchmark).
    dt: float = 1.0
    # Free-stepping adaptive Tsit5 rollout (integrators.tsit5_free): the
    # solver's accepted states ARE the collocation points, steps clipped only
    # at the horizon; adaptive_steps is the fixed attempt budget over
    # [0, train_T]. 1e-6 tolerances already reach the policy-accuracy floor.
    adaptive_steps: int = 32
    tvc_adaptive_steps: int = 64
    rtol: float = 6.3e-7
    atol: float = 2.2e-7
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
    max_steps: int = 1000
    # 1e-14 certifies k rel error ~1.2e-6 against the tight benchmark within
    # the default budget (Fir HPO calibration).
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
# Architecture from the Fir HPO sweep winner: small nets dominate under a
# hard training budget (width 8 certifies ~2x faster than 32 at the same
# accuracy; gelu > silu > tanh on certification rate).
@dataclass(frozen=True)
class InvestmentPolicySettings:
    width: int = 8
    depth: int = 2
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
        # Raw linear head (investment is unconstrained -- reversible for now)
        # anchored at the replacement level i(k_0) = delta k_0, so the
        # near-zero head init makes the untrained policy a flat zero drift at
        # the initial condition.
        h = jnp.atleast_1d((k - self.k_0_ref) / self.k_0_ref)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return self.i_anchor + self.i_scale * self.head(h)[0]


# Early stopping requires BOTH the Euler loss and the exact log-TVC
# certificate below their bars: no transversality penalty enters the training
# loss, the certificate is computed post hoc on the auxiliary simulation.
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}

# The joint loss+TVC convergence gate (and the verbose log) runs on this
# cadence: the certificate rollout is not free.
CERTIFICATE_EVERY_STEPS = 10


def production(k, p):
    k_positive = jnp.maximum(k, DOMAIN_EPS)
    base = k_positive**p.a
    return p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


# Marginal product with the branch selected by the (model-intrinsic) kink.
def mpk(k, p):
    k_positive = jnp.maximum(k, DOMAIN_EPS)
    base = p.A * p.a * k_positive ** (p.a - 1.0)
    return jnp.where(k_positive > p.k_thr, p.b_1 * base, base)


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


# The training rollout and its states live in integrators.py; the policy's own
# accepted states are the collocation points under the free-stepping default.
def rollout_field(policy, p):
    return lambda k: drift(policy, k, p)


def positivity(y):
    return jnp.maximum(y, DOMAIN_EPS)


def collocation_states(policy, p, data_set, dtype):
    if data_set.state_grid != "trajectory":
        raise ValueError(f"Unsupported state_grid: {data_set.state_grid!r}")
    _, states = integrators.tsit5_free(
        rollout_field(policy, p),
        p.k_0,
        data_set.train_T,
        data_set.adaptive_steps,
        rtol=data_set.rtol,
        atol=data_set.atol,
        dt0=data_set.dt,
        project=positivity,
    )
    return states


# Post-hoc transversality certificate (log utility): the max over horizons
# t in [T_start, T_end] of
#   log(e^{-rho_hat t} u'(c_t) k_t) = -rho_hat t - log c(k_t) + log k_t.
# On the saddle path this is -rho_hat t + O(1) < 0 at every horizon past the
# training window; a capital bubble grows into the test, and the degenerate
# near-zero-consumption (over-accumulation) branch shows up as a large
# POSITIVE u'(c) k level -- the level itself is the signal, so the residual is
# deliberately NOT normalized. The flat-start parameterization is what selects
# the saddle path (the continuous-time analog of the stock-policy identity
# initialization); this certificate is the ex-post check of that selection,
# gating early stopping and acceptance.
def log_tvc_residual(policy, T_start, T_end, dt, p, data_set):
    ts, states = integrators.tsit5_free(
        rollout_field(policy, p),
        p.k_0,
        T_end,
        data_set.tvc_adaptive_steps,
        rtol=data_set.rtol,
        atol=data_set.atol,
        dt0=dt,
        project=positivity,
    )
    c = jnp.maximum(jax.vmap(lambda k: consumption(policy, k, p))(states), DOMAIN_EPS)
    log_tvc = -p.rho_hat * ts - jnp.log(c) + jnp.log(states)
    return jnp.max(jnp.where(ts >= T_start, log_tvc, -jnp.inf))


def neoclassical_growth_nn_recursive_i(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    seed: int = 0,
    matmul_highest: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: InvestmentPolicySettings = InvestmentPolicySettings(),
    lm_set: NLLSGramSettings = NLLSGramSettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
    measure_compilation: bool = True,
):
    if matmul_highest:
        config.update("jax_default_matmul_precision", "highest")

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

    dtype = jnp.float64
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
    nn_set = replace(nn_set, i_scale=i_scale, i_anchor=delta * k_0, k_0_ref=k_0)
    rngs = nnx.Rngs(seed)

    graphdef, _ = nnx.split(
        InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(0)), nnx.Param
    )
    num_collocation_states = data_set.adaptive_steps + 1

    def residual_fn(theta, batch):
        policy = nnx.merge(graphdef, theta)
        states = collocation_states(policy, p, data_set, dtype)
        residual = jnp.concatenate(
            [
                euler_residuals(policy, states, p),
                consumption_positivity(policy, states, p),
            ]
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

    @jax.jit
    def tvc_certificate(theta):
        return log_tvc_residual(
            nnx.merge(graphdef, theta),
            data_set.train_T,
            data_set.train_tvc_T,
            data_set.dt,
            p,
            data_set,
        )

    def stopping_callback(ctx):
        # Joint loss+TVC convergence gate on the certificate cadence;
        # divergence is checked every step.
        def certificate_checks(_):
            log_tvc = log_tvc_residual(
                nnx.merge(graphdef, ctx.x),
                data_set.train_T,
                data_set.train_tvc_T,
                data_set.dt,
                p,
                data_set,
            )
            converged = (ctx.info.loss < stop_set.train_loss_threshold) & (
                log_tvc < stop_set.log_tvc_threshold
            )
            if verbose:
                jax.debug.print(
                    "step {step}: train_loss={loss:.3e} log_tvc={tvc:.2f} "
                    "damping={damping:.1e}",
                    step=ctx.step,
                    loss=ctx.info.loss,
                    tvc=log_tvc,
                    damping=ctx.lm_state.damping,
                )
            return converged

        converged = jax.lax.cond(
            ctx.step % CERTIFICATE_EVERY_STEPS == 0,
            certificate_checks,
            lambda _: jnp.asarray(False),
            operand=None,
        )
        diverged = ~jnp.isfinite(ctx.info.loss_candidate)
        status = jnp.where(converged, LMStatus.CONVERGED, LMStatus.NONFINITE)
        return LMSolveAction(stop=converged | diverged, status=status)

    compile_time = np.nan
    if measure_compilation:
        _, template_theta = nnx.split(
            InvestmentPolicyNN(nn_set, dtype=dtype, rngs=nnx.Rngs(0)), nnx.Param
        )
        compile_start = time.perf_counter()
        jax.block_until_ready(
            solver.solve(
                template_theta,
                max_steps=1,
                callback=stopping_callback,
            )
        )
        compile_time = time.perf_counter() - compile_start

    accepted = False
    max_log_tvc = np.nan
    transversality_ok = False
    for attempt in range(1, stop_set.num_attempts + 1):
        policy = InvestmentPolicyNN(nn_set, dtype=dtype, rngs=rngs)
        _, theta = nnx.split(policy, nnx.Param)
        if verbose and stop_set.num_attempts > 1:
            print(f"attempt {attempt}")

        start_time = time.perf_counter()
        result = solver.solve(
            theta,
            max_steps=stop_set.max_steps,
            callback=stopping_callback,
        )
        theta, info, steps = result.x, result.info, int(result.steps)
        stopping_reason = STATUS_TO_REASON.get(int(result.status), "max_steps")
        train_loss = float(info.loss)
        if not np.isfinite(train_loss):
            train_loss = float(info.loss_old)
        max_log_tvc = float(tvc_certificate(theta))
        train_time = time.perf_counter() - start_time
        policy = nnx.merge(graphdef, theta)

        # early_stopping_met already required the certificate below its bar.
        transversality_ok = bool(max_log_tvc < stop_set.log_tvc_threshold)
        accepted = stopping_reason == "early_stopping_met"
        if accepted:
            break

    # Test path: simulate the final policy to test_T and evaluate the Euler
    # residual along it.
    n_test_steps = round(data_set.test_T / data_set.dt)
    test_states = integrators.rk4_grid(
        rollout_field(policy, p),
        p.k_0,
        n_test_steps,
        jnp.asarray(data_set.dt, dtype),
        project=positivity,
    )
    test_i = jax.vmap(policy)(test_states)
    test_c = jax.vmap(lambda k: consumption(policy, k, p))(test_states)
    test_dk = jax.vmap(lambda k: drift(policy, k, p))(test_states)
    test_resid = euler_residuals(policy, test_states, p)
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
    k_grid = np.linspace(grid_lo - margin, grid_hi + margin, data_set.policy_grid_size)
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
        "train_loss": train_loss,
        "test_euler_mean_abs": float(jnp.mean(jnp.abs(test_resid))),
        "max_log_tvc_residual": max_log_tvc,
        "transversality_ok": transversality_ok,
        "min_test_c": float(jnp.min(test_c)),
        "steps": steps,
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "train_time_success": train_time if accepted else np.nan,
        "compile_time": compile_time,
        "total_params": int(ravel_pytree(nnx.state(policy, nnx.Param))[0].size),
        "solution_accepted": bool(accepted),
        "num_collocation_states": num_collocation_states,
        "dt": data_set.dt,
        "train_T": data_set.train_T,
        "test_T": data_set.test_T,
        "train_tvc_T": data_set.train_tvc_T,
    }

    if use_kinked_production:
        # Settle the trained policy to settle_T and classify the basin it
        # reaches (ball of 5% of the kink around a steady state).
        n_settle_steps = round(data_set.settle_T / data_set.dt)
        settled = integrators.rk4_grid(
            rollout_field(policy, p),
            p.k_0,
            n_settle_steps,
            jnp.asarray(data_set.dt, dtype),
            project=positivity,
        )
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
