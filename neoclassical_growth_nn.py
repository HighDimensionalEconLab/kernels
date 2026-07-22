from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

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
    sigma_crra: jax.Array
    k_0: jax.Array
    A: jax.Array
    b_1: jax.Array
    b_2: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 101
    test_tvc_T: float = 200.0
    benchmark_T: float = 80.0
    benchmark_points: int = 300


@dataclass(frozen=True)
class StoppingCriteria:
    # Three-seed calibration converges in 141-181 steps at the default
    # tolerance, leaving a useful margin without a 1,000-step runaway budget.
    max_steps: int = 250
    # 1e-15 is reachable in <1 s at the default architecture and buys ~11x
    # accuracy over 1e-12 (k rel error ~1.3e-6 vs ~1.5e-5 against the tight
    # solve_bvp benchmark). w32 d3 tanh is right-sized: smaller/shallower
    # nets and gelu all measure worse here.
    train_loss_threshold: float = 1e-15
    # Bar on the post-hoc log-TVC check at test_tvc_T: 0.0 demands the
    # discounted value of capital in marginal-utility units is below its O(1)
    # scale (saddle: -rho_hat t + O(1) << 0; a bubble is far positive).
    log_tvc_threshold: float = 0.0
    num_attempts: int = 1  # single attempt: frac_accepted measures first-try robustness


# Local 8-seed grid: dec 0.5 / inc 2.0 is ~25% faster than 0.35/8.0 at the
# same 8/8 acceptance; geodesic acceleration is required (off fails to reach
# the loss threshold at any swept damping).
@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-5
    damping_decrease: float = 0.5
    damping_increase: float = 2.0
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.8


@dataclass(frozen=True)
class GrowthPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    hidden_bias: bool = True
    output_bias: bool = True
    input_scale: float = 40.0


class GrowthPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: GrowthPolicySettings,
        *,
        k_scale: float,
        c_scale: float,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.input_scale = settings.input_scale
        self.k_scale = k_scale
        self.c_scale = c_scale
        self.layers = nnx.List(
            [
                nnx.Linear(
                    1,
                    settings.width,
                    use_bias=settings.hidden_bias,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(settings.depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.width,
                    settings.width,
                    use_bias=settings.hidden_bias,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.head = nnx.Linear(
            settings.width,
            2,
            use_bias=settings.output_bias,
            kernel_init=nnx.initializers.normal(1e-3),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, t):
        h = jnp.atleast_1d(t / self.input_scale)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        output_scale = jnp.asarray(
            [self.k_scale, self.c_scale], dtype=self.head.kernel.dtype
        )
        return output_scale * unit_softplus(self.head(h))


def make_train_t_grid(data_set, dtype):
    if data_set.train_points_list is not None:
        values = list(data_set.train_points_list)
    else:
        values = np.linspace(
            0.0, data_set.train_T, data_set.train_points, dtype=np.float64
        ).tolist()
    values = sorted(set(float(value) for value in values))
    return jnp.asarray(values, dtype=dtype)


STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


def production(k, p):
    base = k**p.a
    return p.A * jnp.maximum(base, p.b_1 * base - p.b_2)


def marginal_product(k, p):
    return jax.vmap(jax.grad(lambda x: production(x, p)))(k)


def policy_values(model, t, p):
    values, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
        t
    )
    k = values[:, 0]
    c = values[:, 1]
    dk_dt = derivatives[:, 0]
    dc_dt = derivatives[:, 1]
    output = production(k, p)
    mpk = marginal_product(k, p)
    resource = dk_dt - (output - p.delta * k - c)
    euler = dc_dt - c * (mpk - p.delta - p.rho_hat) / p.sigma_crra
    log_tvc = -p.rho_hat * t - p.sigma_crra * jnp.log(c) + jnp.log(k)
    return {
        "t": t,
        "k": k,
        "c": c,
        "Y": output,
        "mpk": mpk,
        "resource": resource,
        "euler": euler,
        "log_tvc": log_tvc,
    }


def training_residual(model, data, p):
    values = policy_values(model, data["t"], p)
    ic = model(jnp.asarray(0.0, dtype=data["t"].dtype))[0] - p.k_0
    residual = jnp.concatenate(
        [
            values["resource"],
            values["euler"],
            jnp.atleast_1d(ic),
        ]
    )
    return residual / jnp.sqrt(residual.size)


def snapshot(data, model, p):
    values = jax.device_get(policy_values(model, data["t"], p))
    columns = {key: np.asarray(value) for key, value in values.items()}
    df = pd.DataFrame(columns)
    if "k_benchmark" in data:
        k_bar = np.asarray(jax.device_get(data["k_benchmark"]))
        c_bar = np.asarray(jax.device_get(data["c_benchmark"]))
        df["k_benchmark"] = k_bar
        df["c_benchmark"] = c_bar
        df["k_rel_error"] = np.abs(df["k"] - k_bar) / np.maximum(
            np.abs(k_bar), DOMAIN_EPS
        )
        df["c_rel_error"] = np.abs(df["c"] - c_bar) / np.maximum(
            np.abs(c_bar), DOMAIN_EPS
        )
    return df


def neoclassical_growth_nn(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    sigma_crra: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: GrowthPolicySettings = GrowthPolicySettings(),
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

    dtype = jnp.float64 if use_float64 else jnp.float32
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        k_0=k_0,
        A=A_value,
        b_1=b_1_value,
        b_2=b_2_value,
    )

    t_train = make_train_t_grid(data_set, dtype)
    train_data = {"t": t_train}
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    test_data = {"t": test_t}
    if not use_kinked_production:
        benchmark_grid = jnp.linspace(
            0.0, data_set.benchmark_T, data_set.benchmark_points, dtype=dtype
        )
        benchmark_solution = neoclassical_growth_benchmark(
            a, delta, rho_hat, sigma_crra, k_0, benchmark_grid
        )
        k_train_bar, c_train_bar = benchmark_solution(t_train)
        k_test_bar, c_test_bar = benchmark_solution(test_t)
        train_data = {
            **train_data,
            "k_benchmark": k_train_bar,
            "c_benchmark": c_train_bar,
        }
        test_data = {
            **test_data,
            "k_benchmark": k_test_bar,
            "c_benchmark": c_test_bar,
        }

    initial_output = float(A_value * max(k_0**a, b_1_value * k_0**a - b_2_value))
    c_flow_guess = max(initial_output - delta * k_0, DOMAIN_EPS)
    policy_settings = replace(nn_set, input_scale=max(float(data_set.train_T), 1.0))
    k_scale = max(float(k_0), DOMAIN_EPS)
    c_scale = c_flow_guess
    graphdef, _ = nnx.split(
        GrowthPolicyNN(
            policy_settings,
            k_scale=k_scale,
            c_scale=c_scale,
            dtype=dtype,
            rngs=nnx.Rngs(0),
        ),
        nnx.Param,
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = GrowthPolicyNN(
        policy_settings,
        k_scale=k_scale,
        c_scale=c_scale,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    _, theta_0 = nnx.split(model, nnx.Param)

    def residual_fn(theta, t):
        return training_residual(nnx.merge(graphdef, theta), {"t": t}, p)

    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
        linear_solve_dtype=jnp.float64,
    )

    # solve()'s atol is on the residual norm; the settings threshold is on
    # the sum of squared residuals. Stopping is entirely solve()'s: atol for
    # convergence, max_steps as the budget (a nonfinite candidate is rejected
    # and damping raised, which recovers more often than aborting).
    atol = math.sqrt(stop_set.train_loss_threshold)
    draw = DrawNNXModule(
        GrowthPolicyNN,
        policy_settings,
        k_scale=k_scale,
        c_scale=c_scale,
        dtype=dtype,
    )

    compile_time = np.nan
    if measure_compilation:
        compile_start = time.perf_counter()
        jax.block_until_ready(
            solver.solve(
                theta_0,
                t_train,
                max_steps=1,
                atol=atol,
                multi_start=MultiStart(
                    key=jax.random.key(0),
                    num_starts=stop_set.num_attempts,
                    draw=draw,
                ),
                max_steps_is_success=False,
            ).x
        )
        compile_time = time.perf_counter() - compile_start

    start_time = time.perf_counter()
    result = solver.solve(
        theta_0,
        t_train,
        max_steps=stop_set.max_steps,
        atol=atol,
        multi_start=MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=draw,
        ),
        max_steps_is_success=False,
    )
    jax.block_until_ready(result.x)
    train_time = time.perf_counter() - start_time
    model = nnx.merge(graphdef, result.x)
    steps = int(result.steps)
    attempt = int(result.multi_start.attempt) + 1
    stopping_reason = STATUS_TO_REASON.get(int(result.status), "max_steps")
    train_loss = float(result.info.loss)
    accepted = bool(result.multi_start.accepted)

    train_df = snapshot(train_data, model, p)
    test_df = snapshot(test_data, model, p)
    tvc_t = jnp.asarray(data_set.test_tvc_T, dtype=dtype)
    tvc_values = policy_values(model, jnp.atleast_1d(tvc_t), p)
    log_tvc = float(tvc_values["log_tvc"][0])
    transversality_ok = bool(log_tvc < stop_set.log_tvc_threshold)

    train_resource_mean_abs = float(train_df["resource"].abs().mean())
    train_euler_mean_abs = float(train_df["euler"].abs().mean())
    test_resource_mean_abs = float(test_df["resource"].abs().mean())
    test_euler_mean_abs = float(test_df["euler"].abs().mean())
    max_train_residual = float(
        max(train_df["resource"].abs().max(), train_df["euler"].abs().max())
    )
    max_test_residual = float(
        max(test_df["resource"].abs().max(), test_df["euler"].abs().max())
    )

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "k_0": k_0,
        "A": A_value,
        "b_1": b_1_value,
        "b_2": b_2_value,
        "use_kinked_production": use_kinked_production,
        "seed": seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "linear_solve_dtype": "float64",
        "solver": "nlls-gram",
        "train_loss": train_loss,
        "train_resource_mean_abs": train_resource_mean_abs,
        "train_euler_mean_abs": train_euler_mean_abs,
        "test_resource_mean_abs": test_resource_mean_abs,
        "test_euler_mean_abs": test_euler_mean_abs,
        "max_train_residual": max_train_residual,
        "max_test_residual": max_test_residual,
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "steps": steps,
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "train_time_success": train_time if accepted else np.nan,
        "compile_time": compile_time,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        "nn_width": policy_settings.width,
        "nn_depth": policy_settings.depth,
        "solution_accepted": bool(accepted),
        "train_T": float(t_train[-1]),
        "num_train_points": int(t_train.shape[0]),
        "test_T": data_set.test_T,
        "test_tvc_T": data_set.test_tvc_T,
    }
    if not use_kinked_production:
        results.update(
            test_mean_abs_k_rel_error=float(test_df["k_rel_error"].mean()),
            test_mean_abs_c_rel_error=float(test_df["c_rel_error"].mean()),
            train_mean_abs_k_rel_error=float(train_df["k_rel_error"].mean()),
            train_mean_abs_c_rel_error=float(train_df["c_rel_error"].mean()),
        )

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    hyperparameters = {
        "nn_set": asdict(policy_settings),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "stop_set": asdict(stop_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "A": A_value,
            "b_1": b_1_value,
            "b_2": b_2_value,
        },
        "initial_conditions": {"k_0": k_0},
    }
    return results, {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_nn)
