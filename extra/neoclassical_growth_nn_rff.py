from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

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
from tinydiffeq import cumulative_trapezoid

from extra.random_fourier_features import KernelName, RFFFeatures
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from utilities import unit_softplus

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
    integration_substeps: int = 4


@dataclass(frozen=True)
class StoppingCriteria:
    # Default MLP seeds converge in 30-67 steps; the cap is a failure budget,
    # not the normal runtime path.
    max_steps: int = 250
    # The integrated ansatz's loss floor is set by the trapezoid path
    # reconstruction, not the optimizer: 1e-12 is unreachable, 1e-9 certifies
    # k rel error ~1.3e-4 (the honest floor at integration_substeps=4;
    # substeps 8 buys ~1.4x at ~1.4x the time, tighter thresholds collapse
    # acceptance).
    train_loss_threshold: float = 1e-9
    # Bar on the post-hoc log-TVC check at test_tvc_T (family convention).
    log_tvc_threshold: float = 0.0
    num_attempts: int = 1  # single attempt: frac_accepted measures first-try robustness


# Local 8-seed grid: geodesic acceleration is a ~5-9x step-count win on the
# integrated ansatz (every geodesic-on damping config beats every off one);
# dec 0.5 / inc 2.0 is the fastest corner, acceptance unchanged.
@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-5
    damping_decrease: float = 0.5
    damping_increase: float = 2.0
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.8


@dataclass(frozen=True)
class GrowthDerivativeSettings:
    width: int = 1024
    output_bias: bool = True
    rff_kernel: KernelName | float = "matern12"
    rff_radius: float = 10.0
    rff_sigma: float = 1.0


class GrowthDerivativeNN(nnx.Module):
    def __init__(
        self,
        settings: GrowthDerivativeSettings,
        c_0_guess: float,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.features = RFFFeatures(
            input_dim=1,
            num_features=settings.width,
            kernel=settings.rff_kernel,
            radius=settings.rff_radius,
            sigma=settings.rff_sigma,
            dtype=dtype,
            rngs=rngs,
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
        self.c_0_scale = c_0_guess
        self.c_0_raw = nnx.Param(jnp.zeros((), dtype=dtype))

    def __call__(self, t):
        h = self.features(jnp.reshape(t, (1, 1)))[0]
        return self.head(h)


def make_train_t_grid(data_set, dtype):
    if data_set.train_points_list is not None:
        values = list(data_set.train_points_list)
    else:
        values = np.linspace(
            0.0, data_set.train_T, data_set.train_points, dtype=np.float64
        ).tolist()
    values = sorted({0.0, *(float(value) for value in values)})
    if values[0] < 0.0:
        raise ValueError("training times must be nonnegative")
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


def policy_values(model, t, p, integration_substeps):
    integral, derivative = cumulative_trapezoid(model, t, substeps=integration_substeps)
    k = p.k_0 + integral[:, 0]
    c_0 = model.c_0_scale * unit_softplus(model.c_0_raw[...])
    c = c_0 + integral[:, 1]
    dk_dt = derivative[:, 0]
    dc_dt = derivative[:, 1]
    output = production(k, p)
    mpk = marginal_product(k, p)
    resource = dk_dt - (output - p.delta * k - c)
    euler = dc_dt - c * (mpk - p.delta - p.rho_hat) / p.sigma_crra
    log_tvc = -p.rho_hat * t - p.sigma_crra * jnp.log(c) + jnp.log(k)
    return {
        "t": t,
        "k": k,
        "c": c,
        "c_0": c_0,
        "dk_dt": dk_dt,
        "dc_dt": dc_dt,
        "Y": output,
        "mpk": mpk,
        "resource": resource,
        "euler": euler,
        "log_tvc": log_tvc,
    }


def training_residual(model, data, p, integration_substeps):
    values = policy_values(model, data["t"], p, integration_substeps)
    k_positivity = jnp.minimum(values["k"] - DOMAIN_EPS, 0.0)
    c_positivity = jnp.minimum(values["c"] - DOMAIN_EPS, 0.0)
    residual = jnp.concatenate(
        [
            values["resource"],
            values["euler"],
            k_positivity,
            c_positivity,
        ]
    )
    return residual / jnp.sqrt(residual.size)


def snapshot(data, model, p, integration_substeps):
    values = jax.device_get(policy_values(model, data["t"], p, integration_substeps))
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


def neoclassical_growth_nn_rff(
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
    nn_set: GrowthDerivativeSettings = GrowthDerivativeSettings(),
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
    integration_substeps = int(data_set.integration_substeps)
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
    derivative_settings = nn_set
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = GrowthDerivativeNN(
        derivative_settings,
        c_flow_guess,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, fixed_state = nnx.split(model, nnx.Param, ...)

    def residual_fn(theta, data):
        return training_residual(
            nnx.merge(graphdef, theta, fixed_state), data, p, integration_substeps
        )

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
        GrowthDerivativeNN,
        derivative_settings,
        c_flow_guess,
        dtype=dtype,
    )

    compile_time = np.nan
    if measure_compilation:
        compile_start = time.perf_counter()
        jax.block_until_ready(
            solver.solve(
                theta_0,
                train_data,
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
        train_data,
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
    model = nnx.merge(graphdef, result.x, fixed_state)
    steps = int(result.steps)
    attempt = int(result.multi_start.attempt) + 1
    stopping_reason = STATUS_TO_REASON.get(int(result.status), "max_steps")
    train_loss = float(result.info.loss)
    accepted = bool(result.multi_start.accepted)

    train_df = snapshot(train_data, model, p, integration_substeps)
    test_df = snapshot(test_data, model, p, integration_substeps)
    train_gaps = np.diff(np.asarray(t_train))
    positive_gaps = train_gaps[train_gaps > 0.0]
    train_spacing = (
        float(np.median(positive_gaps))
        if positive_gaps.size
        else max(float(data_set.train_T), 1.0)
    )
    tvc_intervals = max(1, math.ceil(data_set.test_tvc_T / train_spacing))
    tvc_t = jnp.linspace(0.0, data_set.test_tvc_T, tvc_intervals + 1, dtype=dtype)
    tvc_values = policy_values(model, tvc_t, p, integration_substeps)
    log_tvc = float(tvc_values["log_tvc"][-1])
    tvc_positive = bool(
        jnp.all(jnp.isfinite(tvc_values["k"]))
        and jnp.all(jnp.isfinite(tvc_values["c"]))
        and jnp.min(tvc_values["k"]) > DOMAIN_EPS
        and jnp.min(tvc_values["c"]) > DOMAIN_EPS
    )
    transversality_ok = bool(log_tvc < stop_set.log_tvc_threshold)
    finite_positive = bool(
        np.isfinite(train_df[["k", "c", "resource", "euler"]].to_numpy()).all()
        and np.isfinite(test_df[["k", "c", "resource", "euler"]].to_numpy()).all()
        and (train_df[["k", "c"]].to_numpy() > DOMAIN_EPS).all()
        and (test_df[["k", "c"]].to_numpy() > DOMAIN_EPS).all()
    )

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
        "ansatz": "integrated_derivative",
        "train_loss": train_loss,
        "train_resource_mean_abs": train_resource_mean_abs,
        "train_euler_mean_abs": train_euler_mean_abs,
        "test_resource_mean_abs": test_resource_mean_abs,
        "test_euler_mean_abs": test_euler_mean_abs,
        "max_train_residual": max_train_residual,
        "max_test_residual": max_test_residual,
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "tvc_positive": tvc_positive,
        "finite_positive": finite_positive,
        "steps": steps,
        "attempt": attempt,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "train_time_success": train_time if accepted else np.nan,
        "compile_time": compile_time,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        "approximation": "rff",
        "nn_width": derivative_settings.width,
        "rff_kernel": derivative_settings.rff_kernel,
        "rff_radius": derivative_settings.rff_radius,
        "rff_sigma": derivative_settings.rff_sigma,
        "solution_accepted": bool(accepted),
        "train_T": float(t_train[-1]),
        "num_train_points": int(t_train.shape[0]),
        "test_T": data_set.test_T,
        "test_tvc_T": data_set.test_tvc_T,
        "integration_substeps": integration_substeps,
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
        "nn_set": asdict(derivative_settings),
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
    jsonargparse.CLI(neoclassical_growth_nn_rff)
