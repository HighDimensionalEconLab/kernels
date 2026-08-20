from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    DrawNNXModule,
    LevenbergMarquardt,
    MultiStart,
)

from models.neoclassical_growth import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth import (
    StructuralParameters,
    mpk,
    production,
    steady_flow_consumption_guess,
)
from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.utilities import (
    ACTIVATIONS,
    OptimizerSettings,
    Stopwatch,
    make_train_t_grid,
    unit_softplus,
)


@dataclass
class DataSettings:
    train_T: float = CONSTANTS.train_T
    train_points: int = 101
    train_points_list: list[float] | None = None
    test_T: float = CONSTANTS.sequence_test_T
    test_points: int = 101
    test_tvc_T: float = CONSTANTS.transversality_T


@dataclass(frozen=True)
class ToleranceSettings:
    max_steps: int = 250
    train_loss_threshold: float = 1e-15
    log_tvc_threshold: float = 0.0
    num_attempts: int = 5


@dataclass(frozen=True)
class GrowthPolicySettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    hidden_bias: bool = True
    output_bias: bool = True
    input_scale: float = 40.0
    # Trend e^{g_hat t} on both heads: a learned Param (init g_hat_init) or
    # fixed at the constructor's g_hat.
    learn_g_hat: bool = False
    g_hat_init: float = 0.0


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    nondiff: nnx.State
    t: jax.Array
    z_trend: jax.Array


class GrowthPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: GrowthPolicySettings,
        *,
        k_scale: float,
        c_scale: float,
        g_hat: float = 0.0,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        self.input_scale = nnx.static(settings.input_scale)
        # Non-Param Variables: the output scales are data (no recompilation
        # when they change) but sit outside the differentiated Param state.
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        self.c_scale = nnx.Variable(jnp.asarray(c_scale, dtype=dtype))
        # trend is static so the trendless graph stays exactly the legacy one.
        self.trend = nnx.static(settings.learn_g_hat or g_hat != 0.0)
        if settings.learn_g_hat:
            self.g_hat = nnx.Param(jnp.full((), settings.g_hat_init, dtype=dtype))
        else:
            self.g_hat = nnx.Variable(jnp.asarray(g_hat, dtype=dtype))
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
        output_scale = jnp.stack([self.k_scale[...], self.c_scale[...]])
        out = output_scale * unit_softplus(self.head(h))
        if self.trend:
            out = jnp.exp(self.g_hat[...] * t) * out
        return out


def residual_fn(theta, args, p):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    values, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
        args.t
    )
    k = values[:, 0]
    c = values[:, 1]
    dk_dt = derivatives[:, 0]
    dc_dt = derivatives[:, 1]
    z = p.z_0 * (1.0 if args.z_trend is None else args.z_trend)
    resource = dk_dt - (production(z, k, p) - p.delta * k - c)
    euler = dc_dt - c * (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
    ic = model(jnp.asarray(0.0, dtype=args.t.dtype))[0] - p.k_0
    residual = jnp.concatenate([resource, euler, jnp.atleast_1d(ic)])
    return residual / jnp.sqrt(residual.size)


# Model paths and residuals on a time grid, with relative errors against the
# benchmark when one exists (the smooth family).
def snapshot(model, t, p, benchmark=None, z=1.0):
    values, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
        t
    )
    k = values[:, 0]
    c = values[:, 1]
    dk_dt = derivatives[:, 0]
    dc_dt = derivatives[:, 1]
    df = pd.DataFrame(
        {
            "t": np.asarray(t),
            "k": np.asarray(k),
            "c": np.asarray(c),
            "resource": np.asarray(dk_dt - (production(z, k, p) - p.delta * k - c)),
            "euler": np.asarray(
                dc_dt - c * (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
            ),
        }
    )
    if benchmark is not None:
        k_b, c_b = benchmark(t)
        df["k_benchmark"] = np.asarray(k_b)
        df["c_benchmark"] = np.asarray(c_b)
        df["k_rel_error"] = np.abs(df["k"] - df["k_benchmark"]) / np.abs(
            df["k_benchmark"]
        )
        df["c_rel_error"] = np.abs(df["c"] - df["c_benchmark"]) / np.abs(
            df["c_benchmark"]
        )
    return df


def neoclassical_growth_nn(
    a: float = CONSTANTS.a,
    delta: float = CONSTANTS.delta,
    rho_hat: float = CONSTANTS.rho_hat,
    k_0: float = 1.0,
    # Hicks-neutral technology z(t) = z_0 e^{(1-a) g t}: the unit trend enters
    # the residuals as data, z_0 rides in p; nn_set.learn_g_hat frees the
    # trend rate instead of fixing it at g.
    z_0: float = 1.0,
    g: float = 0.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    sigma_crra: float = CONSTANTS.sigma_crra,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: GrowthPolicySettings = GrowthPolicySettings(),
    # Geodesic acceleration is required: without it the solve stalls short of
    # the loss threshold.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.5,
        damping_increase=2.0,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
    # Model setup.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    # Kinked production is enabled exactly when all of A, b_1, b_2 are given.
    use_kinked_production = not (A is None or b_1 is None or b_2 is None)
    if use_kinked_production:
        A, b_1, b_2 = float(A), float(b_1), float(b_2)
        k_thr = (b_2 / (b_1 - 1.0)) ** (1.0 / a)
    else:
        A, b_1, b_2, k_thr = 1.0, 1.0, 0.0, float("inf")

    dtype = jnp.float64 if use_float64 else jnp.float32
    p = StructuralParameters.create(
        dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=z_0,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        k_thr=k_thr,
    )

    t_train = make_train_t_grid(data_set, dtype)
    z_trend_train = jnp.exp((1.0 - a) * g * t_train) if g != 0.0 else None
    c_flow_guess = steady_flow_consumption_guess(z_0, k_0, a, delta, A, b_1, b_2)
    policy_settings = replace(nn_set, input_scale=max(float(data_set.train_T), 1.0))
    k_scale = float(k_0)
    c_scale = c_flow_guess
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = GrowthPolicyNN(
        policy_settings,
        k_scale=k_scale,
        c_scale=c_scale,
        g_hat=g,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = ResidualArgs(
        graphdef=graphdef, nondiff=nondiff, t=t_train, z_trend=z_trend_train
    )

    # Multi-start Levenberg-Marquardt solve.
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )

    # solve()'s atol is on the residual norm; the settings threshold is on
    # the sum of squared residuals.
    atol = math.sqrt(tolerance_set.train_loss_threshold)
    solve_options = dict(
        max_steps=tolerance_set.max_steps,
        atol=atol,
        max_steps_is_success=False,
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=tolerance_set.num_attempts,
        draw=DrawNNXModule(
            GrowthPolicyNN,
            policy_settings,
            k_scale=k_scale,
            c_scale=c_scale,
            g_hat=g,
            dtype=dtype,
        ),
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
    model = nnx.merge(graphdef, result.x, nondiff)
    steps = int(result.steps)
    attempt = int(result.multi_start.attempt) + 1
    train_loss = float(result.info.loss)
    accepted = bool(result.multi_start.accepted)

    # Diagnostics against the benchmark.
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    z_trend_test = jnp.exp((1.0 - a) * g * test_t) if g != 0.0 else None
    benchmark = (
        None
        if use_kinked_production
        else neoclassical_growth_benchmark(
            a, delta, rho_hat, sigma_crra, g=g, z_0=z_0, k_0=k_0
        )
    )
    train_df = snapshot(
        model,
        t_train,
        p,
        benchmark,
        z=z_0 if z_trend_train is None else z_0 * z_trend_train,
    )
    test_df = snapshot(
        model,
        test_t,
        p,
        benchmark,
        z=z_0 if z_trend_test is None else z_0 * z_trend_test,
    )

    tvc_t = jnp.asarray(data_set.test_tvc_T, dtype=dtype)
    k_T, c_T = model(tvc_t)
    log_tvc = float(-p.rho_hat * tvc_t - p.sigma_crra * jnp.log(c_T) + jnp.log(k_T))
    transversality_ok = bool(log_tvc < tolerance_set.log_tvc_threshold)

    # Published results.
    g_hat_value = float(model.g_hat[...])

    results = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "z_0": z_0,
        "k_0": k_0,
        "g": g,
        "g_hat": g_hat_value,
        "g_hat_error": g_hat_value - g,
        "A": A,
        "b_1": b_1,
        "b_2": b_2,
        "seed": seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(k_T.dtype),
        "train_loss": train_loss,
        "final_damping": float(result.lm_state.damping),
        "test_resource_mean_abs": float(test_df["resource"].abs().mean()),
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "max_train_residual": float(
            max(train_df["resource"].abs().max(), train_df["euler"].abs().max())
        ),
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "steps": steps,
        "attempt": attempt,
        "solve_time": solve_time,
        "solve_time_success": solve_time if accepted else np.nan,
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
            test_k_rel_error_mean=float(test_df["k_rel_error"].mean()),
            test_c_rel_error_mean=float(test_df["c_rel_error"].mean()),
            test_k_rel_error_max=float(test_df["k_rel_error"].abs().max()),
            test_c_rel_error_max=float(test_df["c_rel_error"].abs().max()),
        )

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    hyperparameters = {
        "nn_set": asdict(policy_settings),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "parameters": {
            "a": a,
            "delta": delta,
            "rho_hat": rho_hat,
            "sigma_crra": sigma_crra,
            "g": g,
            "A": A,
            "b_1": b_1,
            "b_2": b_2,
        },
        "initial_conditions": {"z_0": z_0, "k_0": k_0},
    }

    return results, {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
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
    jsonargparse.CLI(neoclassical_growth_nn)
