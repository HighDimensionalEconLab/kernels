from __future__ import annotations

from dataclasses import asdict, replace

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
    MultiStart,
)

from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)
from models.neoclassical_growth_shared import (
    DataSettings,
    GrowthPolicySettings,
    StructuralParameters,
    ToleranceSettings,
    mpk,
    production,
)
from models.utilities import (
    ACTIVATIONS,
    NNXResidualArgs,
    OptimizerSettings,
    Stopwatch,
    acceptance_fields,
    make_train_t_grid,
    unit_softplus,
    value_and_grad_fwd,
)


# Single-head capital path k(t); consumption is derived from the resource
# constraint rather than approximated.
class CapitalPathNN(nnx.Module):
    def __init__(
        self,
        settings: GrowthPolicySettings,
        *,
        k_scale: float,
        g_hat: float = 0.0,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        self.input_scale = nnx.static(settings.input_scale)
        # Non-Param Variable: the output scale is data (no recompilation when
        # it changes) but sits outside the differentiated Param state.
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        # trend is static so the g = 0 graph carries no trend factor.
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
            1,
            use_bias=settings.output_bias,
            kernel_init=nnx.initializers.normal(1e-3),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    # Scalar t to scalar k, so the nested jvps in the residual stay clean.
    def __call__(self, t):
        h = jnp.atleast_1d(t / self.input_scale)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        k = self.k_scale[...] * unit_softplus(self.head(h)[0])
        if self.trend:
            k = jnp.exp(self.g_hat[...] * t) * k
        return k


# c(t) = f(z(t), k(t)) - delta k(t) - k_dot(t): the resource constraint holds
# identically, leaving only the Euler equation and the initial condition.
def substituted_consumption(model, t, p):
    k, dk_dt = value_and_grad_fwd(model)(t)
    z = p.z_0 * jnp.exp((1.0 - p.a) * p.g * t)
    return production(z, k, p) - p.delta * k - dk_dt


# Euler in log-consumption form: dlog(c)/dt = (f'(k) - delta - rho_hat) /
# sigma_crra, with c substituted from the resource constraint.
def euler_residual(model, t, p):
    log_c, dlog_c_dt = value_and_grad_fwd(
        lambda s: jnp.log(substituted_consumption(model, s, p))
    )(t)
    z = p.z_0 * jnp.exp((1.0 - p.a) * p.g * t)
    # Zero on the feasible domain; keeps the residual NaN when c <= 0, where
    # the tangent alone would stay finite.
    domain_guard = 0.0 * log_c
    return (
        dlog_c_dt
        + domain_guard
        - (mpk(z, model(t), p) - p.delta - p.rho_hat) / p.sigma_crra
    )


def residual_fn(theta, args, p):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    t_train = args.batch["t"]
    euler = jax.vmap(lambda t: euler_residual(model, t, p))(t_train)
    ic = model(jnp.asarray(0.0, dtype=t_train.dtype)) - p.k_0
    residual = jnp.concatenate([euler, jnp.atleast_1d(ic)])
    return residual / jnp.sqrt(residual.size), {"euler": euler}


# Model paths and residuals on a time grid, with relative errors against the
# benchmark when one exists (the smooth family). No resource column: it is
# zero by construction; k and c are derived here, off the solve path.
def snapshot(theta, args, p, benchmark=None):
    model = nnx.merge(args.graphdef, theta, args.nondiff)
    t = args.batch["t"]
    values = residual_fn(theta, args, p)[1]
    df = pd.DataFrame(
        {
            "t": np.asarray(t),
            "k": np.asarray(jax.vmap(model)(t)),
            "c": np.asarray(
                jax.vmap(lambda s: substituted_consumption(model, s, p))(t)
            ),
            "euler": np.asarray(values["euler"]),
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


def neoclassical_growth_nn_k(
    a: float = CONSTANTS["a"],
    delta: float = CONSTANTS["delta"],
    rho_hat: float = CONSTANTS["rho_hat"],
    k_0: float = 1.0,
    # Hicks-neutral technology z(t) = z_0 e^{(1-a) g t} enters the residual
    # analytically; nn_set.learn_g_hat frees the trend rate instead of fixing
    # it at g.
    z_0: float = 1.0,
    g: float = 0.0,
    A: float | None = None,
    b_1: float | None = None,
    b_2: float | None = None,
    sigma_crra: float = CONSTANTS["sigma_crra"],
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(test_points=101),
    nn_set: GrowthPolicySettings = GrowthPolicySettings(),
    # atol is on the residual norm (the loss stops at atol^2).
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.5,
        damping_increase=2.0,
        max_steps=250,
        atol=3.162277660168379e-08,
        num_attempts=5,
    ),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = True,
):
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
        dtype=dtype,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        z_0=z_0,
        k_0=k_0,
        g=g,
        A=A,
        b_1=b_1,
        b_2=b_2,
        k_thr=k_thr,
    )

    t_train = make_train_t_grid(data_set, dtype)
    policy_settings = replace(nn_set, input_scale=max(float(data_set.train_T), 1.0))
    k_scale = float(k_0)
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = CapitalPathNN(
        policy_settings,
        k_scale=k_scale,
        g_hat=g,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    graphdef, theta_0, nondiff = nnx.split(model, nnx.Param, ...)
    residual_args = NNXResidualArgs(
        batch={"t": t_train}, graphdef=graphdef, nondiff=nondiff
    )

    # Multi-start Levenberg-Marquardt solve.
    solver = LevenbergMarquardt(
        residual_fn,
        has_aux=True,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
    )

    solve_options = dict(
        max_steps=lm_set.max_steps,
        atol=lm_set.atol,
        max_steps_is_success=False,
    )
    multi_start = MultiStart(
        key=multi_start_key,
        num_starts=lm_set.num_attempts,
        draw=DrawNNXModule(
            CapitalPathNN,
            policy_settings,
            k_scale=k_scale,
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
    rejection_reasons = [] if accepted else ["train_loss"]

    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    benchmark = None if use_kinked_production else neoclassical_growth_benchmark(p)
    train_df = snapshot(result.x, residual_args, p, benchmark)
    test_df = snapshot(
        result.x, residual_args.replace(batch={"t": test_t}), p, benchmark
    )

    tvc_t = jnp.asarray(data_set.test_tvc_T, dtype=dtype)
    k_T = model(tvc_t)
    c_T = substituted_consumption(model, tvc_t, p)
    log_tvc = float(-p.rho_hat * tvc_t - p.sigma_crra * jnp.log(c_T) + jnp.log(k_T))
    transversality_ok = bool(log_tvc < tolerance_set.log_tvc_threshold)

    g_hat_value = float(model.g_hat[...])

    results = {
        "parameters": p.to_dict(),
        "nn_set": asdict(policy_settings),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "g_hat": g_hat_value,
        "g_hat_error": g_hat_value - g,
        "seed": seed,
        "train_loss": train_loss,
        "final_damping": float(result.lm_state.damping),
        "test_euler_mean_abs": float(test_df["euler"].abs().mean()),
        "max_train_residual": float(train_df["euler"].abs().max()),
        "log_tvc_residual": log_tvc,
        "transversality_ok": transversality_ok,
        "steps": steps,
        "attempt": attempt,
        "solve_time": solve_time,
        "solve_time_success": solve_time if not rejection_reasons else np.nan,
        "total_params": int(ravel_pytree(nnx.state(model, nnx.Param))[0].size),
        **acceptance_fields(rejection_reasons),
        "train_T": float(t_train[-1]),
        "num_train_points": int(t_train.shape[0]),
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

    return results, {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
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
    jsonargparse.CLI(neoclassical_growth_nn_k)
