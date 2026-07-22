from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from flax import nnx, struct
from jax.flatten_util import ravel_pytree
from nlls_gram import LevenbergMarquardt, LMStatus

from multicountry_growth_kernel import make_train_grid
from trade_growth.model import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    consumption_basket_log_price,
    multicountry_trade_dae_residual,
    multicountry_trade_max_dae_residual,
    multicountry_trade_path_values,
    multicountry_trade_pointwise_residual_norms,
    multicountry_trade_tvc_diagnostics,
    static_trade_equilibrium,
)
from utilities import ACTIVATIONS, unit_softplus

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
ACCOUNTING_RESIDUAL_TOL = 5e-5
INITIAL_CLEARING_TOL = 1e-10
STATUS_TO_REASON = {
    LMStatus.CONVERGED: "early_stopping_met",
    LMStatus.NONFINITE: "loss_divergence",
    LMStatus.MAX_STEPS: "max_steps",
}


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 120.0
    train_points: int = 32
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 8
    test_T: float = 80.0
    test_points: int = 161
    validation_points: int = 241
    quality_T: float = 100.0


@dataclass(frozen=True)
class SequentialNNSettings:
    width: int = 32
    depth: int = 3
    hidden_activation: str = "tanh"
    input_scale: float = 40.0
    head_scale: float = 1e-3


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-5
    damping_decrease: float = 0.5
    damping_increase: float = 2.0
    geodesic_acceleration: bool = True
    geodesic_acceptance_ratio: float = 0.8
    max_steps: int = 150
    residual_atol: float = 1e-12


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    fixed_state: nnx.State
    train_data: jax.Array
    k_0: jax.Array
    a_0: jax.Array
    q_reference: jax.Array
    marginal_wealth_reference: jax.Array
    output_scale: jax.Array
    static_reference_log: jax.Array
    parameters: TradeGrowthParameters = struct.field(pytree_node=False)
    num_countries: int = struct.field(pytree_node=False)
    n_train: int = struct.field(pytree_node=False)


class MulticountryTransitionNN(nnx.Module):
    def __init__(
        self,
        num_countries: int,
        settings: SequentialNNSettings,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.num_countries = num_countries
        self.hidden_activation = ACTIVATIONS[settings.hidden_activation]
        self.input_scale = settings.input_scale
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
            12 * num_countries + 1,
            kernel_init=nnx.initializers.normal(settings.head_scale),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, t):
        h = jnp.atleast_1d(t / self.input_scale)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return self.head(h)


def dynamic_coordinates(
    policy,
    points,
    k_0,
    a_0,
    q_reference,
    marginal_wealth_reference,
    output_scale,
):
    def decoded_path(t):
        raw = policy(t)[: 4 * policy.num_countries].reshape(4, policy.num_countries)
        time_coordinate = t / policy.input_scale
        return (
            k_0 * jnp.exp(time_coordinate * raw[0]),
            a_0 + output_scale * time_coordinate * raw[1],
            q_reference * unit_softplus(raw[2]),
            marginal_wealth_reference * unit_softplus(raw[3]),
        )

    values, derivatives = jax.vmap(
        lambda t: jax.jvp(decoded_path, (t,), (jnp.ones_like(t),))
    )(points)
    k, a, tobin_q, marginal_wealth = values
    dk_dt, da_dt, dq_dt, dm_dt = derivatives
    return {
        "k": k,
        "a": a,
        "tobin_q": tobin_q,
        "marginal_wealth": marginal_wealth,
        "dlog_k_dt": dk_dt / k,
        "da_dt": da_dt,
        "dlog_q_dt": dq_dt / tobin_q,
        "dlog_marginal_wealth_dt": dm_dt / marginal_wealth,
    }


def residual_fn(theta, args):
    policy = nnx.merge(args.graphdef, theta, args.fixed_state)
    coordinates = dynamic_coordinates(
        policy,
        args.train_data,
        args.k_0,
        args.a_0,
        args.q_reference,
        args.marginal_wealth_reference,
        args.output_scale,
    )
    raw = jax.vmap(policy)(args.train_data)
    interest_rate = args.parameters.rho * (1.0 + raw[:, 4 * args.num_countries])
    log_static = args.static_reference_log + raw[
        :, 4 * args.num_countries + 1 :
    ].reshape(args.n_train, len(STATIC_BLOCK_NAMES), args.num_countries)
    values = multicountry_trade_path_values(
        **coordinates,
        interest_rate=interest_rate,
        log_static=log_static,
        parameters=args.parameters,
    )
    return multicountry_trade_dae_residual(values, log_static, args.parameters)


def multicountry_trade_growth_nn(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: TradeGrowthParameters = TradeGrowthParameters(),
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(),
    nn_set: SequentialNNSettings = SequentialNNSettings(),
    lm_set: SolverSettings = SolverSettings(),
    verbose: bool = True,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    num_countries = parameters.num_countries
    if k_0 is None:
        k_0 = (1.0,) * num_countries
    if a_0 is None:
        a_0 = (0.0,) * num_countries
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive value per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one value per country")
    if nn_set.width <= 0 or nn_set.depth <= 0:
        raise ValueError("NN width and depth must be positive")
    if nn_set.input_scale <= 0.0 or nn_set.head_scale <= 0.0:
        raise ValueError("NN input_scale and head_scale must be positive")
    if lm_set.residual_atol <= 0.0 or lm_set.max_steps <= 0:
        raise ValueError("solver tolerance and step budget must be positive")
    dtype = jnp.float64 if use_float64 else jnp.float32
    train_data = make_train_grid(data_set).astype(dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.quality_T <= 0.0:
        raise ValueError("quality_T must be positive")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0,
        float(train_data[-1]),
        data_set.validation_points,
        dtype=dtype,
    )
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    initial_asset_gap = jnp.sum(populations * a_0_array)
    if abs(float(initial_asset_gap)) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")

    base_q = jnp.full(
        num_countries,
        1.0 / parameters.installation_elasticity,
        dtype=dtype,
    )
    base_m = jnp.ones(num_countries, dtype=dtype)
    base_static, _ = static_trade_equilibrium(
        k_0_array,
        base_q,
        base_m,
        parameters,
    )
    basket_price = jnp.exp(
        consumption_basket_log_price(base_static["log_static"], parameters)
    )
    price_scale = 1.0 / basket_price
    q_reference = price_scale * base_q
    marginal_wealth_reference = base_m / price_scale
    static_reference, static_reference_diagnostics = static_trade_equilibrium(
        k_0_array,
        q_reference,
        marginal_wealth_reference,
        parameters,
        log_initial=base_static["log_static"] + jnp.log(price_scale),
    )
    output_scale = static_reference["world_gdp"] / jnp.sum(populations)

    model = MulticountryTransitionNN(
        num_countries,
        nn_set,
        dtype=dtype,
        rngs=nnx.Rngs(jax.random.key(seed)),
    )
    graphdef, theta_0, fixed_state = nnx.split(model, nnx.Param, ...)
    static_reference_log = static_reference["log_static"]

    residual_count = (12 * num_countries + 1) * n_train
    residual_args = ResidualArgs(
        graphdef=graphdef,
        fixed_state=fixed_state,
        train_data=train_data,
        k_0=k_0_array,
        a_0=a_0_array,
        q_reference=q_reference,
        marginal_wealth_reference=marginal_wealth_reference,
        output_scale=output_scale,
        static_reference_log=static_reference_log,
        parameters=parameters,
        num_countries=num_countries,
        n_train=n_train,
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
    solve_start = time.perf_counter()
    result = solver.solve(
        theta_0,
        residual_args,
        max_steps=lm_set.max_steps,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )
    jax.block_until_ready(result)
    solve_time = time.perf_counter() - solve_start
    theta = result.x
    policy = nnx.merge(graphdef, theta, fixed_state)

    def evaluate_grid(points):
        coordinates = dynamic_coordinates(
            policy,
            points,
            k_0_array,
            a_0_array,
            q_reference,
            marginal_wealth_reference,
            output_scale,
        )
        raw = jax.vmap(policy)(points)
        interest_rate = parameters.rho * (1.0 + raw[:, 4 * num_countries])
        log_static = static_reference_log + raw[:, 4 * num_countries + 1 :].reshape(
            len(points),
            len(STATIC_BLOCK_NAMES),
            num_countries,
        )
        return (
            multicountry_trade_path_values(
                **coordinates,
                interest_rate=interest_rate,
                log_static=log_static,
                parameters=parameters,
            ),
            log_static,
        )

    train_eval, train_log_static = evaluate_grid(train_data)
    validation_eval, validation_log_static = evaluate_grid(validation_data)
    test_eval, test_log_static = evaluate_grid(test_data)
    quality_T = min(data_set.quality_T, float(validation_data[-1]))
    quality_count = int(
        np.searchsorted(np.asarray(validation_data), quality_T, side="right")
    )
    quality_slice = slice(quality_count)

    train_coordinates = dynamic_coordinates(
        policy,
        train_data[:1],
        k_0_array,
        a_0_array,
        q_reference,
        marginal_wealth_reference,
        output_scale,
    )
    max_initial_residual = max(
        float(jnp.max(jnp.abs(jnp.log(train_coordinates["k"][0] / k_0_array)))),
        float(jnp.max(jnp.abs((train_coordinates["a"][0] - a_0_array) / output_scale))),
    )
    max_train_residual = max(
        float(
            multicountry_trade_max_dae_residual(
                train_eval,
                train_log_static,
                parameters,
            )
        ),
        max_initial_residual,
    )
    max_validation_residual = float(
        multicountry_trade_max_dae_residual(
            validation_eval,
            validation_log_static,
            parameters,
            quality_slice,
        )
    )
    max_full_horizon_validation_residual = float(
        multicountry_trade_max_dae_residual(
            validation_eval,
            validation_log_static,
            parameters,
        )
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"][quality_slice]))
    )
    max_full_horizon_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"]))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"][quality_slice]))
    )
    max_full_horizon_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"]))
    )
    validation_residual_norms = multicountry_trade_pointwise_residual_norms(
        validation_eval,
        validation_log_static,
        parameters,
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name][selection]))
            and jnp.min(values[name][selection]) > 0.0
            for values, selection in (
                (train_eval, slice(None)),
                (validation_eval, quality_slice),
                (test_eval, slice(None)),
            )
            for name in ("k", "q", "marginal_wealth", "consumption", "investment")
        )
        and all(
            jnp.all(jnp.isfinite(values[name][selection]))
            for values, selection in (
                (train_eval, slice(None)),
                (validation_eval, quality_slice),
                (test_eval, slice(None)),
            )
            for name in ("a", "interest_rate", "current_account")
        )
    )
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    if max_asset_clearing_residual > ACCOUNTING_RESIDUAL_TOL:
        rejection_reasons.append("asset_clearing")
    if max_rate_consistency_residual > ACCOUNTING_RESIDUAL_TOL:
        rejection_reasons.append("rate_consistency")

    tvc = multicountry_trade_tvc_diagnostics(
        validation_data,
        validation_eval,
        parameters,
    )

    def endpoint_max(name: str, index: int) -> float:
        return float(jnp.max(jnp.abs(tvc[name][index])))

    def endpoint_decay_ratio(name: str) -> float:
        initial = endpoint_max(name, 0)
        if initial <= 1e-10:
            return 0.0
        return endpoint_max(name, -1) / initial

    theta_initial_flat, _ = ravel_pytree(theta_0)
    theta_final_flat, _ = ravel_pytree(theta)
    network_flat, _ = ravel_pytree(theta)
    unknown_count = int(theta_final_flat.size)
    results = {
        "alpha": parameters.alpha,
        "gamma": parameters.gamma,
        "delta": parameters.delta,
        "rho": parameters.rho,
        "installation_elasticity": parameters.installation_elasticity,
        "trade_elasticity": parameters.trade_elasticity,
        "num_countries": num_countries,
        "seed": seed,
        "residual_dtype": str(jnp.dtype(dtype)),
        "linear_solve_dtype": "float64",
        "train_T": float(train_data[-1]),
        "quality_T": quality_T,
        "quality_validation_point_count": quality_count,
        "num_train_points": n_train,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_full_horizon_validation_residual": (max_full_horizon_validation_residual),
        "max_initial_residual": max_initial_residual,
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_full_horizon_asset_clearing_residual": (
            max_full_horizon_asset_clearing_residual
        ),
        "max_rate_consistency_residual": max_rate_consistency_residual,
        "max_full_horizon_rate_consistency_residual": (
            max_full_horizon_rate_consistency_residual
        ),
        **{
            f"max_validation_{name}": float(jnp.max(values[quality_slice]))
            for name, values in validation_residual_norms.items()
        },
        **{
            f"max_full_horizon_{name}": float(jnp.max(values))
            for name, values in validation_residual_norms.items()
        },
        "terminal_residual_count": 0,
        "dae_residual_count": residual_count,
        "initial_residual_count": 0,
        "unknown_count": unknown_count,
        "network_parameter_count": int(network_flat.size),
        "network_output_count": 12 * num_countries + 1,
        "algebraic_network_output_count": 8 * num_countries + 1,
        "numeraire": "population_weighted_geometric_consumption_price",
        "ansatz": "sequential_time_level_nn_with_algebraic_trade_values",
        "selection": "minimum_identity_metric_parameter_step",
        "metric": "identity_parameter_metric",
        "initialization": "normalized_static_reference_near_zero_head",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "dtype": str(jnp.dtype(dtype)),
        "minimum_interest_rate": float(
            jnp.min(validation_eval["interest_rate"][quality_slice])
        ),
        "full_horizon_minimum_interest_rate": float(
            jnp.min(validation_eval["interest_rate"])
        ),
        "ex_post_discounted_assets_initial": endpoint_max("discounted_assets", 0),
        "ex_post_discounted_assets": endpoint_max("discounted_assets", -1),
        "ex_post_discounted_assets_decay_ratio": endpoint_decay_ratio(
            "discounted_assets"
        ),
        "ex_post_asset_tvc_initial": endpoint_max("asset_tvc", 0),
        "ex_post_asset_tvc": endpoint_max("asset_tvc", -1),
        "ex_post_asset_tvc_decay_ratio": endpoint_decay_ratio("asset_tvc"),
        "ex_post_capital_tvc_initial": endpoint_max("capital_tvc", 0),
        "ex_post_capital_tvc": endpoint_max("capital_tvc", -1),
        "ex_post_capital_tvc_decay_ratio": endpoint_decay_ratio("capital_tvc"),
        "parameter_norm_initial": float(jnp.linalg.norm(theta_initial_flat)),
        "parameter_norm_final": float(jnp.linalg.norm(theta_final_flat)),
        "parameter_step_norm": float(
            jnp.linalg.norm(theta_final_flat - theta_initial_flat)
        ),
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "solve_time": solve_time,
        "solver": "nlls-gram",
        "solution_accepted": not rejection_reasons,
        "rejection_reason": (
            "accepted" if not rejection_reasons else ",".join(rejection_reasons)
        ),
        "static_reference_max_residual": static_reference_diagnostics["max_residual"],
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
    }
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"population_{suffix}": parameters.populations[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"terminal_k_{suffix}": float(validation_eval["k"][-1, country]),
                f"terminal_consumption_{suffix}": float(
                    validation_eval["consumption"][-1, country]
                ),
                f"terminal_current_account_{suffix}": float(
                    validation_eval["current_account"][-1, country]
                ),
                f"terminal_trade_balance_{suffix}": float(
                    validation_eval["trade_balance"][-1, country]
                ),
            }
        )

    if num_countries >= 3:
        symmetry_paths = {
            "capital": "k",
            "assets": "a",
            "tobin_q": "q",
            "marginal_wealth": "marginal_wealth",
            "consumption": "consumption",
            "investment_rate": "investment_rate",
            "trade_balance": "trade_balance",
            "current_account": "current_account",
            **{name: name for name in STATIC_BLOCK_NAMES},
        }
        for label, name in symmetry_paths.items():
            results[f"max_country_2_3_{label}_gap"] = float(
                jnp.max(
                    jnp.abs(
                        validation_eval[name][quality_slice, 1]
                        - validation_eval[name][quality_slice, 2]
                    )
                )
            )
        permutation = jnp.asarray((0, 2, 1, *range(3, num_countries)))
        permuted_trade_shares = jnp.take(
            jnp.take(validation_eval["trade_shares"], permutation, axis=-2),
            permutation,
            axis=-1,
        )
        results["max_country_2_3_trade_share_permutation_gap"] = float(
            jnp.max(
                jnp.abs(
                    validation_eval["trade_shares"][quality_slice]
                    - permuted_trade_shares[quality_slice]
                )
            )
        )

    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    def frame(points, values, log_static):
        data = {"t": np.asarray(points)}
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "marginal_wealth",
                "consumption",
                "investment",
                "investment_rate",
                "trade_balance",
                "current_account",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        data["interest_rate"] = np.asarray(values["interest_rate"])
        data["consumption_basket_log_price"] = np.asarray(
            consumption_basket_log_price(log_static, parameters)
        )
        for name, residual_norm in multicountry_trade_pointwise_residual_norms(
            values,
            log_static,
            parameters,
        ).items():
            data[name] = np.asarray(residual_norm)
        return pd.DataFrame(data)

    hyperparameters = {
        "parameters": asdict(parameters),
        "nn_set": asdict(nn_set),
        "data_set": asdict(data_set),
        "lm_set": asdict(lm_set),
        "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
    }
    return results, {
        "model": policy,
        "train_df": frame(train_data, train_eval, train_log_static),
        "validation_df": frame(
            validation_data,
            validation_eval,
            validation_log_static,
        ),
        "test_df": frame(test_data, test_eval, test_log_static),
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_trade_growth_nn)
