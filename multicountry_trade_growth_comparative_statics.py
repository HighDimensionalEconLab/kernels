from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.flatten_util import ravel_pytree
from nlls_gram import (
    UnderdeterminedLevenbergMarquardt,
    metric_from_cholesky,
    metric_from_diagonal,
    metric_with_compute_dtype,
    repeated_blockdiag_metric,
)

from kernels import integrated_kernel_matrices, normalize_kernel
from multicountry_growth_kernel_nlls import make_train_grid
from multicountry_trade_growth import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    consumption_basket_log_price,
    multicountry_trade_dae_residual,
    multicountry_trade_path_values,
    static_trade_equilibrium,
)
from multicountry_trade_growth_kernel_nlls import (
    DataSettings,
    KernelSettings,
    NLLSGramSettings,
)


@dataclass(frozen=True)
class DifferentiableTradeGrowthParameters:
    alpha: jax.Array
    gamma: jax.Array
    delta: jax.Array
    rho: jax.Array
    installation_elasticity: jax.Array
    trade_elasticity: jax.Array
    variety_elasticity: jax.Array
    populations: jax.Array
    intermediate_productivities: jax.Array
    consumption_productivities: jax.Array
    investment_productivities: jax.Array
    consumption_value_added_shares: jax.Array
    intermediate_value_added_shares: jax.Array
    investment_value_added_shares: jax.Array
    trade_costs: jax.Array

    @property
    def num_countries(self):
        return self.populations.shape[0]

    @property
    def installation_efficiency(self):
        return self.delta ** (1.0 - self.installation_elasticity)

    @property
    def price_index_constant(self):
        shape = 1.0 + (1.0 - self.variety_elasticity) / self.trade_elasticity
        return jnp.exp(jsp.special.gammaln(shape) / (1.0 - self.variety_elasticity))


@dataclass(frozen=True)
class MulticountryTradeComparativeStatics:
    solver: UnderdeterminedLevenbergMarquardt
    theta_0: tuple[jax.Array, ...]
    structural_parameters: dict[str, jax.Array]
    statistic_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_coordinate_names: tuple[str, ...]
    parameter_coordinate_scales: jax.Array
    aggregate_income_scale: float
    max_steps: int
    residual_atol: float

    def solve(self, structural_parameters=None):
        p = (
            self.structural_parameters
            if structural_parameters is None
            else structural_parameters
        )
        return self.solver.solve(
            self.theta_0,
            p=p,
            max_steps=self.max_steps,
            atol=self.residual_atol,
        )

    def statistics(self, structural_parameters=None):
        return self.solve(structural_parameters).aux

    def jacobian(self, structural_parameters=None):
        p = (
            self.structural_parameters
            if structural_parameters is None
            else structural_parameters
        )
        flat_p, unravel_p = ravel_pytree(p)

        def statistics_from_flat(values):
            return self.statistics(unravel_p(values))

        statistics, pullback = jax.vjp(statistics_from_flat, flat_p)
        basis = jnp.eye(statistics.size, dtype=statistics.dtype)
        jacobian = jax.vmap(lambda row: pullback(row)[0])(basis)
        return statistics, jacobian

    def interpretable_jacobian(self, structural_parameters=None):
        statistics, jacobian = self.jacobian(structural_parameters)
        return statistics, jacobian * self.parameter_coordinate_scales[None, :]


def trade_growth_structural_parameters(
    k_0,
    a_0,
    parameters,
    dtype=jnp.float64,
):
    trade_costs = jnp.asarray(parameters.trade_costs, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    aggregate_initial_assets = populations * jnp.asarray(a_0, dtype=dtype)
    num_countries = parameters.num_countries
    off_diagonal = ~jnp.eye(num_countries, dtype=jnp.bool_)
    return {
        "alpha": jnp.asarray(parameters.alpha, dtype=dtype),
        "consumption_productivities": jnp.asarray(
            parameters.consumption_productivities, dtype=dtype
        ),
        "consumption_value_added_shares": jnp.asarray(
            parameters.consumption_value_added_shares, dtype=dtype
        ),
        "delta": jnp.asarray(parameters.delta, dtype=dtype),
        "gamma": jnp.asarray(parameters.gamma, dtype=dtype),
        "installation_elasticity": jnp.asarray(
            parameters.installation_elasticity, dtype=dtype
        ),
        "intermediate_productivities": jnp.asarray(
            parameters.intermediate_productivities, dtype=dtype
        ),
        "intermediate_value_added_shares": jnp.asarray(
            parameters.intermediate_value_added_shares, dtype=dtype
        ),
        "investment_productivities": jnp.asarray(
            parameters.investment_productivities, dtype=dtype
        ),
        "investment_value_added_shares": jnp.asarray(
            parameters.investment_value_added_shares, dtype=dtype
        ),
        "initial_aggregate_assets_free": aggregate_initial_assets[:-1],
        "initial_log_capital": jnp.log(jnp.asarray(k_0, dtype=dtype)),
        "populations": populations,
        "rho": jnp.asarray(parameters.rho, dtype=dtype),
        "trade_costs_off_diagonal": trade_costs[off_diagonal],
        "trade_elasticity": jnp.asarray(parameters.trade_elasticity, dtype=dtype),
        "variety_elasticity": jnp.asarray(parameters.variety_elasticity, dtype=dtype),
    }


def structural_parameter_names(structural_parameters, off_diagonal_pairs):
    names = []
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(structural_parameters)
    for path, leaf in path_leaves:
        name = str(path[0].key)
        shape = np.shape(leaf)
        if not shape:
            names.append(name)
        elif name == "trade_costs_off_diagonal":
            names.extend(
                f"trade_costs[{importer + 1},{exporter + 1}]"
                for importer, exporter in off_diagonal_pairs
            )
        else:
            names.extend(
                f"{name}[{','.join(str(index + 1) for index in indices)}]"
                for indices in np.ndindex(shape)
            )
    return tuple(names)


def structural_parameter_coordinates(parameter_names, structural_parameters):
    flat_parameters, _ = ravel_pytree(structural_parameters)
    level_roots = {
        "alpha",
        "consumption_value_added_shares",
        "initial_aggregate_assets_free",
        "installation_elasticity",
        "intermediate_value_added_shares",
        "investment_value_added_shares",
    }
    coordinate_names = []
    coordinate_scales = []
    for index, name in enumerate(parameter_names):
        root = name.split("[", maxsplit=1)[0]
        if root == "initial_log_capital":
            coordinate_names.append(
                name.replace("initial_log_capital", "log_initial_capital")
            )
            coordinate_scales.append(1.0)
        elif root in level_roots:
            coordinate_names.append(name)
            coordinate_scales.append(1.0)
        else:
            coordinate_names.append(f"log_{name}")
            coordinate_scales.append(flat_parameters[index])
    return tuple(coordinate_names), jnp.asarray(
        coordinate_scales,
        dtype=flat_parameters.dtype,
    )


def build_multicountry_trade_comparative_statics(
    k_0,
    a_0,
    parameters: TradeGrowthParameters,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(
        train_T=120.0,
        train_points=32,
        dense_T=40.0,
        tail_points=8,
        test_T=80.0,
        test_points=161,
        validation_points=241,
        quality_T=100.0,
    ),
    lm_set: NLLSGramSettings = NLLSGramSettings(
        init_damping=1e-8,
        residual_atol=1e-10,
        max_steps=100,
    ),
    statistic_times=(0.0, 40.0, 100.0),
    implicit_penalty=0.0,
):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    dtype = jnp.float64
    num_countries = parameters.num_countries
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive value per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one value per country")
    if lm_set.metric_eps <= 0.0:
        raise ValueError("metric_eps must be positive")
    if lm_set.linear_solver not in ("cholesky", "qr"):
        raise ValueError("linear_solver must be 'cholesky' or 'qr'")

    train_data = make_train_grid(data_set).astype(dtype)
    n_train = len(train_data)
    statistic_indices = []
    for time in statistic_times:
        matches = np.flatnonzero(np.isclose(np.asarray(train_data), time))
        if matches.size != 1:
            raise ValueError(f"statistic time {time} must be a collocation date")
        statistic_indices.append(int(matches[0]))
    initial_index, transition_index, long_index = statistic_indices

    k_0_array = jnp.asarray(k_0, dtype=dtype)
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    aggregate_initial_assets = populations * a_0_array
    if abs(float(jnp.sum(aggregate_initial_assets))) > 1e-10:
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
    static_reference, _ = static_trade_equilibrium(
        k_0_array,
        q_reference,
        marginal_wealth_reference,
        parameters,
        log_initial=base_static["log_static"] + jnp.log(price_scale),
    )
    output_scale = static_reference["world_gdp"] / jnp.sum(populations)

    kernel = normalize_kernel(kernel_set.kernel)
    K, K_tilde = integrated_kernel_matrices(
        train_data,
        train_data,
        kernel,
        kernel_set.sigma,
        kernel_set.ell,
    )
    K = 0.5 * (K + K.T)
    eps = lm_set.metric_eps
    alpha_metric = metric_from_cholesky(
        jnp.linalg.cholesky(K + eps * jnp.eye(n_train, dtype=dtype))
    )

    static_size = n_train * len(STATIC_BLOCK_NAMES) * num_countries
    algebraic_size = n_train + 2 * num_countries + static_size

    def split_algebraic(algebraic):
        rate_coordinates = algebraic[:n_train]
        q_offsets = algebraic[n_train : n_train + num_countries]
        marginal_wealth_start = n_train + num_countries
        marginal_wealth_offsets = algebraic[
            marginal_wealth_start : marginal_wealth_start + num_countries
        ]
        log_static = algebraic[marginal_wealth_start + num_countries :].reshape(
            n_train, len(STATIC_BLOCK_NAMES), num_countries
        )
        return rate_coordinates, q_offsets, marginal_wealth_offsets, log_static

    off_diagonal_pairs = tuple(
        (importer, exporter)
        for importer in range(num_countries)
        for exporter in range(num_countries)
        if importer != exporter
    )
    off_diagonal_importers = jnp.asarray(
        tuple(pair[0] for pair in off_diagonal_pairs), dtype=jnp.int32
    )
    off_diagonal_exporters = jnp.asarray(
        tuple(pair[1] for pair in off_diagonal_pairs), dtype=jnp.int32
    )

    def runtime_parameters(p):
        trade_costs = (
            jnp.eye(num_countries, dtype=dtype)
            .at[off_diagonal_importers, off_diagonal_exporters]
            .set(p["trade_costs_off_diagonal"])
        )
        return DifferentiableTradeGrowthParameters(
            alpha=p["alpha"],
            gamma=p["gamma"],
            delta=p["delta"],
            rho=p["rho"],
            installation_elasticity=p["installation_elasticity"],
            trade_elasticity=p["trade_elasticity"],
            variety_elasticity=p["variety_elasticity"],
            populations=p["populations"],
            intermediate_productivities=p["intermediate_productivities"],
            consumption_productivities=p["consumption_productivities"],
            investment_productivities=p["investment_productivities"],
            consumption_value_added_shares=p["consumption_value_added_shares"],
            intermediate_value_added_shares=p["intermediate_value_added_shares"],
            investment_value_added_shares=p["investment_value_added_shares"],
            trade_costs=trade_costs,
        )

    def evaluate(theta, p):
        alpha_k, alpha_a, alpha_q, alpha_m, algebraic = theta
        rate_coordinates, q_offsets, marginal_wealth_offsets, log_static = (
            split_algebraic(algebraic)
        )
        runtime = runtime_parameters(p)
        aggregate_assets = jnp.concatenate(
            [
                p["initial_aggregate_assets_free"],
                -jnp.sum(p["initial_aggregate_assets_free"], keepdims=True),
            ]
        )
        initial_assets = aggregate_assets / runtime.populations
        log_k = p["initial_log_capital"] + K_tilde @ alpha_k.T
        assets = initial_assets + output_scale * (K_tilde @ alpha_a.T)
        log_q = jnp.log(q_reference) + q_offsets + K_tilde @ alpha_q.T
        log_marginal_wealth = (
            jnp.log(marginal_wealth_reference)
            + marginal_wealth_offsets
            + K_tilde @ alpha_m.T
        )
        dlog_k_dt = K @ alpha_k.T
        da_dt = output_scale * (K @ alpha_a.T)
        dlog_q_dt = K @ alpha_q.T
        dlog_marginal_wealth_dt = K @ alpha_m.T
        interest_rate = runtime.rho * (1.0 + rate_coordinates)
        values = multicountry_trade_path_values(
            k=jnp.exp(log_k),
            a=assets,
            tobin_q=jnp.exp(log_q),
            marginal_wealth=jnp.exp(log_marginal_wealth),
            interest_rate=interest_rate,
            log_static=log_static,
            dlog_k_dt=dlog_k_dt,
            da_dt=da_dt,
            dlog_q_dt=dlog_q_dt,
            dlog_marginal_wealth_dt=dlog_marginal_wealth_dt,
            parameters=runtime,
        )
        return values, log_static, runtime

    def statistics(values, runtime):
        selected_indices = jnp.asarray(statistic_indices)
        log_consumption = jnp.log(values["consumption"][selected_indices])
        population_weights = runtime.populations / jnp.sum(runtime.populations)
        selected_trade_shares = values["trade_shares"][selected_indices]
        home_trade_shares = jnp.diagonal(
            selected_trade_shares,
            axis1=-2,
            axis2=-1,
        )
        aggregate_home_share = jnp.sum(
            population_weights * home_trade_shares,
            axis=-1,
        )
        aggregate_import_share = jnp.sum(
            population_weights * (1.0 - home_trade_shares),
            axis=-1,
        )
        long_income = values["factor_income"][long_index]
        transition_income = values["factor_income"][transition_index]
        return jnp.concatenate(
            [
                jnp.log(values["interest_rate"][selected_indices]),
                jnp.sum(population_weights * log_consumption, axis=-1),
                (log_consumption[:, 1:] - log_consumption[:, :1]).reshape(-1),
                jnp.log(values["k"][selected_indices[1:]]).reshape(-1),
                values["a"][long_index] / long_income,
                values["current_account"][transition_index] / transition_income,
                jnp.log(aggregate_home_share),
                jnp.log(aggregate_import_share),
                jnp.log(selected_trade_shares).reshape(-1),
            ]
        )

    def residual_fn(theta, _, p):
        values, log_static, runtime = evaluate(theta, p)
        residual = multicountry_trade_dae_residual(values, log_static, runtime)
        return residual, statistics(values, runtime)

    num_dynamic_functions = 4 * num_countries
    metric = metric_with_compute_dtype(
        repeated_blockdiag_metric(
            alpha_metric,
            n_train,
            num_dynamic_functions,
            additional=(
                metric_from_diagonal(jnp.full(algebraic_size, eps, dtype=dtype)),
                algebraic_size,
            ),
        ),
        dtype,
    )
    initial_algebraic = jnp.concatenate(
        [
            jnp.zeros(n_train + 2 * num_countries, dtype=dtype),
            jnp.broadcast_to(
                static_reference["log_static"],
                (n_train, len(STATIC_BLOCK_NAMES), num_countries),
            ).reshape(-1),
        ]
    )
    theta_0 = (
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        jnp.zeros((num_countries, n_train), dtype=dtype),
        initial_algebraic,
    )
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver=lm_set.linear_solver,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        dual_solve_dtype=(dtype if lm_set.linear_solver == "cholesky" else None),
        implicit_penalty=implicit_penalty,
        metric=metric,
        has_aux=True,
    )

    structural_parameters = trade_growth_structural_parameters(
        k_0,
        a_0,
        parameters,
        dtype,
    )
    statistic_names = (
        *(f"log_interest_rate_t{time:g}" for time in statistic_times),
        *(f"population_weighted_log_consumption_t{time:g}" for time in statistic_times),
        *(
            f"log_consumption_{country + 1}_over_1_t{time:g}"
            for time in statistic_times
            for country in range(1, num_countries)
        ),
        *(
            f"log_capital_{country + 1}_t{time:g}"
            for time in statistic_times[1:]
            for country in range(num_countries)
        ),
        *(
            f"nfa_to_factor_income_{country + 1}_t{statistic_times[-1]:g}"
            for country in range(num_countries)
        ),
        *(
            f"current_account_to_factor_income_{country + 1}_t{statistic_times[1]:g}"
            for country in range(num_countries)
        ),
        *(
            f"log_population_weighted_home_trade_share_t{time:g}"
            for time in statistic_times
        ),
        *(
            f"log_population_weighted_import_share_t{time:g}"
            for time in statistic_times
        ),
        *(
            f"log_trade_share_{importer + 1}_from_{exporter + 1}_t{time:g}"
            for time in statistic_times
            for importer in range(num_countries)
            for exporter in range(num_countries)
        ),
    )
    parameter_names = structural_parameter_names(
        structural_parameters, off_diagonal_pairs
    )
    flat_parameters, _ = ravel_pytree(structural_parameters)
    if len(parameter_names) != flat_parameters.size:
        raise RuntimeError("structural parameter names do not match the pytree")
    parameter_coordinate_names, parameter_coordinate_scales = (
        structural_parameter_coordinates(parameter_names, structural_parameters)
    )
    return MulticountryTradeComparativeStatics(
        solver=solver,
        theta_0=theta_0,
        structural_parameters=structural_parameters,
        statistic_names=statistic_names,
        parameter_names=parameter_names,
        parameter_coordinate_names=parameter_coordinate_names,
        parameter_coordinate_scales=parameter_coordinate_scales,
        aggregate_income_scale=float(output_scale * jnp.sum(populations)),
        max_steps=lm_set.max_steps,
        residual_atol=lm_set.residual_atol,
    )


def full_structural_jacobian(problem, structural_parameters=None):
    return problem.jacobian(structural_parameters)
