from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from extra.trade_growth.kernel import (
    DataSettings,
    KernelSettings,
    SolverSettings,
    ToleranceSettings,
    multicountry_trade_growth_kernel,
)
from extra.trade_growth.model import TradeGrowthParameters
from multicountry_growth import InitialConditionSettings


def trade_growth_statistics(values, p):
    populations = p["parameters"].populations
    population_weights = populations / jnp.sum(populations)
    log_consumption = jnp.log(values["consumption"])
    home_trade_shares = jnp.diagonal(
        values["trade_shares"],
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
    transition_index = min(1, values["interest_rate"].shape[0] - 1)
    long_index = values["interest_rate"].shape[0] - 1
    long_income = values["factor_income"][long_index]
    transition_income = values["factor_income"][transition_index]
    return jnp.concatenate(
        [
            jnp.log(values["interest_rate"]),
            jnp.sum(population_weights * log_consumption, axis=-1),
            (log_consumption[:, 1:] - log_consumption[:, :1]).reshape(-1),
            jnp.log(values["k"][1:]).reshape(-1),
            values["a"][long_index] / long_income,
            values["current_account"][transition_index] / transition_income,
            jnp.log(aggregate_home_share),
            jnp.log(aggregate_import_share),
            jnp.log(values["trade_shares"]).reshape(-1),
        ]
    )


def statistic_names(statistic_times, num_countries):
    times = tuple(float(time) for time in statistic_times)
    transition_time = times[min(1, len(times) - 1)]
    long_time = times[-1]
    return (
        *(f"log_interest_rate_t{time:g}" for time in times),
        *(f"population_weighted_log_consumption_t{time:g}" for time in times),
        *(
            f"log_consumption_{country + 1}_over_1_t{time:g}"
            for time in times
            for country in range(1, num_countries)
        ),
        *(
            f"log_capital_{country + 1}_t{time:g}"
            for time in times[1:]
            for country in range(num_countries)
        ),
        *(
            f"nfa_to_factor_income_{country + 1}_t{long_time:g}"
            for country in range(num_countries)
        ),
        *(
            f"current_account_to_factor_income_{country + 1}_t{transition_time:g}"
            for country in range(num_countries)
        ),
        *(f"log_population_weighted_home_trade_share_t{time:g}" for time in times),
        *(f"log_population_weighted_import_share_t{time:g}" for time in times),
        *(
            f"log_trade_share_{importer + 1}_from_{exporter + 1}_t{time:g}"
            for time in times
            for importer in range(num_countries)
            for exporter in range(num_countries)
        ),
    )


def structural_parameter_names(p):
    names = []
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(p)
    for path, leaf in path_leaves:
        parts = []
        for key in path:
            if hasattr(key, "key"):
                parts.append(str(key.key))
            elif hasattr(key, "name"):
                parts.append(str(key.name))
            elif hasattr(key, "idx"):
                parts.append(str(key.idx))
            else:
                parts.append(str(key))
        root = ".".join(parts)
        if not np.shape(leaf):
            names.append(root)
        else:
            names.extend(
                f"{root}[{','.join(str(index + 1) for index in indices)}]"
                for indices in np.ndindex(np.shape(leaf))
            )
    return tuple(names)


@dataclass(frozen=True)
class MulticountryTradeComparativeStatics:
    solve_parametric: Callable
    evaluate_parametric: Callable
    residual_fn: Callable
    theta_0: tuple[jax.Array, ...]
    baseline_solution: Any
    structural_parameters: dict[str, Any]
    statistic_times: jax.Array
    statistic_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_coordinate_names: tuple[str, ...]
    parameter_coordinate_scales: jax.Array
    aggregate_income_scale: float

    def solve(self, structural_parameters=None):
        p = (
            self.structural_parameters
            if structural_parameters is None
            else structural_parameters
        )
        return self.solve_parametric(p)

    def statistics(self, structural_parameters=None):
        p = (
            self.structural_parameters
            if structural_parameters is None
            else structural_parameters
        )
        solved = self.solve_parametric(p)
        values = self.evaluate_parametric(self.statistic_times, solved.x, p)
        return trade_growth_statistics(values, p)

    def directional_derivative(self, direction, structural_parameters=None):
        p = (
            self.structural_parameters
            if structural_parameters is None
            else structural_parameters
        )
        return jax.jvp(self.statistics, (p,), (direction,))

    def jacobian(self, structural_parameters=None):
        p = (
            self.structural_parameters
            if structural_parameters is None
            else structural_parameters
        )
        flat_p, unravel_p = ravel_pytree(p)

        def statistics_from_flat(flat_values):
            return self.statistics(unravel_p(flat_values))

        statistics, pullback = jax.vjp(statistics_from_flat, flat_p)
        basis = jnp.eye(statistics.size, dtype=statistics.dtype)
        jacobian = jax.vmap(lambda row: pullback(row)[0])(basis)
        return statistics, jacobian

    def interpretable_jacobian(self, structural_parameters=None):
        statistics, jacobian = self.jacobian(structural_parameters)
        return statistics, jacobian * self.parameter_coordinate_scales[None, :]


def build_multicountry_trade_comparative_statics(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: TradeGrowthParameters = TradeGrowthParameters(),
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    statistic_times=(0.0, 20.0, 40.0),
):
    results, extra = multicountry_trade_growth_kernel(
        k_0=k_0,
        a_0=a_0,
        parameters=parameters,
        use_float64=use_float64,
        initial_set=initial_set,
        kernel_set=kernel_set,
        data_set=data_set,
        lm_set=lm_set,
        tolerance_set=tolerance_set,
    )
    dtype = extra["train_data"].dtype
    times = jnp.asarray(statistic_times, dtype=dtype)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("statistic_times must contain at least two dates")
    if bool(jnp.any(jnp.diff(times) <= 0.0)):
        raise ValueError("statistic_times must be strictly increasing")
    if float(times[0]) < 0.0 or float(times[-1]) > float(extra["train_data"][-1]):
        raise ValueError("statistic_times must lie within the training horizon")
    p = extra["p"]
    values = extra["evaluate_parametric"](times, extra["theta"], p)
    names = structural_parameter_names(p)
    flat_p, _ = ravel_pytree(p)
    if len(names) != flat_p.size:
        raise RuntimeError("parameter names do not match the p pytree")
    additive = {
        "a_0",
        "parameters.alpha",
        "parameters.consumption_value_added_shares",
        "parameters.installation_elasticity",
        "parameters.intermediate_value_added_shares",
        "parameters.investment_value_added_shares",
    }
    coordinate_names = []
    coordinate_scales = []
    for index, name in enumerate(names):
        root = name.split("[", maxsplit=1)[0]
        if root in additive:
            coordinate_names.append(name)
            coordinate_scales.append(1.0)
        else:
            coordinate_names.append(f"log_{name}")
            coordinate_scales.append(flat_p[index])
    baseline_solution = extra["result"]
    return MulticountryTradeComparativeStatics(
        solve_parametric=extra["solve_parametric"],
        evaluate_parametric=extra["evaluate_parametric"],
        residual_fn=extra["residual_fn"],
        theta_0=extra["theta_0"],
        baseline_solution=baseline_solution,
        structural_parameters=p,
        statistic_times=times,
        statistic_names=statistic_names(times, parameters.num_countries),
        parameter_names=names,
        parameter_coordinate_names=tuple(coordinate_names),
        parameter_coordinate_scales=jnp.asarray(coordinate_scales, dtype=dtype),
        aggregate_income_scale=float(values["world_gdp"][0]),
    )


def full_structural_jacobian(problem, structural_parameters=None):
    return problem.jacobian(structural_parameters)
