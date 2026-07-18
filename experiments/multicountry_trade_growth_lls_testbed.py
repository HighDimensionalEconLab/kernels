from __future__ import annotations

import importlib
import json
import math
import pathlib
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jsonargparse
import numpy as np
from jax.flatten_util import ravel_pytree

from multicountry_trade_growth import STATIC_BLOCK_NAMES, TradeGrowthParameters
from multicountry_trade_growth_comparative_statics import (
    build_multicountry_trade_comparative_statics,
)
from multicountry_trade_growth_kernel_nlls import (
    DataSettings,
    NLLSGramSettings,
)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


@dataclass(frozen=True)
class RandomCaseSettings:
    num_countries: int = 3
    train_points: int = 5
    train_T: float = 20.0
    seed: int = 0
    heterogeneity: float = 0.06
    initial_capital_dispersion: float = 0.12
    aggregate_asset_scale: float = 0.10
    metric_eps: float = 1e-7
    damping: float = 1e-6
    linearization_steps: int = 0


@dataclass(frozen=True)
class LinearSolveSettings:
    iterative_tol: float = 1e-10
    iterative_atol: float = 0.0
    iterative_maxiter: int = 500


@dataclass(frozen=True)
class UnknownLayout:
    dynamic_block_size: int
    dynamic_function_count: int
    rate_start: int
    q_offset_start: int
    marginal_wealth_offset_start: int
    static_start: int
    total_size: int


@dataclass(frozen=True)
class LinearizedLLSCase:
    name: str
    settings: RandomCaseSettings
    parameters: TradeGrowthParameters
    k_0: tuple[float, ...]
    a_0: tuple[float, ...]
    theta: tuple[jax.Array, ...]
    theta_flat: jax.Array
    residual: jax.Array
    damping: jax.Array
    residual_time_permutation: jax.Array
    residual_native_permutation: jax.Array
    unknown_layout: UnknownLayout
    nonlinear_residual_fn: Callable
    metric_solve: Callable
    metric_norm: Callable
    jacobian_matvec: Callable
    jacobian_transpose_matvec: Callable
    dual_matvec_base: Callable
    recover_step: Callable

    @property
    def residual_size(self):
        return self.residual.size

    @property
    def parameter_size(self):
        return self.theta_flat.size

    @property
    def residual_time_block_size(self):
        return 12 * self.settings.num_countries + 1

    def to_time_major(self, value):
        return value[self.residual_time_permutation]

    def to_native_order(self, value):
        return value[self.residual_native_permutation]

    def dual_matvec_with(self, damping):
        damping_value = jnp.asarray(damping, dtype=self.residual.dtype)

        def matvec(value):
            return self.dual_matvec_base(value) + damping_value * value

        return matvec

    def dual_matvec(self, value):
        return self.dual_matvec_with(self.damping)(value)

    def dense_dual_matrix(self, damping=None):
        active_damping = self.damping if damping is None else damping
        basis = jnp.eye(self.residual_size, dtype=self.residual.dtype)
        return self.dual_matvec_with(active_damping)(basis)

    def dense_jacobian(self):
        basis = jnp.eye(self.residual_size, dtype=self.residual.dtype)
        return self.jacobian_transpose_matvec(basis).T


def random_trade_growth_inputs(settings: RandomCaseSettings):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 in the process entry point")
    if settings.num_countries < 2:
        raise ValueError("num_countries must be at least two")
    if settings.train_points < 3:
        raise ValueError("train_points must be at least three")
    if settings.train_T <= 0.0:
        raise ValueError("train_T must be positive")
    if settings.heterogeneity < 0.0:
        raise ValueError("heterogeneity must be nonnegative")
    if settings.initial_capital_dispersion < 0.0:
        raise ValueError("initial_capital_dispersion must be nonnegative")
    if settings.aggregate_asset_scale < 0.0:
        raise ValueError("aggregate_asset_scale must be nonnegative")
    if settings.metric_eps <= 0.0 or settings.damping <= 0.0:
        raise ValueError("metric_eps and damping must be positive")
    if settings.linearization_steps < 0:
        raise ValueError("linearization_steps must be nonnegative")

    num_countries = settings.num_countries
    dtype = jnp.float64
    keys = jax.random.split(jax.random.key(settings.seed), 10)

    def positive_vector(key, center=1.0):
        noise = jax.random.normal(key, (num_countries,), dtype=dtype)
        return center * jnp.exp(settings.heterogeneity * noise)

    def share_vector(key, center):
        noise = jax.random.normal(key, (num_countries,), dtype=dtype)
        center_logit = math.log(center / (1.0 - center))
        return jax.nn.sigmoid(center_logit + settings.heterogeneity * noise)

    populations = positive_vector(keys[0])
    intermediate_productivities = positive_vector(keys[1])
    consumption_productivities = positive_vector(keys[2])
    investment_productivities = positive_vector(keys[3])
    consumption_value_added_shares = share_vector(keys[4], 0.75)
    intermediate_value_added_shares = share_vector(keys[5], 0.50)
    investment_value_added_shares = share_vector(keys[6], 0.50)

    trade_noise = jax.random.normal(
        keys[7], (num_countries, num_countries), dtype=dtype
    )
    trade_center = math.log(math.expm1(0.45))
    off_diagonal_trade_costs = 1.0 + jax.nn.softplus(
        trade_center + settings.heterogeneity * trade_noise
    )
    trade_costs = jnp.where(
        jnp.eye(num_countries, dtype=jnp.bool_),
        jnp.ones_like(off_diagonal_trade_costs),
        off_diagonal_trade_costs,
    )

    capital_noise = jax.random.normal(keys[8], (num_countries,), dtype=dtype)
    k_0 = jnp.exp(settings.initial_capital_dispersion * capital_noise)
    aggregate_assets_free = settings.aggregate_asset_scale * jax.random.normal(
        keys[9], (num_countries - 1,), dtype=dtype
    )
    aggregate_assets = jnp.concatenate(
        [aggregate_assets_free, -jnp.sum(aggregate_assets_free, keepdims=True)]
    )
    a_0 = aggregate_assets / populations

    def host_tuple(value):
        return tuple(float(x) for x in np.asarray(value))

    parameters = TradeGrowthParameters(
        populations=host_tuple(populations),
        intermediate_productivities=host_tuple(intermediate_productivities),
        consumption_productivities=host_tuple(consumption_productivities),
        investment_productivities=host_tuple(investment_productivities),
        consumption_value_added_shares=host_tuple(consumption_value_added_shares),
        intermediate_value_added_shares=host_tuple(intermediate_value_added_shares),
        investment_value_added_shares=host_tuple(investment_value_added_shares),
        trade_costs=tuple(host_tuple(row) for row in np.asarray(trade_costs)),
    )
    return parameters, host_tuple(k_0), host_tuple(a_0)


def residual_time_permutations(num_countries: int, num_train_points: int):
    dynamic_size = 4 * num_countries * num_train_points
    static_size = len(STATIC_BLOCK_NAMES) * num_countries * num_train_points
    time_major = []
    for time_index in range(num_train_points):
        for dynamic_index in range(4):
            start = (
                dynamic_index * num_train_points * num_countries
                + time_index * num_countries
            )
            time_major.extend(range(start, start + num_countries))
        static_start = (
            dynamic_size + time_index * len(STATIC_BLOCK_NAMES) * num_countries
        )
        time_major.extend(
            range(
                static_start,
                static_start + len(STATIC_BLOCK_NAMES) * num_countries,
            )
        )
        time_major.append(dynamic_size + static_size + time_index)
    time_major = jnp.asarray(time_major, dtype=jnp.int32)
    native = jnp.argsort(time_major)
    return time_major, native


def build_linearized_lls_case(settings: RandomCaseSettings):
    parameters, k_0, a_0 = random_trade_growth_inputs(settings)
    middle_index = settings.train_points // 2
    middle_time = settings.train_T * middle_index / (settings.train_points - 1)
    data_set = DataSettings(
        train_T=settings.train_T,
        train_points=settings.train_points,
        sampling="uniform",
        dense_T=0.5 * settings.train_T,
        tail_points=1,
        test_T=settings.train_T,
        test_points=settings.train_points,
        validation_points=settings.train_points,
        quality_T=settings.train_T,
    )
    lm_set = NLLSGramSettings(
        init_damping=settings.damping,
        max_steps=max(1, settings.linearization_steps),
        residual_atol=0.0,
        linear_solver="cholesky",
        metric_eps=settings.metric_eps,
    )
    nonlinear_problem = build_multicountry_trade_comparative_statics(
        k_0=k_0,
        a_0=a_0,
        parameters=parameters,
        data_set=data_set,
        lm_set=lm_set,
        statistic_times=(0.0, middle_time, settings.train_T),
    )
    theta = nonlinear_problem.theta_0
    if settings.linearization_steps:
        warmup = nonlinear_problem.solver.solve(
            theta,
            p=nonlinear_problem.structural_parameters,
            max_steps=settings.linearization_steps,
            atol=0.0,
        )
        jax.block_until_ready(warmup)
        theta = warmup.x

    structural_parameters = nonlinear_problem.structural_parameters

    def nonlinear_residual_fn(theta_value):
        residual, _ = nonlinear_problem.solver.residual_fn(
            theta_value, None, structural_parameters
        )
        return jnp.ravel(residual)

    theta_flat, unravel_theta = ravel_pytree(theta)

    def residual_from_flat(flat_value):
        return nonlinear_residual_fn(unravel_theta(flat_value))

    residual, jvp_fn = jax.linearize(residual_from_flat, theta_flat)
    transpose_fn = jax.linear_transpose(jvp_fn, theta_flat)

    def apply_columns(function, value):
        if value.ndim == 1:
            return function(value)
        if value.ndim == 2:
            return jax.vmap(function, in_axes=1, out_axes=1)(value)
        raise ValueError("linear operators require a vector or column matrix")

    def jacobian_matvec(value):
        return apply_columns(jvp_fn, value)

    def transpose_vector(value):
        return transpose_fn(value)[0]

    def jacobian_transpose_matvec(value):
        return apply_columns(transpose_vector, value)

    metric_solve = nonlinear_problem.solver.metric_solve
    metric_norm = nonlinear_problem.solver.metric_norm
    damping = jnp.asarray(settings.damping, dtype=residual.dtype)

    def dual_matvec_base(value):
        transposed = jacobian_transpose_matvec(value)
        return jacobian_matvec(metric_solve(transposed))

    def recover_step(dual_solution):
        return -metric_solve(jacobian_transpose_matvec(dual_solution))

    num_countries = settings.num_countries
    num_train_points = settings.train_points
    dynamic_block_size = num_countries * num_train_points
    dynamic_size = 4 * dynamic_block_size
    rate_start = dynamic_size
    q_offset_start = rate_start + num_train_points
    marginal_wealth_offset_start = q_offset_start + num_countries
    static_start = marginal_wealth_offset_start + num_countries
    time_major, native = residual_time_permutations(num_countries, num_train_points)
    name = (
        f"N{num_countries}_nt{num_train_points}_seed{settings.seed}"
        f"_lm{settings.linearization_steps}"
    )
    return LinearizedLLSCase(
        name=name,
        settings=settings,
        parameters=parameters,
        k_0=k_0,
        a_0=a_0,
        theta=theta,
        theta_flat=theta_flat,
        residual=residual,
        damping=damping,
        residual_time_permutation=time_major,
        residual_native_permutation=native,
        unknown_layout=UnknownLayout(
            dynamic_block_size=dynamic_block_size,
            dynamic_function_count=4 * num_countries,
            rate_start=rate_start,
            q_offset_start=q_offset_start,
            marginal_wealth_offset_start=marginal_wealth_offset_start,
            static_start=static_start,
            total_size=theta_flat.size,
        ),
        nonlinear_residual_fn=nonlinear_residual_fn,
        metric_solve=metric_solve,
        metric_norm=metric_norm,
        jacobian_matvec=jacobian_matvec,
        jacobian_transpose_matvec=jacobian_transpose_matvec,
        dual_matvec_base=dual_matvec_base,
        recover_step=recover_step,
    )


def dense_cholesky(case, rhs, _):
    linear_matrix = case.dense_dual_matrix()
    factor = jsp.linalg.cho_factor(linear_matrix)
    solution = jsp.linalg.cho_solve(factor, rhs)
    finite = jnp.all(jnp.isfinite(solution))
    return solution, jnp.asarray(1, dtype=jnp.int32), finite


def cg_identity(case, rhs, settings):
    matvec = case.dual_matvec_with(case.damping)
    rhs_norm = jnp.linalg.norm(rhs)
    tolerance = jnp.maximum(
        jnp.asarray(settings.iterative_atol, dtype=rhs.dtype),
        jnp.asarray(settings.iterative_tol, dtype=rhs.dtype) * rhs_norm,
    )
    initial_solution = jnp.zeros_like(rhs)
    initial_residual = rhs
    initial_direction = initial_residual
    initial_inner = initial_residual @ initial_residual
    initial_norm = jnp.sqrt(initial_inner)
    initial_finite = jnp.all(jnp.isfinite(initial_residual))
    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        initial_solution,
        initial_residual,
        initial_direction,
        initial_inner,
        initial_norm,
        initial_finite,
    )

    def continue_iteration(state):
        iteration, _, _, _, _, residual_norm, finite = state
        return (
            (iteration < settings.iterative_maxiter)
            & (residual_norm > tolerance)
            & finite
        )

    def iteration_step(state):
        iteration, solution, residual, direction, inner, _, finite = state
        operator_direction = matvec(direction)
        denominator = direction @ operator_direction
        valid = (
            finite
            & jnp.isfinite(denominator)
            & jnp.isfinite(inner)
            & (denominator > 0.0)
            & (inner >= 0.0)
        )
        safe_denominator = jnp.where(valid, denominator, jnp.ones_like(denominator))
        step_size = inner / safe_denominator
        candidate_solution = solution + step_size * direction
        candidate_residual = residual - step_size * operator_direction
        candidate_inner = candidate_residual @ candidate_residual
        safe_inner = jnp.where(inner > 0.0, inner, jnp.ones_like(inner))
        direction_scale = candidate_inner / safe_inner
        candidate_direction = candidate_residual + direction_scale * direction
        candidate_norm = jnp.sqrt(candidate_inner)
        candidate_finite = (
            valid
            & jnp.all(jnp.isfinite(candidate_solution))
            & jnp.all(jnp.isfinite(candidate_residual))
            & jnp.isfinite(candidate_inner)
        )
        solution = jnp.where(candidate_finite, candidate_solution, solution)
        residual = jnp.where(candidate_finite, candidate_residual, residual)
        direction = jnp.where(candidate_finite, candidate_direction, direction)
        inner = jnp.where(candidate_finite, candidate_inner, inner)
        residual_norm = jnp.where(candidate_finite, candidate_norm, jnp.inf)
        return (
            iteration + 1,
            solution,
            residual,
            direction,
            inner,
            residual_norm,
            candidate_finite,
        )

    final_state = jax.lax.while_loop(
        continue_iteration,
        iteration_step,
        initial_state,
    )
    iterations, solution, _, _, _, residual_norm, finite = final_state
    converged = finite & (residual_norm <= tolerance)
    return solution, iterations, converged


@dataclass(frozen=True)
class FunctionAlgorithm:
    function: Callable

    def prepare(self, case, settings):
        return ()

    def solve(self, case, prepared, rhs, settings):
        return self.function(case, rhs, settings)


@dataclass(frozen=True)
class PreparedDenseCholesky:
    def prepare(self, case, settings):
        return jnp.linalg.cholesky(case.dense_dual_matrix())

    def solve(self, case, factor, rhs, settings):
        intermediate = jsp.linalg.solve_triangular(factor, rhs, lower=True)
        solution = jsp.linalg.solve_triangular(factor.T, intermediate, lower=False)
        finite = jnp.all(jnp.isfinite(solution))
        return solution, jnp.asarray(1, dtype=jnp.int32), finite


BUILTIN_ALGORITHMS = {
    "dense_cholesky": PreparedDenseCholesky(),
    "cg_identity": FunctionAlgorithm(cg_identity),
}


def resolve_algorithm(spec: str):
    if spec in BUILTIN_ALGORITHMS:
        return BUILTIN_ALGORITHMS[spec]
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "algorithm must be a built-in name or an import spec module:function"
        )
    algorithm = getattr(importlib.import_module(module_name), function_name)
    if hasattr(algorithm, "prepare") and hasattr(algorithm, "solve"):
        return algorithm
    if callable(algorithm):
        return FunctionAlgorithm(algorithm)
    raise TypeError(
        f"algorithm {spec} must be callable or provide prepare and solve methods"
    )


def dense_reference(case: LinearizedLLSCase):
    matrix = case.dense_dual_matrix()
    jax.block_until_ready(matrix)
    # at damping=0 the undamped dual J P J' can be singular / indefinite;
    # guard the Cholesky reference and fall back to a least-squares solve.
    min_eigenvalue = float(jnp.linalg.eigvalsh(matrix)[0])
    definite = min_eigenvalue > 0.0
    start = time.perf_counter()
    if definite:
        factor = jsp.linalg.cho_factor(matrix)
        solution = jsp.linalg.cho_solve(factor, case.residual)
    else:
        solution = jnp.linalg.lstsq(matrix, case.residual)[0]
    jax.block_until_ready(solution)
    elapsed = time.perf_counter() - start
    if not definite:
        print(
            f"  dense reference {case.name} at damping={float(case.damping):.1e} "
            f"is indefinite/singular (min eigenvalue {min_eigenvalue:.2e}); "
            f"using least-squares solve",
            flush=True,
        )
    return solution, case.recover_step(solution), elapsed, min_eigenvalue


def write_matrix_dump(case, dampings, dump_dir, max_residuals):
    dump_path = pathlib.Path(dump_dir)
    if case.residual_size > max_residuals:
        print(
            f"  skipping matrix dump {case.name}: residual size "
            f"{case.residual_size} exceeds matrix_dump_max_residuals="
            f"{max_residuals}",
            flush=True,
        )
        return None
    dump_path.mkdir(parents=True, exist_ok=True)
    damping_values = np.asarray([float(d) for d in dampings], dtype=np.float64)
    dual_matrices = np.stack(
        [np.asarray(case.dense_dual_matrix(damping=d)) for d in damping_values]
    )
    layout = case.unknown_layout
    file_path = dump_path / f"{case.name}_matrices.npz"
    np.savez_compressed(
        file_path,
        dampings=damping_values,
        dual_matrices=dual_matrices,
        jacobian=np.asarray(case.dense_jacobian()),
        residual=np.asarray(case.residual),
        time_permutation=np.asarray(case.residual_time_permutation),
        native_permutation=np.asarray(case.residual_native_permutation),
        residual_time_block_size=np.asarray(case.residual_time_block_size),
        num_countries=np.asarray(case.settings.num_countries),
        num_train_points=np.asarray(case.settings.train_points),
        seed=np.asarray(case.settings.seed),
        linearization_steps=np.asarray(case.settings.linearization_steps),
        metric_eps=np.asarray(case.settings.metric_eps),
        dynamic_block_size=np.asarray(layout.dynamic_block_size),
        dynamic_function_count=np.asarray(layout.dynamic_function_count),
        rate_start=np.asarray(layout.rate_start),
        q_offset_start=np.asarray(layout.q_offset_start),
        marginal_wealth_offset_start=np.asarray(layout.marginal_wealth_offset_start),
        static_start=np.asarray(layout.static_start),
        total_size=np.asarray(layout.total_size),
    )
    print(
        f"  wrote matrix dump {file_path} "
        f"(dual {tuple(dual_matrices.shape)}, "
        f"jacobian {case.residual_size}x{case.parameter_size})",
        flush=True,
    )
    return file_path


def benchmark_algorithm(
    case: LinearizedLLSCase,
    algorithm_spec: str,
    solve_settings: LinearSolveSettings,
    reference=None,
    timing_repeats: int = 3,
):
    if timing_repeats < 1:
        raise ValueError("timing_repeats must be positive")
    algorithm = resolve_algorithm(algorithm_spec)

    def prepare():
        return algorithm.prepare(case, solve_settings)

    compiled_prepare = jax.jit(prepare)
    setup_compile_start = time.perf_counter()
    prepared = compiled_prepare()
    jax.block_until_ready(prepared)
    setup_compile_seconds = time.perf_counter() - setup_compile_start
    setup_timings = timeit.repeat(
        lambda: jax.block_until_ready(compiled_prepare()),
        number=1,
        repeat=timing_repeats,
    )

    def solve(prepared_value, rhs):
        return algorithm.solve(case, prepared_value, rhs, solve_settings)

    compiled_solve = jax.jit(solve)
    solve_compile_start = time.perf_counter()
    output = compiled_solve(prepared, case.residual)
    jax.block_until_ready(output)
    solve_compile_seconds = time.perf_counter() - solve_compile_start
    solve_timings = timeit.repeat(
        lambda: jax.block_until_ready(compiled_solve(prepared, case.residual)),
        number=1,
        repeat=timing_repeats,
    )
    dual_solution, iterations, reported_converged = output
    step = case.recover_step(dual_solution)
    linear_residual = case.dual_matvec(dual_solution) - case.residual
    predicted_residual = case.residual + case.jacobian_matvec(step)
    rhs_norm = jnp.linalg.norm(case.residual)
    scale = jnp.maximum(rhs_norm, jnp.finfo(case.residual.dtype).tiny)
    relative_linear_residual = jnp.linalg.norm(linear_residual) / scale
    relative_predicted_residual = jnp.linalg.norm(predicted_residual) / scale
    metric_step_norm = case.metric_norm(step)
    finite = (
        jnp.all(jnp.isfinite(dual_solution))
        & jnp.all(jnp.isfinite(step))
        & jnp.isfinite(relative_linear_residual)
    )
    if reference is None:
        relative_dual_error = None
        relative_step_error = None
        reference_min_eigenvalue = None
    else:
        reference_dual, reference_step = reference[0], reference[1]
        reference_min_eigenvalue = float(reference[3])
        dual_scale = jnp.maximum(
            jnp.linalg.norm(reference_dual),
            jnp.finfo(reference_dual.dtype).tiny,
        )
        step_scale = jnp.maximum(
            jnp.linalg.norm(reference_step),
            jnp.finfo(reference_step.dtype).tiny,
        )
        relative_dual_error = float(
            jnp.linalg.norm(dual_solution - reference_dual) / dual_scale
        )
        relative_step_error = float(jnp.linalg.norm(step - reference_step) / step_scale)
    jax.block_until_ready(
        (
            relative_linear_residual,
            relative_predicted_residual,
            metric_step_norm,
            finite,
        )
    )
    return {
        "case": case.name,
        "algorithm": algorithm_spec,
        "num_countries": case.settings.num_countries,
        "num_train_points": case.settings.train_points,
        "residual_size": case.residual_size,
        "parameter_size": case.parameter_size,
        "residual_time_block_size": case.residual_time_block_size,
        "seed": case.settings.seed,
        "linearization_steps": case.settings.linearization_steps,
        "damping": float(case.damping),
        "metric_eps": case.settings.metric_eps,
        "setup_compile_seconds": setup_compile_seconds,
        "solve_compile_seconds": solve_compile_seconds,
        "compile_seconds": setup_compile_seconds + solve_compile_seconds,
        "median_setup_seconds": float(np.median(setup_timings)),
        "median_solve_seconds": float(np.median(solve_timings)),
        "median_seconds": float(np.median(setup_timings) + np.median(solve_timings)),
        "minimum_setup_seconds": float(np.min(setup_timings)),
        "minimum_solve_seconds": float(np.min(solve_timings)),
        "minimum_seconds": float(np.min(setup_timings) + np.min(solve_timings)),
        "iterations": int(iterations),
        "reported_converged": bool(reported_converged),
        "finite": bool(finite),
        "relative_linear_residual": float(relative_linear_residual),
        "relative_predicted_residual": float(relative_predicted_residual),
        "metric_step_norm": float(metric_step_norm),
        "relative_dual_error": relative_dual_error,
        "relative_step_error": relative_step_error,
        "reference_min_eigenvalue": reference_min_eigenvalue,
        "reference_available": reference is not None,
        "backend": jax.default_backend(),
        "dtype": str(case.residual.dtype),
    }


def run(
    country_counts: tuple[int, ...] = (2, 3),
    train_point_counts: tuple[int, ...] = (5,),
    seeds: tuple[int, ...] = (0,),
    algorithms: tuple[str, ...] = ("dense_cholesky", "cg_identity"),
    train_T: float = 20.0,
    heterogeneity: float = 0.06,
    initial_capital_dispersion: float = 0.12,
    aggregate_asset_scale: float = 0.10,
    metric_eps: float = 1e-7,
    damping: float = 1e-6,
    dampings: tuple[float, ...] | None = None,
    linearization_steps: int = 0,
    iterative_tol: float = 1e-10,
    iterative_atol: float = 0.0,
    iterative_maxiter: int = 500,
    timing_repeats: int = 3,
    dense_reference_max_residuals: int = 3000,
    matrix_dump_dir: str | None = None,
    matrix_dump_max_residuals: int = 3000,
    output_path: str | None = None,
):
    solve_settings = LinearSolveSettings(
        iterative_tol=iterative_tol,
        iterative_atol=iterative_atol,
        iterative_maxiter=iterative_maxiter,
    )
    damping_sweep = tuple(dampings) if dampings is not None else (damping,)
    results = []
    for num_countries in country_counts:
        for train_points in train_point_counts:
            for seed in seeds:
                case_settings = RandomCaseSettings(
                    num_countries=num_countries,
                    train_points=train_points,
                    train_T=train_T,
                    seed=seed,
                    heterogeneity=heterogeneity,
                    initial_capital_dispersion=initial_capital_dispersion,
                    aggregate_asset_scale=aggregate_asset_scale,
                    metric_eps=metric_eps,
                    damping=damping,
                    linearization_steps=linearization_steps,
                )
                build_start = time.perf_counter()
                case = build_linearized_lls_case(case_settings)
                jax.block_until_ready(case.residual)
                build_seconds = time.perf_counter() - build_start
                if matrix_dump_dir is not None:
                    write_matrix_dump(
                        case,
                        damping_sweep,
                        matrix_dump_dir,
                        matrix_dump_max_residuals,
                    )
                for damping_value in damping_sweep:
                    damped_case = replace(
                        case,
                        damping=jnp.asarray(damping_value, dtype=case.residual.dtype),
                    )
                    reference = None
                    if damped_case.residual_size <= dense_reference_max_residuals:
                        reference = dense_reference(damped_case)
                    for algorithm_spec in algorithms:
                        if (
                            algorithm_spec == "dense_cholesky"
                            and damped_case.residual_size
                            > dense_reference_max_residuals
                        ):
                            continue
                        result = benchmark_algorithm(
                            damped_case,
                            algorithm_spec,
                            solve_settings,
                            reference=reference,
                            timing_repeats=timing_repeats,
                        )
                        result["case_build_seconds"] = build_seconds
                        result["reference_compile_seconds"] = (
                            None if reference is None else reference[2]
                        )
                        results.append(result)
                        print(
                            f"{damped_case.name:20} {algorithm_spec:16} "
                            f"lam={float(damped_case.damping):.1e} "
                            f"m={damped_case.residual_size:5d} "
                            f"iters={result['iterations']:4d} "
                            f"relres={result['relative_linear_residual']:.2e} "
                            f"step_err={result['relative_step_error']} "
                            f"solve={result['median_seconds']:.4f}s",
                            flush=True,
                        )
    if output_path is not None:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


if __name__ == "__main__":
    jsonargparse.CLI(run)
