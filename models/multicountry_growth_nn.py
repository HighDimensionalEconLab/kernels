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
    QR,
    Cholesky,
    DrawNNXModule,
    LevenbergMarquardt,
    MultiStart,
)

from models.multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    multicountry_long_run_diagnostics,
    multicountry_output_scale,
    multicountry_path_values,
    multicountry_reference_mu,
    multicountry_steady_state,
    multicountry_tail_diagnostics,
    multicountry_tvc_diagnostics,
)
from models.multicountry_growth_benchmark import multicountry_growth_benchmark
from models.utilities import (
    ACTIVATIONS,
    STATUS_TO_REASON,
    Stopwatch,
    make_sampled_train_grid,
    unit_softplus,
)

# Validity of the inputs, not an acceptance gate: checked in exact float64 on
# the raw values, so it holds at any working dtype.
INITIAL_CLEARING_TOL = 1e-10


# Equation-diagnostic gates, CLI-reachable so a float32 caller can pass its
# own bar (the float32 replication driver does). The train gate is calibrated
# for the drawn heterogeneous initial conditions (measured 1.17e-5 max train
# residual at the default seed and training budget; 1e-5 was the
# symmetric-k_0 margin).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 2e-5
    validation_residual: float = 5e-3
    accounting: float = 5e-5
    benchmark_match: float = 2e-3


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 160.0
    train_points: int = 96
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 15
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401
    benchmark_T: float = 200.0
    benchmark_points: int = 241
    # Long-run validity checks (enabled by the long_run_diagnostics flag)
    # are evaluated at t ~ long_run_check_T, with flatness measured against
    # t = long_run_check_T - long_run_window.
    long_run_check_T: float = 100.0
    long_run_window: float = 10.0


@dataclass(frozen=True)
class SequentialNNSettings:
    # None resolves width/depth from the country count inside the driver
    # (verified through N=10; the NN is the small/moderate-N baseline and
    # the kernel cg path is the scaling vehicle); explicit values pin the
    # architecture exactly.
    width: int | None = None
    depth: int | None = None
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
    linear_solver: str = "auto"


@dataclass(frozen=True)
class StoppingCriteria:
    max_steps: int = 250
    train_loss_threshold: float = 1e-12
    num_attempts: int = 5


@struct.dataclass
class ResidualArgs:
    graphdef: nnx.GraphDef
    train_data: jax.Array
    k_0: jax.Array
    a_0: jax.Array
    q_reference: jax.Array
    mu_reference: jax.Array
    output_scale: jax.Array
    parameters: MulticountryParameters = struct.field(pytree_node=False)


class MulticountryGrowthNN(nnx.Module):
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
            4 * num_countries,
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


def policy_coordinates(
    policy,
    points,
    k_0,
    a_0,
    q_reference,
    mu_reference,
    output_scale,
    parameters,
):
    def decoded_path(t):
        raw = policy(t).reshape(4, policy.num_countries)
        time_coordinate = t / policy.input_scale
        return (
            k_0 * jnp.exp(time_coordinate * raw[0]),
            a_0 + output_scale * time_coordinate * raw[1],
            q_reference * unit_softplus(raw[2]),
            mu_reference * unit_softplus(raw[3]),
        )

    values, derivatives = jax.vmap(
        lambda t: jax.jvp(decoded_path, (t,), (jnp.ones_like(t),))
    )(points)
    k, a, q, mu = values
    dk_dt, da_dt, dq_dt, dm_dt = derivatives
    dlog_mu_dt = dm_dt / mu
    interest_rate = parameters.r - jnp.mean(dlog_mu_dt, axis=1)
    return {
        "k": k,
        "a": a,
        "q": q,
        "mu": mu,
        "interest_rate": interest_rate,
        "dlog_k_dt": dk_dt / k,
        "da_dt": da_dt,
        "dlog_q_dt": dq_dt / q,
        "dlog_mu_dt": dlog_mu_dt,
    }


def evaluate_policy(
    policy,
    points,
    k_0,
    a_0,
    q_reference,
    mu_reference,
    output_scale,
    parameters,
):
    coordinates = policy_coordinates(
        policy,
        points,
        k_0,
        a_0,
        q_reference,
        mu_reference,
        output_scale,
        parameters,
    )
    return multicountry_path_values(**coordinates, parameters=parameters)


def residual_fn(theta, args):
    policy = nnx.merge(args.graphdef, theta)
    values = evaluate_policy(
        policy,
        args.train_data,
        args.k_0,
        args.a_0,
        args.q_reference,
        args.mu_reference,
        args.output_scale,
        args.parameters,
    )
    # No algebraic row contains the rate (clearing is a stock condition on
    # the states, so the DAE has index > 1); the rate path is disciplined by
    # the assets and Euler rows. The t=0 clearing row is identically zero
    # because a_0 clears, and is kept for uniformity.
    residual = jnp.concatenate(
        [
            values["capital"].reshape(-1),
            values["assets"].reshape(-1),
            values["tobin_q"].reshape(-1),
            values["marginal_wealth_euler"].reshape(-1),
            values["asset_clearing"],
        ]
    )
    return residual / jnp.sqrt(residual.size)


def multicountry_growth_nn(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    seed: int = 0,
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    data_set: DataSettings = DataSettings(),
    nn_set: SequentialNNSettings = SequentialNNSettings(),
    lm_set: SolverSettings = SolverSettings(),
    stop_set: StoppingCriteria = StoppingCriteria(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    long_run_diagnostics: bool = False,
    verbose: bool = True,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    num_countries = parameters.num_countries
    # Identical initial conditions are degenerate (no cross-country
    # variation), so unspecified k_0/a_0 are DRAWN from the seeded
    # initial_set distributions rather than defaulted to symmetric values.
    if k_0 is None or a_0 is None:
        drawn_k_0, drawn_a_0 = draw_initial_conditions(initial_set, parameters)
        if k_0 is None:
            k_0 = drawn_k_0
        if a_0 is None:
            a_0 = drawn_a_0
    if len(k_0) != num_countries or any(value <= 0.0 for value in k_0):
        raise ValueError("k_0 must contain one positive capital stock per country")
    if len(a_0) != num_countries:
        raise ValueError("a_0 must contain one NFA position per country")
    # Unpinned capacity resolves from the country count over the NN's
    # intended small/moderate-N domain, where "enough" means the fixed
    # 2e-5 train gate. Measured (seed 0, drawn ICs, symmetric parameters):
    # 32/3 passes N<=3 but reaches 2.03e-5 at N=10, and 32/4 passes N=10
    # at 1.42e-5. The NN baseline is deliberately NOT the large-N vehicle:
    # past N~10 the rms early stop caps quality below the max-residual
    # gate (measured 2.06e-5 at N=25 even at width 48), the dense LM cost
    # grows with the iterate-dependent Jacobian (9,697 x 12,052 at N=25
    # width 48), and no kernel-style preconditioner structure exists --
    # use the kernel driver's cg path for scale, and pin width/depth
    # explicitly to experiment beyond this range.
    if nn_set.width is None or nn_set.depth is None:
        nn_set = replace(
            nn_set,
            width=nn_set.width if nn_set.width is not None else 32,
            depth=(
                nn_set.depth
                if nn_set.depth is not None
                else (3 if num_countries <= 3 else 4)
            ),
        )
    if nn_set.width <= 0 or nn_set.depth <= 0:
        raise ValueError("NN width and depth must be positive")
    if nn_set.input_scale <= 0.0 or nn_set.head_scale <= 0.0:
        raise ValueError("NN scales must be positive")
    if stop_set.train_loss_threshold <= 0.0 or stop_set.num_attempts < 1:
        raise ValueError("stopping threshold and attempt count must be positive")

    train_data = make_sampled_train_grid(data_set, dtype)
    if data_set.test_T > float(train_data[-1]):
        raise ValueError("test_T must not exceed the training horizon")
    if data_set.benchmark_T < max(data_set.test_T, float(train_data[-1])):
        raise ValueError("benchmark_T must cover the train and test horizons")
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0,
        float(train_data[-1]),
        data_set.validation_points,
        dtype=dtype,
    )
    benchmark_grid = np.linspace(
        0.0,
        data_set.benchmark_T,
        data_set.benchmark_points,
    )
    n_train = len(train_data)
    k_0_array = jnp.asarray(k_0, dtype=dtype)
    populations_np = np.asarray(parameters.populations, dtype=np.float64)
    initial_asset_gap = float(
        np.sum(populations_np * np.asarray(a_0, dtype=np.float64))
        / float(multicountry_output_scale(parameters, jnp.float64))
    )
    if abs(initial_asset_gap) > INITIAL_CLEARING_TOL:
        raise ValueError("population-weighted initial NFA positions must sum to zero")
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    output_scale = multicountry_output_scale(parameters, dtype)
    # Demeaning in the working dtype clears the cast residue a float32 copy
    # of exactly-clearing float64 positions carries.
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    a_0_array = a_0_array - jnp.sum(populations * a_0_array) / jnp.sum(populations)
    q_reference = jnp.full(
        num_countries,
        1.0 / parameters.eta,
        dtype=dtype,
    )
    mu_reference = multicountry_reference_mu(
        parameters,
        dtype,
    )

    graphdef, _ = nnx.split(
        MulticountryGrowthNN(
            num_countries,
            nn_set,
            dtype=dtype,
            rngs=nnx.Rngs(0),
        ),
        nnx.Param,
    )
    init_key, multi_start_key = jax.random.split(jax.random.key(seed))
    model = MulticountryGrowthNN(
        num_countries,
        nn_set,
        dtype=dtype,
        rngs=nnx.Rngs(init_key),
    )
    _, theta_0 = nnx.split(model, nnx.Param)
    residual_args = ResidualArgs(
        graphdef=graphdef,
        train_data=train_data,
        k_0=k_0_array,
        a_0=a_0_array,
        q_reference=q_reference,
        mu_reference=mu_reference,
        output_scale=output_scale,
        parameters=parameters,
    )
    solver = LevenbergMarquardt(
        residual_fn,
        init_damping=lm_set.init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
        geodesic_acceleration=lm_set.geodesic_acceleration,
        geodesic_acceptance_ratio=lm_set.geodesic_acceptance_ratio,
        linear_solver={"auto": Cholesky(), "cholesky": Cholesky(), "qr": QR()}[
            lm_set.linear_solver
        ],
    )
    draw = DrawNNXModule(
        MulticountryGrowthNN,
        num_countries,
        nn_set,
        dtype=dtype,
    )
    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        residual_args,
        max_steps=stop_set.max_steps,
        atol=math.sqrt(stop_set.train_loss_threshold),
        max_steps_is_success=False,
        multi_start=MultiStart(
            key=multi_start_key,
            num_starts=stop_set.num_attempts,
            draw=draw,
        ),
    )
    train_time = timer.elapsed(result)
    policy = nnx.merge(graphdef, result.x)

    def evaluate_grid(points):
        return evaluate_policy(
            policy,
            points,
            k_0_array,
            a_0_array,
            q_reference,
            mu_reference,
            output_scale,
            parameters,
        )

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)
    jax.block_until_ready((train_eval, validation_eval, test_eval))
    residual_names = (
        "capital",
        "assets",
        "tobin_q",
        "marginal_wealth_euler",
    )
    train_global_constraints = train_eval["asset_clearing"]
    max_train_residual = max(
        max(float(jnp.max(jnp.abs(train_eval[name]))) for name in residual_names),
        float(jnp.max(jnp.abs(train_global_constraints))),
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name])))
        for name in (*residual_names, "resource")
    )
    max_asset_clearing_residual = float(
        jnp.max(jnp.abs(validation_eval["asset_clearing"]))
    )
    max_rate_consistency_residual = float(
        jnp.max(jnp.abs(validation_eval["interest_rate_country_gap"]))
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "x", "marginal_wealth")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account", "interest_rate")
        )
    )
    diagnostic_reasons = []
    if not finite_positive:
        diagnostic_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        diagnostic_reasons.append("train_residual")
    if max_validation_residual > tolerance_set.validation_residual:
        diagnostic_reasons.append("validation_residual")
    if max_asset_clearing_residual > tolerance_set.accounting:
        diagnostic_reasons.append("asset_clearing")
    if max_rate_consistency_residual > tolerance_set.accounting:
        diagnostic_reasons.append("rate_consistency")
    solution_accepted = bool(result.multi_start.accepted)
    equation_diagnostics_pass = not diagnostic_reasons

    benchmark_available = num_countries == 2
    benchmark_values = None
    benchmark_diagnostics = None
    if benchmark_available:
        benchmark, benchmark_diagnostics = multicountry_growth_benchmark(
            k_0,
            (a_0[0],),
            benchmark_grid,
            parameters=parameters,
        )
        benchmark_values = benchmark(np.asarray(test_data))
        k_relative_error = jnp.abs(
            test_eval["k"] - jnp.asarray(benchmark_values["k"])
        ) / jnp.asarray(benchmark_values["k"])
        q_relative_error = jnp.abs(
            test_eval["q"] - jnp.asarray(benchmark_values["q"])
        ) / jnp.asarray(benchmark_values["q"])
        c_relative_error = jnp.abs(
            test_eval["c"] - jnp.asarray(benchmark_values["c"])
        ) / jnp.asarray(benchmark_values["c"])
        x_relative_error = jnp.abs(
            test_eval["x"] - jnp.asarray(benchmark_values["x"])
        ) / jnp.asarray(benchmark_values["x"])
        a_scaled_error = (
            jnp.abs(test_eval["a"] - jnp.asarray(benchmark_values["a"])) / output_scale
        )
        interest_rate_scaled_error = (
            jnp.abs(
                test_eval["interest_rate"]
                - jnp.asarray(benchmark_values["interest_rate"])
            )
            / parameters.r
        )
        max_benchmark_error = max(
            float(jnp.max(k_relative_error)),
            float(jnp.max(q_relative_error)),
            float(jnp.max(c_relative_error)),
            float(jnp.max(x_relative_error)),
            float(jnp.max(a_scaled_error)),
            float(jnp.max(interest_rate_scaled_error)),
        )
        benchmark_match = max_benchmark_error <= tolerance_set.benchmark_match
    else:
        k_relative_error = jnp.asarray(jnp.nan)
        q_relative_error = jnp.asarray(jnp.nan)
        c_relative_error = jnp.asarray(jnp.nan)
        x_relative_error = jnp.asarray(jnp.nan)
        a_scaled_error = jnp.asarray(jnp.nan)
        interest_rate_scaled_error = jnp.asarray(jnp.nan)
        max_benchmark_error = float("nan")
        benchmark_match = False

    resolved_steady = multicountry_steady_state(
        validation_eval["marginal_wealth"][-1],
        parameters,
    )
    tail = multicountry_tail_diagnostics(validation_eval, parameters)
    tvc = multicountry_tvc_diagnostics(validation_data, validation_eval, parameters)
    long_run = (
        multicountry_long_run_diagnostics(
            validation_data,
            validation_eval,
            a_0_array,
            parameters,
            check_T=data_set.long_run_check_T,
            window=data_set.long_run_window,
        )
        if long_run_diagnostics
        else None
    )
    relative_weights = test_eval["relative_welfare_weights"][0]
    log_relative_weights = jnp.log(relative_weights)
    theta_initial, _ = ravel_pytree(theta_0)
    theta_final, _ = ravel_pytree(result.x)
    residual_count = (4 * num_countries + 1) * n_train
    results = {
        **asdict(parameters),
        "num_countries": num_countries,
        "seed": seed,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["capital"].dtype),
        "solver": "nlls-gram",
        "linear_solver": lm_set.linear_solver,
        "ansatz": "reduced_sequential_time_level_nn",
        "selection": "minimum_identity_metric_parameter_step",
        "metric": "identity_parameter_metric",
        "initialization": "steady_reference_near_zero_head",
        "global_closure": "explicit_assets_and_index_consistent_clearing",
        "terminal_residual_count": 0,
        "benchmark_training_residual_count": 0,
        "tvc_training_residual_count": 0,
        "dae_residual_count": residual_count,
        "dynamic_network_output_count": 4 * num_countries,
        "calculated_algebraic_value_count": 2 * num_countries + 1,
        "unknown_count": int(theta_final.size),
        "total_params": int(theta_final.size),
        "parameter_norm_initial": float(jnp.linalg.norm(theta_initial)),
        "parameter_norm_final": float(jnp.linalg.norm(theta_final)),
        "parameter_step_norm": float(jnp.linalg.norm(theta_final - theta_initial)),
        "nn_width": nn_set.width,
        "nn_depth": nn_set.depth,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "train_loss": float(result.info.loss),
        "final_damping": float(result.lm_state.damping),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "max_asset_clearing_residual": max_asset_clearing_residual,
        "max_rate_consistency_residual": max_rate_consistency_residual,
        "benchmark_available": benchmark_available,
        "benchmark_match": benchmark_match,
        "max_benchmark_error": max_benchmark_error,
        "k_relative_error_max": float(jnp.max(k_relative_error)),
        "q_relative_error_max": float(jnp.max(q_relative_error)),
        "c_relative_error_max": float(jnp.max(c_relative_error)),
        "x_relative_error_max": float(jnp.max(x_relative_error)),
        "a_scaled_error_max": float(jnp.max(a_scaled_error)),
        "interest_rate_scaled_error_max": float(jnp.max(interest_rate_scaled_error)),
        "ex_post_k_log_gap": float(jnp.max(tail["k_log_gap"])),
        "ex_post_q_log_gap": float(jnp.max(tail["q_log_gap"])),
        "ex_post_asset_scaled_gap": float(jnp.max(tail["asset_scaled_gap"])),
        "ex_post_interest_rate_gap": float(tail["interest_rate_gap"]),
        "ex_post_discounted_assets": float(jnp.max(tvc["discounted_assets"])),
        "ex_post_asset_tvc": float(jnp.max(tvc["asset_tvc"])),
        "ex_post_capital_tvc": float(jnp.max(tvc["capital_tvc"])),
        "steps": int(result.steps),
        "attempt": int(result.multi_start.attempt) + 1,
        "stopping_reason": STATUS_TO_REASON.get(int(result.status), "unknown"),
        "train_time": train_time,
        "train_time_success": train_time if solution_accepted else np.nan,
        "wall_time": train_time,
        "solution_accepted": solution_accepted,
        "rejection_reason": (
            "accepted"
            if solution_accepted
            else STATUS_TO_REASON.get(int(result.status), "training_failure")
        ),
        "equation_diagnostics_pass": equation_diagnostics_pass,
        "equation_diagnostic_reason": (
            "passed" if equation_diagnostics_pass else ",".join(diagnostic_reasons)
        ),
    }
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"population_{suffix}": parameters.populations[country],
                f"productivity_{suffix}": parameters.productivities[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
                f"a_0_{suffix}_over_output_ss": a_0[country] / float(output_scale),
                f"relative_weight_{suffix}": float(relative_weights[country]),
                f"log_relative_weight_{suffix}": float(log_relative_weights[country]),
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"long_run_a_{suffix}": float(resolved_steady["a"][country]),
            }
        )
    if long_run is not None:
        results.update(
            {
                "long_run_check_T": data_set.long_run_check_T,
                "long_run_window": data_set.long_run_window,
                "long_run_c_flatness": float(long_run["c_flatness"]),
                "long_run_k_flatness": float(long_run["k_flatness"]),
                "long_run_a_flatness": float(long_run["a_flatness"]),
                "long_run_k_theory_gap": float(long_run["k_theory_gap"]),
                "long_run_mpk_equalization_gap": float(
                    long_run["mpk_equalization_gap"]
                ),
                "long_run_interest_rate_gap": float(long_run["interest_rate_gap"]),
                "long_run_consumption_ratio_drift": float(
                    long_run["consumption_ratio_drift"]
                ),
                "long_run_wealth_proportionality_gap": float(
                    long_run["wealth_proportionality_gap"]
                ),
                "long_run_wealth_rank_violation": float(
                    long_run["wealth_rank_violation"]
                ),
                "long_run_c_rank_matches_wealth": long_run["c_rank_matches_wealth"],
                "long_run_c_rank_matches_a_0": long_run["c_rank_matches_a_0"],
                "long_run_terminal_a_rank_matches_a_0": long_run[
                    "terminal_a_rank_matches_a_0"
                ],
            }
        )
        for country in range(num_countries):
            suffix = country + 1
            results[f"long_run_c_{suffix}"] = float(resolved_steady["c"][country])
            results[f"initial_wealth_{suffix}"] = float(
                long_run["initial_wealth"][country]
            )
    if benchmark_available:
        results.update(
            {
                "log_relative_weight": float(log_relative_weights[1]),
                "relative_weight": float(relative_weights[1]),
                "benchmark_log_relative_weight": benchmark_diagnostics[
                    "log_relative_weight"
                ],
                "benchmark_boundary_residual": benchmark_diagnostics[
                    "boundary_residual_max"
                ],
                "stable_roots": benchmark_diagnostics["stable_roots"],
                "unstable_roots": benchmark_diagnostics["unstable_roots"],
                "center_roots": benchmark_diagnostics["center_roots"],
                "minimum_rate_denominator": benchmark_diagnostics[
                    "minimum_rate_denominator"
                ],
            }
        )

    def frame(values, points):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "resource_residual": np.asarray(values["resource"]),
            "asset_clearing_residual": np.asarray(values["asset_clearing"]),
            "interest_rate_country_gap": np.asarray(
                jnp.max(jnp.abs(values["interest_rate_country_gap"]), axis=1)
            ),
        }
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "x",
                "output",
                "resources",
                "current_account",
                "capital",
                "assets",
                "tobin_q",
                "marginal_wealth",
                "marginal_wealth_euler",
                "relative_welfare_weights",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        return pd.DataFrame(data)

    train_df = frame(train_eval, train_data)
    validation_df = frame(validation_eval, validation_data)
    test_df = frame(test_eval, test_data)
    if benchmark_available:
        for country in range(num_countries):
            suffix = country + 1
            for name in ("k", "a", "q", "c", "x", "output", "current_account"):
                test_df[f"{name}_{suffix}_benchmark"] = benchmark_values[name][
                    :, country
                ]
        test_df["interest_rate_benchmark"] = benchmark_values["interest_rate"]

    hyperparameters = {
        "parameters": asdict(parameters),
        "data_set": asdict(data_set),
        "nn_set": asdict(nn_set),
        "lm_set": asdict(lm_set),
        "stop_set": asdict(stop_set),
        "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
    }
    if verbose:
        print("\nFinal results:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    return results, {
        "model": policy,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
    }


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_growth_nn)
