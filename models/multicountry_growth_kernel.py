from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jsonargparse
import numpy as np
import pandas as pd
from nlls_gram import (
    CG,
    QR,
    AnnealRidge,
    Cholesky,
    LMStatus,
    RidgeLevenbergMarquardt,
)

from models.kernels import (
    KernelSettings,
    integrated_kernel_matrices,
    integrated_metric_factorization,
    normalize_kernel,
    rkhs_norm_squared,
)
from models.multicountry_growth import (
    InitialConditionSettings,
    MulticountryParameters,
    draw_initial_conditions,
    investment_expenditure,
    long_run_diagnostics,
    multicountry_parameter_values,
    path_residuals,
    production,
    relative_welfare_weights,
    resolved_steady_state,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
    validate_initial_conditions,
)
from models.multicountry_growth_preconditioner import (
    ArrowWoodburyCallback,
    build_arrow_woodbury,
)
from models.utilities import (
    Stopwatch,
    acceptance_fields,
    make_sampled_train_grid,
)

RESIDUAL_NAMES = (
    "capital",
    "assets",
    "tobin_q",
    "marginal_wealth",
    "consumption_foc",
    "asset_clearing",
)


class InitialValues(NamedTuple):
    q_offsets: jax.Array
    c_offsets: jax.Array
    mu_offsets: jax.Array
    rate_offset: jax.Array


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# the (N, n_train) blocks ravel country-major.
class Theta(NamedTuple):
    alpha_k: jax.Array
    alpha_a: jax.Array
    alpha_q: jax.Array
    alpha_c: jax.Array
    alpha_mu: jax.Array
    alpha_rate: jax.Array
    initial: InitialValues


# Numerical acceptance and ex-post quality thresholds, CLI-reachable so a
# float32 caller can pass its own bars (the float32 replication driver does).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    # The validation threshold is an HPO/quality diagnostic; it does not
    # redefine convergence of a finite, feasible numerical solution.
    validation_residual: float = 1e-2


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 160.0
    train_points: int = 192
    train_points_list: list[float] | None = None
    sampling: str = "front_loaded"
    dense_T: float = 40.0
    tail_points: int = 15
    test_T: float = 50.0
    test_points: int = 201
    validation_points: int = 401
    # Long-run validity checks are evaluated at t ~ long_run_check_T (flat
    # consumption/capital/debt, capital at the theory steady state, interest
    # rate ~ r), with flatness measured against
    # t = long_run_check_T - long_run_window.
    long_run_check_T: float = 100.0
    long_run_window: float = 10.0


@dataclass(frozen=True)
class SolverSettings:
    init_damping: float = 1e-8
    damping_decrease: float = 0.1
    damping_increase: float = 4.0
    max_steps: int = 100
    # Conjunctive stopping in the whitened variable y = F_bar theta the
    # solver works in: gtol bounds the whitened stationarity, residual_atol
    # additionally demands the scaled residual norm solved (the converged
    # residual floors at O(ridge * ||dual||), so residual_atol sits above
    # it). The ridge keeps the minimizer O(ridge) from the pure
    # RKHS-seminorm interpolant with the free offsets unpenalized.
    residual_atol: float = 5e-7
    ridge: float = 3e-12
    gtol: float = 5e-8
    # ridge_init=None solves at the fixed ridge above; a value enables ridge
    # continuation annealed from ridge_init down to `ridge` (the floor) by
    # factors of `ridge_decrease` on inner stationarity.
    ridge_init: float | None = 1e-7
    ridge_decrease: float = 0.01
    # Relative stationarity threshold that triggers the next ridge level:
    # looser anneals faster (fewer steps per level), tighter solves each
    # level out more completely before descending.
    ridge_grad_rtol: float = 1e-2
    # "auto" (whitened cholesky) and "qr" solve dense whitened systems; "cg"
    # is matrix-free preconditioned CG on the whitened normal operator with
    # the arrow+Woodbury preconditioner rebuilt every LM step (derivation in
    # models/multicountry_growth_preconditioner.py). Implicit-AD re-solves
    # stay dense-only: the undamped system sits at the float64 conditioning
    # plateau, where CG would silently degrade the tangents.
    linear_solver: str = "auto"
    # cg_tol is unreachable in the endgame, so cg_maxiter is the effective
    # budget there; cg_damping_floor keeps the preconditioner's block
    # Cholesky positive definite while the live LM damping anneals freely.
    cg_tol: float = 1e-10
    cg_maxiter: int = 40
    cg_damping_floor: float = 1e-10


# Module-level so callers can evaluate solved paths on their own kernel
# matrices.
def path_coordinates(theta, K_eval, K_tilde_eval, p):
    parameters_p = p["parameters"]
    populations_p = parameters_p.populations
    k_0_p = p["k_0"]
    # Population demeaning is a no-op at the primal point (the inputs
    # already clear) and projects AD perturbations of a_0 onto the
    # feasible clearing set, so every direction is meaningful.
    a_0_p = p["a_0"] - jnp.sum(populations_p * p["a_0"]) / jnp.sum(populations_p)
    steady_p = steady_state(parameters_p, k_0_p.dtype)
    # The asset block is the one path with no unpenalized offset, so its
    # coefficient scale enters the minimum-norm selection metric; the
    # scale must be differentiation-inert or every parameter sensitivity
    # picks up a selection-drift term from the metric moving with p.
    # The other normalizations are absorbed by their free offsets and
    # need no such guard.
    output_scale_p = jax.lax.stop_gradient(steady_p["output_scale"])
    log_consumption_scale_p = jnp.log(steady_p["consumption_scale"])
    alpha_k, alpha_a, alpha_q, alpha_c, alpha_mu, alpha_rate = (
        theta.alpha_k,
        theta.alpha_a,
        theta.alpha_q,
        theta.alpha_c,
        theta.alpha_mu,
        theta.alpha_rate,
    )
    q_offsets = theta.initial.q_offsets
    c_offsets = theta.initial.c_offsets
    mu_offsets = theta.initial.mu_offsets
    rate_offset = theta.initial.rate_offset
    log_k = jnp.log(k_0_p) + K_tilde_eval @ alpha_k.T
    a = a_0_p + output_scale_p * (K_tilde_eval @ alpha_a.T)
    log_q = -jnp.log(parameters_p.eta) + q_offsets + K_tilde_eval @ alpha_q.T
    log_c = log_consumption_scale_p + c_offsets + K_tilde_eval @ alpha_c.T
    log_mu = (
        -parameters_p.gammas * log_consumption_scale_p
        + mu_offsets
        + K_tilde_eval @ alpha_mu.T
    )
    log_rate = jnp.log(parameters_p.r) + rate_offset + K_tilde_eval @ alpha_rate
    return {
        "k": jnp.exp(log_k),
        "a": a,
        "q": jnp.exp(log_q),
        "c": jnp.exp(log_c),
        "mu": jnp.exp(log_mu),
        "interest_rate": jnp.exp(log_rate),
        "dlog_k_dt": K_eval @ alpha_k.T,
        "da_dt": output_scale_p * (K_eval @ alpha_a.T),
        "dlog_q_dt": K_eval @ alpha_q.T,
        "dlog_mu_dt": K_eval @ alpha_mu.T,
    }


def multicountry_growth_kernel(
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    parameters: MulticountryParameters = MulticountryParameters(),
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: KernelSettings = KernelSettings(ell=20.0),
    data_set: DataSettings = DataSettings(),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
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
    validate_initial_conditions(k_0, a_0, parameters)

    kernel = normalize_kernel(kernel_set.kernel)
    train_data = make_sampled_train_grid(data_set, dtype)
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_data)
    populations = jnp.asarray(parameters.populations, dtype=dtype)
    total_population = jnp.sum(populations)
    steady = steady_state(parameters, dtype)
    output_scale = steady["output_scale"]

    # There are (5N+1)n DAE equations in (5N+1)n + 3N + 1 unknowns. No
    # terminal equation is appended: the ridge penalty supplies the
    # (3N+1)-dimensional selection. Every linear solver works on the
    # whitened subproblem; the metric takes the upper Cholesky factor F with
    # K = F'F and the trailing offsets stay in the unpenalized free block,
    # untransformed.
    num_rkhs_functions = 5 * num_countries + 1
    num_initial_values = 3 * num_countries + 1
    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, kernel_set.sigma, kernel_set.ell, repeats=num_rkhs_functions
    )

    # The structural parameters and initial conditions ride in the solver's
    # `p` argument so the solved paths are implicitly differentiable with
    # respect to every one of them (nlls-gram's implicit JVP/VJP at the
    # ridge stationary point). All parameter-dependent quantities the paths
    # consume are recomputed from p inside.
    # Demeaning clears the sub-tolerance clearing residue the validation
    # admits; the demeaned values are what the paths anchor to, so they are
    # also what the benchmark, results, and hyperparameters report.
    a_0_array = jnp.asarray(a_0, dtype=dtype)
    a_0_array = a_0_array - jnp.sum(populations * a_0_array) / total_population
    a_0 = tuple(float(value) for value in a_0_array)
    p = {
        "parameters": multicountry_parameter_values(parameters, dtype),
        "k_0": jnp.asarray(k_0, dtype=dtype),
        "a_0": a_0_array,
    }

    def coordinate_residuals(coordinates, parameters_p):
        return path_residuals(
            coordinates["k"],
            coordinates["a"],
            coordinates["q"],
            coordinates["c"],
            coordinates["mu"],
            parameters_p.productivities,
            coordinates["interest_rate"],
            coordinates["dlog_k_dt"],
            coordinates["da_dt"],
            coordinates["dlog_q_dt"],
            coordinates["dlog_mu_dt"],
            parameters_p,
        )

    residual_count = (5 * num_countries + 1) * n_train
    residual_scale = 1.0 / jnp.sqrt(float(residual_count))

    # p carries the structural parameters and initial conditions for
    # implicit differentiation; the model equations never read args.
    def residual_fn(theta, _, p):
        coordinates = path_coordinates(theta, K, K_tilde, p)
        residuals = coordinate_residuals(coordinates, p["parameters"])
        # No algebraic row contains the rate (clearing is a stock condition
        # on the states, so the DAE has index > 1); the rate path is
        # disciplined by the assets and Euler rows, with the pointwise slack
        # resolved by the minimum-norm selection. The t=0 clearing row is
        # identically zero because a_0 clears, and is kept for uniformity.
        return residual_scale * jnp.concatenate(
            [
                residuals["capital"].reshape(-1),
                residuals["assets"].reshape(-1),
                residuals["tobin_q"].reshape(-1),
                residuals["marginal_wealth"].reshape(-1),
                residuals["consumption_foc"].reshape(-1),
                residuals["asset_clearing"],
            ]
        )

    # The zero coordinate contains no saddle-path or present-value
    # information: it is flat in the derivative RKHS at the model's scale
    # normalizations.
    theta_0 = Theta(
        alpha_k=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_a=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_q=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_c=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_mu=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_rate=jnp.zeros(n_train, dtype=dtype),
        initial=InitialValues(
            q_offsets=jnp.zeros(num_countries, dtype=dtype),
            c_offsets=jnp.zeros(num_countries, dtype=dtype),
            mu_offsets=jnp.zeros(num_countries, dtype=dtype),
            rate_offset=jnp.zeros((), dtype=dtype),
        ),
    )

    if lm_set.ridge_init is None:
        initial_ridge = lm_set.ridge
        anneal = None
    else:
        initial_ridge = lm_set.ridge_init
        anneal = AnnealRidge(
            ridge_floor=lm_set.ridge,
            decrease=lm_set.ridge_decrease,
            grad_rtol=lm_set.ridge_grad_rtol,
        )
    if lm_set.linear_solver == "cg":
        # The kernel matrices ride in the solver's args so the rebuild
        # callback stays a value-hashable static; the initial preconditioner
        # only seeds the state, the callback replaces it every step.
        kernel_args = {
            "K": K,
            "K_tilde": K_tilde,
            "K_check": jsp_linalg.solve_triangular(metric.F.T, K_tilde.T, lower=True).T,
            "F_t": metric.F.T,
        }
        init_damping = lm_set.init_damping
        linear_solver = CG(
            build_arrow_woodbury(
                theta_0,
                kernel_args,
                p,
                initial_ridge,
                max(init_damping, lm_set.cg_damping_floor),
                num_countries,
            ),
            tol=lm_set.cg_tol,
            maxiter=lm_set.cg_maxiter,
        )
        callback = ArrowWoodburyCallback(
            num_countries=num_countries,
            damping_floor=lm_set.cg_damping_floor,
            anneal=anneal,
        )
    else:
        kernel_args = None
        init_damping = lm_set.init_damping
        linear_solver = {"auto": Cholesky(), "cholesky": Cholesky(), "qr": QR()}[
            lm_set.linear_solver
        ]
        callback = anneal
    if callback is None:
        continuation_options = {}
    else:
        continuation_options = {"callback": callback}
        if anneal is not None:
            continuation_options["user_state"] = anneal.init_state(dtype)
    solver = RidgeLevenbergMarquardt(
        residual_fn,
        metric=metric,
        ridge=initial_ridge,
        linear_solver=linear_solver,
        init_damping=init_damping,
        damping_decrease=lm_set.damping_decrease,
        damping_increase=lm_set.damping_increase,
    )
    solve_options = dict(
        max_steps=lm_set.max_steps,
        gtol=lm_set.gtol,
        atol=lm_set.residual_atol,
        max_steps_is_success=False,
    )

    timer = Stopwatch()
    result = solver.solve(
        theta_0,
        kernel_args,
        p=p,
        **solve_options,
        **continuation_options,
    )
    solve_time = timer.elapsed(result)
    theta = result.x

    def evaluate_coordinates(theta_value, K_eval, K_tilde_eval, p_value):
        coordinates = path_coordinates(theta_value, K_eval, K_tilde_eval, p_value)
        parameters_p = p_value["parameters"]
        residuals = coordinate_residuals(coordinates, parameters_p)
        x = investment_expenditure(coordinates["k"], coordinates["q"], parameters_p)
        output = production(coordinates["k"], parameters_p.productivities, parameters_p)
        resources = output - x
        populations_p = parameters_p.populations
        # Goods-market clearing is implied by the budgets and the clearing
        # row (Walras), so the per-capita goods gap is a reported
        # diagnostic, not a residual.
        resource_gap = jnp.sum(
            populations_p * (resources - coordinates["c"]), axis=-1
        ) / jnp.sum(populations_p)
        return {
            **coordinates,
            "x": x,
            "output": output,
            "resources": resources,
            "current_account": coordinates["da_dt"],
            "relative_welfare_weights": relative_welfare_weights(coordinates["mu"]),
            "resource_gap": resource_gap,
            **residuals,
        }

    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        values = evaluate_coordinates(theta, K_eval, K_tilde_eval, p)
        values["t"] = points
        return values

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    max_train_residual = max(
        float(jnp.max(jnp.abs(train_eval[name]))) for name in RESIDUAL_NAMES
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval[name]))) for name in RESIDUAL_NAMES
    )
    max_resource_gap = float(jnp.max(jnp.abs(validation_eval["resource_gap"])))
    max_asset_clearing = float(jnp.max(jnp.abs(validation_eval["asset_clearing"])))

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "mu", "x")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account", "interest_rate")
        )
    )
    rejection_reasons = []
    if int(result.status) != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    validation_pass = max_validation_residual <= tolerance_set.validation_residual

    resolved_steady = resolved_steady_state(validation_eval["mu"][-1], parameters)
    tail = tail_diagnostics(validation_eval, parameters)
    tvc = tvc_diagnostics(validation_data, validation_eval, parameters)
    long_run = long_run_diagnostics(
        validation_data,
        validation_eval,
        a_0_array,
        parameters,
        check_T=data_set.long_run_check_T,
        window=data_set.long_run_window,
    )
    relative_weights = test_eval["relative_welfare_weights"][0]
    log_relative_weights = jnp.log(relative_weights)
    coefficient_groups = {
        "k": theta.alpha_k,
        "a": theta.alpha_a,
        "q": theta.alpha_q,
        "c": theta.alpha_c,
        "m": theta.alpha_mu,
    }
    rkhs_norms = {
        f"{name}_{country + 1}": rkhs_norm_squared(coefficients[country], K)
        for name, coefficients in coefficient_groups.items()
        for country in range(num_countries)
    }
    rkhs_norm_rate = rkhs_norm_squared(theta.alpha_rate, K)
    rkhs_norm_total = sum(rkhs_norms.values()) + rkhs_norm_rate

    results = {
        "alpha": parameters.alpha,
        "delta": parameters.delta,
        "r": parameters.r,
        "eta": parameters.eta,
        "num_countries": num_countries,
        # Measured, not requested: a float64 leak upstream shows up here.
        "residual_dtype": str(train_eval["capital"].dtype),
        "kernel": kernel,
        "ell": kernel_set.ell,
        "sigma": kernel_set.sigma,
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "test_T": data_set.test_T,
        "test_points": data_set.test_points,
        "output_scale": float(output_scale),
        "consumption_scale": float(steady["consumption_scale"]),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "validation_pass": validation_pass,
        "max_resource_gap": max_resource_gap,
        "max_asset_clearing": max_asset_clearing,
        "terminal_residual_count": 0,
        "differential_residual_count": 4 * num_countries * n_train,
        "algebraic_residual_count": (num_countries + 1) * n_train,
        "dae_residual_count": residual_count,
        "unknown_count": residual_count + num_initial_values,
        "dynamic_rkhs_function_count": 4 * num_countries,
        "algebraic_rkhs_function_count": num_countries + 1,
        "free_initial_value_count": num_initial_values,
        "metric_selection_dimension": num_initial_values,
        "long_run_check_T": data_set.long_run_check_T,
        "long_run_window": data_set.long_run_window,
        "long_run_c_flatness": float(long_run["c_flatness"]),
        "long_run_k_flatness": float(long_run["k_flatness"]),
        "long_run_a_flatness": float(long_run["a_flatness"]),
        "long_run_k_theory_gap": float(long_run["k_theory_gap"]),
        "long_run_mpk_equalization_gap": float(long_run["mpk_equalization_gap"]),
        "long_run_interest_rate_gap": float(long_run["interest_rate_gap"]),
        "ex_post_k_log_gap": float(jnp.max(tail["k_log_gap"])),
        "ex_post_q_log_gap": float(jnp.max(tail["q_log_gap"])),
        "ex_post_asset_scaled_gap": float(jnp.max(tail["asset_scaled_gap"])),
        "ex_post_interest_rate_gap": float(tail["interest_rate_gap"]),
        "ex_post_discounted_assets": float(jnp.max(tvc["discounted_assets"])),
        "ex_post_asset_tvc": float(jnp.max(tvc["asset_tvc"])),
        "ex_post_capital_tvc": float(jnp.max(tvc["capital_tvc"])),
        "rkhs_norm_total": float(rkhs_norm_total),
        "rkhs_norm_interest_rate": float(rkhs_norm_rate),
        "solve_time": solve_time,
        "cg_tol": lm_set.cg_tol,
        "cg_maxiter": lm_set.cg_maxiter,
        "cg_damping_floor": lm_set.cg_damping_floor,
        "initial_seed": initial_set.seed,
        "k_0_log_sigma": initial_set.k_0_log_sigma,
        "a_0_sigma": initial_set.a_0_sigma,
        # Measured, not requested: under ridge continuation the implicit
        # dual is solved at the achieved ridge, which can sit above the
        # configured floor if the anneal stops early.
        "ridge": float(result.lm_state.ridge),
        "ridge_floor": lm_set.ridge,
        "gtol": lm_set.gtol,
        "residual_atol": lm_set.residual_atol,
        "train_loss": float(result.info.resid_loss),
        "final_grad_norm": float(result.info.grad_norm),
        "final_damping": float(result.lm_state.damping),
        "steps": int(result.steps),
        **acceptance_fields(rejection_reasons),
    }
    if long_run["common_gamma"]:
        results.update(
            {
                "long_run_consumption_ratio_drift": float(
                    long_run["consumption_ratio_drift"]
                ),
                "long_run_wealth_proportionality_gap": float(
                    long_run["wealth_proportionality_gap"]
                ),
                "long_run_wealth_rank_violation": float(
                    long_run["wealth_rank_violation"]
                ),
            }
        )
    for country in range(num_countries):
        suffix = country + 1
        results.update(
            {
                f"gamma_{suffix}": parameters.gammas[country],
                f"population_{suffix}": parameters.populations[country],
                f"productivity_{suffix}": parameters.productivities[country],
                f"k_0_{suffix}": k_0[country],
                f"a_0_{suffix}": a_0[country],
                f"a_0_{suffix}_over_output_ss": a_0[country] / float(output_scale),
                f"q_0_{suffix}": float(test_eval["q"][0, country]),
                f"c_0_{suffix}": float(test_eval["c"][0, country]),
                f"m_0_{suffix}": float(test_eval["mu"][0, country]),
                f"q_offset_{suffix}": float(theta.initial.q_offsets[country]),
                f"c_offset_{suffix}": float(theta.initial.c_offsets[country]),
                f"mu_offset_{suffix}": float(theta.initial.mu_offsets[country]),
                f"relative_weight_{suffix}": float(relative_weights[country]),
                f"log_relative_weight_{suffix}": float(log_relative_weights[country]),
                f"terminal_a_{suffix}": float(validation_eval["a"][-1, country]),
                f"long_run_a_{suffix}": float(resolved_steady["a"][country]),
                f"long_run_c_{suffix}": float(resolved_steady["c"][country]),
                **{
                    f"rkhs_norm_{name}_{suffix}": float(rkhs_norms[f"{name}_{suffix}"])
                    for name in coefficient_groups
                },
            }
        )
        if long_run["common_gamma"]:
            results[f"initial_wealth_{suffix}"] = float(
                long_run["initial_wealth"][country]
            )
    results["rate_offset"] = float(theta.initial.rate_offset)

    def frame(values, points):
        data = {
            "t": np.asarray(points),
            "interest_rate": np.asarray(values["interest_rate"]),
            "resource_gap": np.asarray(values["resource_gap"]),
            "asset_clearing": np.asarray(values["asset_clearing"]),
        }
        for country in range(num_countries):
            suffix = country + 1
            for name in (
                "k",
                "a",
                "q",
                "c",
                "mu",
                "x",
                "output",
                "resources",
                "current_account",
                "capital",
                "assets",
                "tobin_q",
                "marginal_wealth",
                "consumption_foc",
                "relative_welfare_weights",
            ):
                data[f"{name}_{suffix}"] = np.asarray(values[name][:, country])
        return pd.DataFrame(data)

    train_df = frame(train_eval, train_data)
    validation_df = frame(validation_eval, validation_data)
    test_df = frame(test_eval, test_data)

    if verbose:
        print(f"solve_time(s) = {solve_time}")

    hyperparameters = {
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "initial_set": asdict(initial_set),
        "lm_set": asdict(lm_set),
        "tolerance_set": asdict(tolerance_set),
        "parameters": asdict(parameters),
        "initial_conditions": {"k_0": list(k_0), "a_0": list(a_0)},
    }

    @jax.jit
    def kernel_solution(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        values = evaluate_coordinates(theta, K_eval, K_tilde_eval, p)
        values["t"] = points
        return values

    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "hyperparameters": hyperparameters,
        "solve": {
            "solver": solver,
            "residual_fn": residual_fn,
            "theta_0": theta_0,
            "p": p,
            "residual_args": kernel_args,
            "multi_start": None,
            "options": solve_options,
        },
        "result": result,
    }


if __name__ == "__main__":
    jsonargparse.CLI(multicountry_growth_kernel)
