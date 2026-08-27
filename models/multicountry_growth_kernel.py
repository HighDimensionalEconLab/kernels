from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jsonargparse
import numpy as np
import pandas as pd
from flax import struct
from nlls_gram import (
    CG,
    QR,
    AnnealRidge,
    Cholesky,
    LMStatus,
    RidgeLevenbergMarquardt,
)

from models.kernels import (
    IntegratedPath,
    KernelSettings,
    integrated_kernel_matrices,
    integrated_kernel_paths,
    integrated_metric_factorization,
    rkhs_norm_squared,
)
from models.multicountry_growth_preconditioner import (
    ArrowWoodburyCallback,
    build_arrow_woodbury,
)
from models.multicountry_growth_shared import (
    MULTICOUNTRY_GROWTH_CONSTANTS as CONSTANTS,
)
from models.multicountry_growth_shared import (
    DataSettings,
    InitialConditionSettings,
    StructuralParameters,
    current_account,
    draw_initial_conditions,
    investment_expenditure,
    investment_rate_from_q,
    long_run_diagnostics,
    production,
    q_drift,
    relative_welfare_weights,
    resolved_steady_state,
    steady_state,
    tail_diagnostics,
    tvc_diagnostics,
    validate_initial_conditions,
)
from models.utilities import (
    CGSettings,
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
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


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# the (N, n_train) blocks ravel country-major. The trailing entries are the
# free log-anchors of the jump variables and the world rate.
class Theta(NamedTuple):
    alpha_k: jax.Array
    alpha_a: jax.Array
    alpha_q: jax.Array
    alpha_c: jax.Array
    alpha_mu: jax.Array
    alpha_rate: jax.Array
    log_q_0s: jax.Array
    log_c_0s: jax.Array
    log_mu_0s: jax.Array
    log_rate_0: jax.Array


# CG-path args: whitened factors for the preconditioner rebuild.
@struct.dataclass
class PreconditionedKernelResidualArgs:
    K: jax.Array
    K_tilde: jax.Array
    K_check: jax.Array
    F_t: jax.Array


# Module-level so callers can evaluate solved paths on their own kernel
# matrices.
def path_coordinates(theta, K_eval, K_tilde_eval, p):
    populations_p = p.populations
    k_0_p = p.k_0
    # Population demeaning is a no-op at the primal point (the inputs
    # already clear) and projects AD perturbations of a_0 onto the
    # feasible clearing set, so every direction is meaningful.
    a_0_p = p.a_0 - jnp.sum(populations_p * p.a_0) / jnp.sum(populations_p)
    # The asset block is the one path with no free scalar anchor, so its
    # coefficient scale enters the minimum-norm selection metric; the
    # scale must be differentiation-inert or every parameter sensitivity
    # picks up a selection-drift term from the metric moving with p.
    # Population-weighted time-0 output sets it from the initial
    # conditions alone.
    output_scale_p = jax.lax.stop_gradient(
        jnp.sum(populations_p * production(k_0_p, p.productivities, p))
        / jnp.sum(populations_p)
    )
    # Country-major (N, n_train) alpha blocks enter transposed so the
    # contraction K_tilde @ alpha.T yields (t_grid, N) paths; the (N,)
    # free anchors broadcast over rows.
    paths = integrated_kernel_paths(
        {
            "log_k": IntegratedPath(theta.alpha_k.T, jnp.log(k_0_p)),
            "a_raw": IntegratedPath(
                theta.alpha_a.T, jnp.zeros((), theta.alpha_a.dtype)
            ),
            "log_q": IntegratedPath(theta.alpha_q.T, theta.log_q_0s),
            "log_c": IntegratedPath(theta.alpha_c.T, theta.log_c_0s),
            "log_mu": IntegratedPath(theta.alpha_mu.T, theta.log_mu_0s),
            "log_rate": IntegratedPath(theta.alpha_rate, theta.log_rate_0),
        },
        K_eval,
        K_tilde_eval,
    )
    return {
        "k": jnp.exp(paths["log_k"].value),
        "a": a_0_p + output_scale_p * paths["a_raw"].value,
        "q": jnp.exp(paths["log_q"].value),
        "c": jnp.exp(paths["log_c"].value),
        "mu": jnp.exp(paths["log_mu"].value),
        "interest_rate": jnp.exp(paths["log_rate"].value),
        "dlog_k_dt": paths["log_k"].d_dt,
        "da_dt": output_scale_p * paths["a_raw"].d_dt,
        "dlog_q_dt": paths["log_q"].d_dt,
        "dlog_mu_dt": paths["log_mu"].d_dt,
    }


# p carries the structural parameters and initial conditions for implicit
# differentiation; args carries the kernel matrices (the CG path adds the
# whitened factors its preconditioner rebuild needs).
def residual_fn(theta, args, p):
    coordinates = path_coordinates(theta, args.K, args.K_tilde, p)
    k, a, q = coordinates["k"], coordinates["a"], coordinates["q"]
    c, mu = coordinates["c"], coordinates["mu"]
    rate_column = coordinates["interest_rate"][..., None]
    gammas = jnp.asarray(p.gammas, dtype=c.dtype)
    populations = jnp.asarray(p.populations, dtype=a.dtype)
    chi = p.delta ** (1.0 - p.eta)
    values = {
        "k": k,
        "a": a,
        "q": q,
        "c": c,
        "mu": mu,
        "interest_rate": coordinates["interest_rate"],
        "current_account": coordinates["da_dt"],
        "capital": coordinates["dlog_k_dt"]
        - (chi * investment_rate_from_q(q, p) ** p.eta - p.delta),
        "assets": coordinates["da_dt"]
        - current_account(k, a, q, c, p.productivities, rate_column, p),
        "tobin_q": coordinates["dlog_q_dt"]
        - q_drift(k, q, p.productivities, rate_column, p) / q,
        "marginal_wealth": coordinates["dlog_mu_dt"] - (p.r - rate_column),
        "consumption_foc": mu * c**gammas - 1.0,
        # World bond clearing, per capita so the row size is N-free.
        "asset_clearing": jnp.sum(populations * a, axis=-1) / jnp.sum(populations),
    }
    # No algebraic row contains the rate (clearing is a stock condition
    # on the states, so the DAE has index > 1); the rate path is
    # disciplined by the assets and Euler rows, with the pointwise slack
    # resolved by the minimum-norm selection. The t=0 clearing row is
    # identically zero because a_0 clears, and is kept for uniformity.
    residual = jnp.concatenate(
        [
            values["capital"].reshape(-1),
            values["assets"].reshape(-1),
            values["tobin_q"].reshape(-1),
            values["marginal_wealth"].reshape(-1),
            values["consumption_foc"].reshape(-1),
            values["asset_clearing"],
        ]
    )
    return residual / jnp.sqrt(residual.size), values


# Numerical acceptance and ex-post quality thresholds, CLI-reachable so a
# lower-precision caller can pass its own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-5
    # The validation threshold is an HPO/quality diagnostic; it does not
    # redefine convergence of a finite, feasible numerical solution.
    validation_residual: float = 1e-2


def multicountry_growth_kernel(
    alpha: float = CONSTANTS["alpha"],
    gammas: tuple[float, ...] = CONSTANTS["gammas"],
    delta: float = CONSTANTS["delta"],
    r: float = CONSTANTS["r"],
    eta: float = CONSTANTS["eta"],
    populations: tuple[float, ...] = CONSTANTS["populations"],
    productivities: tuple[float, ...] = CONSTANTS["productivities"],
    k_0: tuple[float, ...] | None = None,
    a_0: tuple[float, ...] | None = None,
    use_float64: bool = True,
    initial_set: InitialConditionSettings = InitialConditionSettings(),
    kernel_set: KernelSettings = KernelSettings(ell=20.0),
    data_set: DataSettings = DataSettings(),
    # Conjunctive stopping in the whitened variable y = F_bar theta the
    # solver works in: gtol bounds the whitened stationarity, atol
    # additionally demands the scaled residual norm solved (the converged
    # residual floors at O(ridge * ||dual||), so atol sits above it).
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-8,
        damping_decrease=0.1,
        damping_increase=4.0,
        max_steps=100,
        atol=5e-7,
        gtol=5e-8,
    ),
    # The ridge keeps the minimizer O(ridge) from the pure RKHS-seminorm
    # interpolant with the free anchors unpenalized; ridge_init enables
    # continuation annealed down to the ridge floor by factors of
    # ridge_decrease on inner stationarity. linear_solver "auto" (whitened
    # cholesky) and "qr" solve dense whitened systems; "cg" is matrix-free
    # preconditioned CG on the whitened normal operator with the
    # arrow+Woodbury preconditioner rebuilt every LM step (derivation in
    # models/multicountry_growth_preconditioner.py). Implicit-AD re-solves
    # stay dense-only: the undamped system sits at the float64 conditioning
    # plateau, where CG would silently degrade the tangents.
    ridge_set: RidgeSettings = RidgeSettings(
        ridge=3e-12,
        ridge_init=1e-7,
        ridge_decrease=0.01,
        ridge_grad_rtol=1e-2,
    ),
    # cg tol is unreachable in the endgame, so maxiter is the effective
    # budget there; damping_floor keeps the preconditioner's block Cholesky
    # positive definite while the live LM damping anneals freely.
    cg_set: CGSettings = CGSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    verbose: bool = False,
):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.float64 if use_float64 else jnp.float32
    num_countries = len(populations)
    # Identical initial conditions are degenerate (no cross-country
    # variation), so unspecified k_0/a_0 are DRAWN from the seeded
    # initial_set distributions rather than defaulted to symmetric values.
    if k_0 is None or a_0 is None:
        drawn_k_0, drawn_a_0 = draw_initial_conditions(initial_set, populations)
        if k_0 is None:
            k_0 = drawn_k_0
        if a_0 is None:
            a_0 = drawn_a_0
    validate_initial_conditions(
        StructuralParameters.create(
            dtype=jnp.float64,
            alpha=alpha,
            gammas=gammas,
            delta=delta,
            r=r,
            eta=eta,
            populations=populations,
            productivities=productivities,
            k_0=k_0,
            a_0=a_0,
        )
    )

    kernel = kernel_set.kernel
    train_data = make_sampled_train_grid(data_set, dtype)
    test_data = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    validation_data = jnp.linspace(
        0.0, float(train_data[-1]), data_set.validation_points, dtype=dtype
    )
    n_train = len(train_data)
    populations_array = jnp.asarray(populations, dtype=dtype)
    total_population = jnp.sum(populations_array)

    # There are (5N+1)n DAE equations in (5N+1)n + 3N + 1 unknowns. No
    # terminal equation is appended: the ridge penalty supplies the
    # (3N+1)-dimensional selection. Every linear solver works on the
    # whitened subproblem; the metric takes the upper Cholesky factor F with
    # K = F'F and the trailing anchors stay in the unpenalized free block,
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
    a_0_array = a_0_array - jnp.sum(populations_array * a_0_array) / total_population
    a_0 = tuple(float(value) for value in a_0_array)
    p = StructuralParameters.create(
        dtype=dtype,
        alpha=alpha,
        gammas=gammas,
        delta=delta,
        r=r,
        eta=eta,
        populations=populations,
        productivities=productivities,
        k_0=k_0,
        a_0=a_0_array,
    )
    steady = steady_state(p, dtype)
    output_scale = steady["output_scale"]

    # Flat alpha blocks carry no saddle-path or present-value information;
    # the free anchors are seeded from the initial conditions alone:
    # per-country consumption at the flow sustainable at k_0_i (mu
    # consistent via the FOC), q at the replacement-investment price
    # 1/eta, the world rate at r.
    log_c_seed = jnp.log(production(p.k_0, p.productivities, p) - p.delta * p.k_0)
    theta_0 = Theta(
        alpha_k=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_a=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_q=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_c=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_mu=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_rate=jnp.zeros(n_train, dtype=dtype),
        log_q_0s=jnp.full(num_countries, -jnp.log(p.eta), dtype=dtype),
        log_c_0s=log_c_seed,
        log_mu_0s=-p.gammas * log_c_seed,
        log_rate_0=jnp.log(p.r),
    )

    if ridge_set.ridge_init is None:
        initial_ridge = ridge_set.ridge
        anneal = None
    else:
        initial_ridge = ridge_set.ridge_init
        anneal = AnnealRidge(
            ridge_floor=ridge_set.ridge,
            decrease=ridge_set.ridge_decrease,
            grad_rtol=ridge_set.ridge_grad_rtol,
        )
    init_damping = lm_set.init_damping
    if ridge_set.linear_solver == "cg":
        # The kernel matrices ride in the solver's args so the rebuild
        # callback stays a value-hashable static; the initial preconditioner
        # only seeds the state, the callback replaces it every step.
        kernel_args = PreconditionedKernelResidualArgs(
            K=K,
            K_tilde=K_tilde,
            K_check=jsp_linalg.solve_triangular(metric.F.T, K_tilde.T, lower=True).T,
            F_t=metric.F.T,
        )
        linear_solver = CG(
            build_arrow_woodbury(
                theta_0,
                kernel_args,
                p,
                initial_ridge,
                max(init_damping, cg_set.damping_floor),
                num_countries,
            ),
            tol=cg_set.tol,
            maxiter=cg_set.maxiter,
        )
        callback = ArrowWoodburyCallback(
            num_countries=num_countries,
            damping_floor=cg_set.damping_floor,
            anneal=anneal,
        )
    else:
        kernel_args = KernelResidualArgs(K=K, K_tilde=K_tilde)
        linear_solver = {"auto": Cholesky(), "cholesky": Cholesky(), "qr": QR()}[
            ridge_set.linear_solver
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
        has_aux=True,
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
        atol=lm_set.atol,
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

    # Paths, model quantities, and DAE residuals on an evaluation grid
    # through fresh cross-kernel matrices against the training grid.
    def evaluate_grid(points):
        K_eval, K_tilde_eval = integrated_kernel_matrices(
            points,
            train_data,
            kernel,
            kernel_set.sigma,
            kernel_set.ell,
        )
        values = residual_fn(
            theta, KernelResidualArgs(K=K_eval, K_tilde=K_tilde_eval), p
        )[1]
        x = investment_expenditure(values["k"], values["q"], p)
        output = production(values["k"], p.productivities, p)
        resources = output - x
        # Goods-market clearing is implied by the budgets and the clearing
        # row (Walras), so the per-capita goods gap is a reported
        # diagnostic, not a residual.
        resource_gap = jnp.sum(
            p.populations * (resources - values["c"]), axis=-1
        ) / jnp.sum(p.populations)
        return {
            "t": points,
            **values,
            "x": x,
            "output": output,
            "resources": resources,
            "relative_welfare_weights": relative_welfare_weights(values["mu"]),
            "resource_gap": resource_gap,
        }

    kernel_solution = jax.jit(evaluate_grid)

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
    residual_count = (5 * num_countries + 1) * n_train
    rejection_reasons = []
    if int(result.status) != LMStatus.CONVERGED:
        rejection_reasons.append("solver_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    validation_pass = max_validation_residual <= tolerance_set.validation_residual

    resolved_steady = resolved_steady_state(validation_eval["mu"][-1], p)
    tail = tail_diagnostics(validation_eval, p)
    tvc = tvc_diagnostics(validation_data, validation_eval, p)
    long_run = long_run_diagnostics(
        validation_data,
        validation_eval,
        a_0_array,
        p,
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
        "parameters": p.to_dict(),
        "kernel_set": asdict(kernel_set),
        "data_set": asdict(data_set),
        "initial_set": asdict(initial_set),
        "lm_set": asdict(lm_set),
        "ridge_set": asdict(ridge_set),
        "cg_set": asdict(cg_set),
        "tolerance_set": asdict(tolerance_set),
        "train_T": float(train_data[-1]),
        "num_train_points": n_train,
        "output_scale": float(output_scale),
        "consumption_scale": float(steady["consumption_scale"]),
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "validation_pass": validation_pass,
        "max_resource_gap": max_resource_gap,
        "max_asset_clearing": max_asset_clearing,
        "differential_residual_count": 4 * num_countries * n_train,
        "algebraic_residual_count": (num_countries + 1) * n_train,
        "dae_residual_count": residual_count,
        "unknown_count": residual_count + num_initial_values,
        "dynamic_rkhs_function_count": 4 * num_countries,
        "algebraic_rkhs_function_count": num_countries + 1,
        "free_initial_value_count": num_initial_values,
        "metric_selection_dimension": num_initial_values,
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
        # Measured, not requested: under ridge continuation the implicit
        # dual is solved at the achieved ridge, which can sit above the
        # configured floor if the anneal stops early.
        "ridge": float(result.lm_state.ridge),
        "train_loss": float(result.info.resid_loss),
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
                f"a_0_{suffix}_over_output_ss": a_0[country] / float(output_scale),
                f"q_0_{suffix}": float(test_eval["q"][0, country]),
                f"c_0_{suffix}": float(test_eval["c"][0, country]),
                f"m_0_{suffix}": float(test_eval["mu"][0, country]),
                f"log_q_0_{suffix}": float(theta.log_q_0s[country]),
                f"log_c_0_{suffix}": float(theta.log_c_0s[country]),
                f"log_mu_0_{suffix}": float(theta.log_mu_0s[country]),
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
    results["log_rate_0"] = float(theta.log_rate_0)

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

    return results, {
        "model": kernel_solution,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
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
