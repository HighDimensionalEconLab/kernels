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
    capital_drift,
    current_account,
    draw_initial_conditions,
    investment_expenditure,
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
    STATUS_TO_REASON,
    CGSettings,
    KernelResidualArgs,
    OptimizerSettings,
    RidgeSettings,
    Stopwatch,
    acceptance_fields,
    make_sampled_train_grid,
)


# Alpha blocks first (the metric penalizes the leading flat coordinates);
# the (N, n_train) blocks ravel country-major. The trailing entries are the
# free level anchors of the jump variables and the world rate.
class Theta(NamedTuple):
    alpha_k: jax.Array
    alpha_a: jax.Array
    alpha_q: jax.Array
    alpha_c: jax.Array
    alpha_mu: jax.Array
    alpha_rate: jax.Array
    q_0s: jax.Array
    c_0s: jax.Array
    mu_0s: jax.Array
    rate_0: jax.Array


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
    # Population demeaning is a no-op at the primal point (the inputs
    # already clear) and projects AD perturbations of a_0 onto the
    # feasible clearing set, so every direction is meaningful.
    a_0_p = p.a_0 - jnp.sum(populations_p * p.a_0) / jnp.sum(populations_p)
    # Country-major (N, n_train) alpha blocks enter transposed so the
    # contraction K_tilde @ alpha.T yields (t_grid, N) paths; the (N,)
    # free anchors broadcast over rows.
    paths = integrated_kernel_paths(
        {
            "k": IntegratedPath(theta.alpha_k.T, p.k_0),
            "a": IntegratedPath(theta.alpha_a.T, a_0_p),
            "q": IntegratedPath(theta.alpha_q.T, theta.q_0s),
            "c": IntegratedPath(theta.alpha_c.T, theta.c_0s),
            "mu": IntegratedPath(theta.alpha_mu.T, theta.mu_0s),
            "rate": IntegratedPath(theta.alpha_rate, theta.rate_0),
        },
        K_eval,
        K_tilde_eval,
    )
    return {
        "k": paths["k"].value,
        "a": paths["a"].value,
        "q": paths["q"].value,
        "c": paths["c"].value,
        "mu": paths["mu"].value,
        "interest_rate": paths["rate"].value,
        "dk_dt": paths["k"].d_dt,
        "da_dt": paths["a"].d_dt,
        "dq_dt": paths["q"].d_dt,
        "dmu_dt": paths["mu"].d_dt,
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
    values = {
        "k": k,
        "a": a,
        "q": q,
        "c": c,
        "mu": mu,
        "interest_rate": coordinates["interest_rate"],
        "current_account": coordinates["da_dt"],
        "capital": coordinates["dk_dt"] - capital_drift(k, q, p),
        "assets": coordinates["da_dt"]
        - current_account(k, a, q, c, p.productivities, rate_column, p),
        "tobin_q": coordinates["dq_dt"]
        - q_drift(k, q, p.productivities, rate_column, p),
        "marginal_wealth": coordinates["dmu_dt"] - mu * (p.r - rate_column),
        "consumption_foc": mu * c**gammas - 1.0,
        # World bond clearing, per capita so the row size is N-free.
        "asset_clearing": jnp.sum(populations * a, axis=-1) / jnp.sum(populations),
    }
    # The t=0 clearing row is identically zero (a_0 clears); kept for
    # uniformity.
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
    # Conjunctive whitened stopping: gtol bounds the whitened stationarity,
    # atol the scaled residual norm (its floor is O(ridge * ||dual||)).
    # atol must sit BELOW the intermediate anneal levels' residuals or the
    # continuation stops above the ridge floor and the clearing-row floor
    # misses the 1e-5 train gate.
    # init_damping well above machine level: the level-form rows are
    # nonlinear enough at the seeds that an undamped first step is rejected
    # repeatedly before the damping recovers.
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-4,
        damping_decrease=0.1,
        damping_increase=4.0,
        max_steps=100,
        atol=1e-7,
        gtol=5e-8,
    ),
    # "cg" is matrix-free preconditioned CG (arrow+Woodbury, rebuilt every LM
    # step); implicit-AD re-solves stay dense-only (CG degrades the tangents).
    ridge_set: RidgeSettings = RidgeSettings(
        ridge=3e-13,
        ridge_init=1e-7,
        ridge_decrease=1e-3,
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

    num_rkhs_functions = 5 * num_countries + 1
    num_initial_values = 3 * num_countries + 1
    K, K_tilde, metric = integrated_metric_factorization(
        train_data, kernel, kernel_set.sigma, kernel_set.ell, repeats=num_rkhs_functions
    )

    # Demeaning clears the sub-tolerance clearing residue the validation
    # admits; the demeaned values are what the paths anchor to, so they are
    # also what the results and hyperparameters report.
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

    # Seeds zero the constant-path residual rows: at q_0 = 1/eta the
    # chi-normalized investment rate equals delta, rate_0 = r zeroes the
    # marginal-wealth rows, and c = domestic resources + bond income zeroes
    # every seed current account (a_0 is demeaned, so the clearing row is
    # zero too). Only the tobin_q rows are nonzero at the seed.
    c_seed = production(p.k_0, p.productivities, p) - p.delta * p.k_0 + p.r * p.a_0
    theta_0 = Theta(
        alpha_k=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_a=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_q=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_c=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_mu=jnp.zeros((num_countries, n_train), dtype=dtype),
        alpha_rate=jnp.zeros(n_train, dtype=dtype),
        q_0s=jnp.full(num_countries, 1.0 / p.eta, dtype=dtype),
        c_0s=c_seed,
        mu_0s=c_seed**-p.gammas,
        rate_0=jnp.asarray(p.r, dtype=dtype),
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
        float(jnp.max(jnp.abs(train_eval["capital"]))),
        float(jnp.max(jnp.abs(train_eval["assets"]))),
        float(jnp.max(jnp.abs(train_eval["tobin_q"]))),
        float(jnp.max(jnp.abs(train_eval["marginal_wealth"]))),
        float(jnp.max(jnp.abs(train_eval["consumption_foc"]))),
        float(jnp.max(jnp.abs(train_eval["asset_clearing"]))),
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(validation_eval["capital"]))),
        float(jnp.max(jnp.abs(validation_eval["assets"]))),
        float(jnp.max(jnp.abs(validation_eval["tobin_q"]))),
        float(jnp.max(jnp.abs(validation_eval["marginal_wealth"]))),
        float(jnp.max(jnp.abs(validation_eval["consumption_foc"]))),
        float(jnp.max(jnp.abs(validation_eval["asset_clearing"]))),
    )
    max_resource_gap = float(jnp.max(jnp.abs(validation_eval["resource_gap"])))
    max_asset_clearing = float(jnp.max(jnp.abs(validation_eval["asset_clearing"])))

    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(values[name])) and jnp.min(values[name]) > 0.0
            for values in (train_eval, validation_eval, test_eval)
            for name in ("k", "q", "c", "mu", "x", "interest_rate")
        )
        and all(
            jnp.all(jnp.isfinite(values[name]))
            for values in (train_eval, validation_eval, test_eval)
            for name in ("a", "current_account")
        )
    )
    residual_count = (5 * num_countries + 1) * n_train
    rejection_reasons = []
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > tolerance_set.train_residual:
        rejection_reasons.append("train_residual")
    validation_pass = max_validation_residual <= tolerance_set.validation_residual
    if not validation_pass:
        rejection_reasons.append("validation_residual")

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
        "lm_set": {
            "init_damping": lm_set.init_damping,
            "damping_decrease": lm_set.damping_decrease,
            "damping_increase": lm_set.damping_increase,
            "max_steps": lm_set.max_steps,
            "atol": lm_set.atol,
            "gtol": lm_set.gtol,
        },
        "ridge_set": asdict(ridge_set),
        "cg_set": asdict(cg_set),
        "tolerance_set": asdict(tolerance_set),
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
        "solver_status": STATUS_TO_REASON.get(int(result.status), "unknown"),
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
                f"q_0_{suffix}": float(theta.q_0s[country]),
                f"c_0_{suffix}": float(theta.c_0s[country]),
                f"mu_0_{suffix}": float(theta.mu_0s[country]),
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
    results["rate_0"] = float(theta.rate_0)

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
