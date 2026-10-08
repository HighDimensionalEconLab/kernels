from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import pandas as pd
from tinydiffeq import PIController, SaveAt, Tsit5, solve_ode

from models.neoclassical_growth_benchmark import (
    BenchmarkParameters,
    neoclassical_growth_benchmark,
)
from models.neoclassical_growth_shared import (
    NEOCLASSICAL_GROWTH_CONSTANTS as C,
)
from models.neoclassical_growth_shared import (
    StructuralParameters,
    mpk,
    production,
    steady_state,
)
from models.utilities import Stopwatch, acceptance_fields


@dataclass(frozen=True)
class ShootingSettings:
    # "newton": undamped Newton steps on x(T) - x_ss with the derivative by
    # autodiff through the integrator. "bisection": bracket expansion from the
    # guess, then bisection on the sign of x(T) - x_ss, a path that leaves the
    # positive orthant before T counting as a guess that is too high.
    method: str = "newton"
    # Initial guess c_0 drawn uniformly in c_0^* (1 +- radius).
    radius: float = 0.1
    T: float = 100.0
    # Adaptive Tsit5 initial value solver, tight enough that integration error
    # sits below the roundoff in c_0 that the horizon amplifies.
    rtol: float = 1e-12
    atol: float = 1e-14
    dt_0: float = 0.1
    # Must exceed the steps of any successful solve (about 1100 at T = 300);
    # a path that runs into k = 0 spends the whole budget before failing.
    max_steps: int = 10000
    # Newton steps, or integrations for the bracket search plus bisection.
    max_iter: int = 50
    # Bisection: the first expansion step is initial_step_fraction * radius *
    # guess, later steps grow geometrically.
    initial_step_fraction: float = 0.25
    expansion_factor: float = 1.5
    # Convergence for both methods: the returned c_0's path reaches T with
    # |x(T) - x_ss| / x_ss below this.
    terminal_tol: float = 1e-6


# Errors are measured on the sequential models' common test window, shared
# with the kernel and NN comparisons, and on the shooting horizon [0, T].
@dataclass(frozen=True)
class ShootingDataSettings:
    test_T: float = C["sequence_test_T"]
    test_points: int = 100


# The boundary-value benchmark is solved out to the longest shooting horizon.
BENCHMARK_T = 300.0


def benchmark_paths(p):
    return neoclassical_growth_benchmark(p, BenchmarkParameters(T=BENCHMARK_T))


def baseline_parameters(
    k_0,
    a=C["a"],
    delta=C["delta"],
    rho_hat=C["rho_hat"],
    sigma_crra=C["sigma_crra"],
):
    return StructuralParameters.create(
        dtype=jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        g=0.0,
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        k_thr=jnp.inf,
        z_0=1.0,
        k_0=k_0,
    )


# Reduced (k, c) system with the co-state eliminated through mu = c^{-gamma}.
def field(state, p):
    k, c = state
    dk = production(1.0, k, p) - p.delta * k - c
    dc = c / p.sigma_crra * (mpk(1.0, k, p) - p.rho_hat - p.delta)
    return jnp.array([dk, dc])


# Forward integration with the adaptive solver. `ok` is false when the
# integration fails before T, which for this field happens when the path runs
# into the boundary k = 0; the saved states are then NaN throughout. The
# dynamic loop stops at the step budget, so a failed path costs a few
# milliseconds; it supports forward-mode AD only, which is all the scalar
# derivatives below need.
@partial(jax.jit, static_argnames=("s",))
def integrate(state_0, T, s, p, ts=None):
    def f(x, t, args, p):
        return field(x, p)

    solution = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        state_0,
        p=p,
        dt_0=s.dt_0,
        controller=PIController(rtol=s.rtol, atol=s.atol),
        max_steps=s.max_steps,
        save_at=SaveAt(t_1=True) if ts is None else SaveAt(ts=ts),
        adaptive_loop="forward",
    )
    return jnp.where(solution.ok, solution.xs, jnp.nan), solution.ok


# Terminal capital and its derivative with respect to the guess by forward
# mode through the integrator, NaN when the path leaves the positive orthant
# before T.
def terminal_capital(c_0, k_0, T, s, p):
    def k_T(c_0):
        state, _ = integrate(jnp.array([k_0, c_0]), T, s, p)
        return state[0]

    return jax.jvp(k_T, (c_0,), (jnp.ones_like(c_0),))


# Newton shooting: iterate c <- c - (x(T; c) - x_ss) / x'(T; c) from the guess
# until the relative terminal miss is below the tolerance, the budget is
# spent, or an iterate's path blows up (NaN miss or derivative, or c <= 0).
@partial(jax.jit, static_argnames=("max_iter", "s"))
def newton_shooting(c_init, k_0, k_ss, T, max_iter, s, p):
    def cond(carry):
        _, k_T, _, iters, blew_up = carry
        return (
            (jnp.abs(k_T - k_ss) > s.terminal_tol * k_ss)
            & (iters < max_iter)
            & ~blew_up
        )

    def body(carry):
        c, k_T, dk_T, iters, _ = carry
        c_next = c - (k_T - k_ss) / dk_T
        k_next, dk_next = terminal_capital(c_next, k_0, T, s, p)
        blew_up = (
            ~jnp.isfinite(dk_T)
            | ~jnp.isfinite(c_next)
            | (c_next <= 0.0)
            | ~jnp.isfinite(k_next)
        )
        return (c_next, k_next, dk_next, iters + 1, blew_up)

    k_init, dk_init = terminal_capital(c_init, k_0, T, s, p)
    c, k_T, _, n_iterations, blew_up = jax.lax.while_loop(
        cond, body, (c_init, k_init, dk_init, 0, ~jnp.isfinite(k_init))
    )
    miss = k_T - k_ss
    converged = ~blew_up & (jnp.abs(miss) <= s.terminal_tol * k_ss)
    return c, converged, blew_up, n_iterations, miss / k_ss


# Shooting error sign for bisection: +1 when c_0 is too low (capital
# overshoots the steady state at T), -1 when too high. A path whose
# integration fails before T has consumed the capital stock, so it counts as
# too high; a non-positive guess is integrated at a clamped value and its
# sign overridden to too low.
def terminal_sign(c_0, k_0, k_ss, T, s, p):
    state, ok = integrate(jnp.array([k_0, jnp.maximum(c_0, 1e-12)]), T, s, p)
    too_low = ok & jnp.all(jnp.isfinite(state)) & (state[0] > k_ss)
    return jnp.where((c_0 <= 0.0) | too_low, 1.0, -1.0)


# Bracket expansion from the guess, then bisection, under one integration
# budget that includes the evaluation of the guess itself and the two final
# evaluations that pick the end of the bracket with the smaller terminal miss
# (the midpoint of adjacent floats rounds to one of them). The bracket grows
# geometrically in the direction that corrects the terminal sign until the
# sign flips.
@partial(jax.jit, static_argnames=("max_iter", "s"))
def bisection_shooting(c_init, k_0, k_ss, T, max_iter, s, p):
    sign = partial(terminal_sign, k_0=k_0, k_ss=k_ss, T=T, s=s, p=p)
    s_init = sign(c_init)

    def expand_cond(carry):
        _, _, _, _, iters, found = carry
        return (~found) & (iters < max_iter)

    def expand_body(carry):
        _, c_curr, step, s_curr, iters, _ = carry
        c_next = c_curr + s_init * step
        s_next = sign(c_next)
        return (
            c_curr,
            c_next,
            step * s.expansion_factor,
            s_next,
            iters + 1,
            s_next != s_curr,
        )

    c_prev, c_last, _, _, n_expansions, found = jax.lax.while_loop(
        expand_cond,
        expand_body,
        (
            c_init,
            c_init,
            s.initial_step_fraction * s.radius * c_init,
            s_init,
            1,
            jnp.asarray(False),
        ),
    )
    lo = jnp.minimum(c_prev, c_last)
    hi = jnp.maximum(c_prev, c_last)

    def bisect_cond(carry):
        _, _, iters = carry
        return iters < max_iter - 2

    def bisect_body(carry):
        lo, hi, iters = carry
        mid = 0.5 * (lo + hi)
        too_low = sign(mid) > 0.0
        return (jnp.where(too_low, mid, lo), jnp.where(too_low, hi, mid), iters + 1)

    lo, hi, n_total = jax.lax.while_loop(
        bisect_cond, bisect_body, (lo, hi, n_expansions)
    )

    def terminal_miss(c):
        state, ok = integrate(jnp.array([k_0, c]), T, s, p)
        return jnp.where(ok, jnp.abs(state[0] - k_ss), jnp.inf)

    c_0 = jnp.where(terminal_miss(lo) <= terminal_miss(hi), lo, hi)
    return jnp.where(found, c_0, jnp.nan), found, n_total + 2


# Basin of convergence of Newton by horizon: on each side of c_0^*, the
# largest relative offset from which Newton converges, found on a coarse log
# grid and then by bisection in log offset between the last converging and
# the first failing grid point (the converging set is contiguous).
def newton_basins(
    horizons, k_0=1.0, shooting_set=ShootingSettings(), offsets=None, refinements=12
):
    if offsets is None:
        offsets = np.logspace(-12, -0.25, 48)
    p = baseline_parameters(k_0)
    k_ss, _ = steady_state(p)
    c_0_true = float(benchmark_paths(p)(jnp.array([0.0]))[1][0])

    def converges(offset, direction, T):
        _, converged, _, _, _ = newton_shooting(
            jnp.asarray(c_0_true * (1.0 + direction * offset)),
            p.k_0,
            k_ss,
            T,
            shooting_set.max_iter,
            shooting_set,
            p,
        )
        return bool(converged)

    rows = []
    for T in horizons:
        basin = {}
        for side, direction in (("below", -1.0), ("above", 1.0)):
            inside = [o for o in offsets if converges(o, direction, T)]
            if not inside:
                basin[side] = 0.0
                continue
            lo = max(inside)
            outside = [o for o in offsets if o > lo]
            if not outside:
                basin[side] = lo
                continue
            hi = min(outside)
            for _ in range(refinements):
                mid = float(np.sqrt(lo * hi))
                if converges(mid, direction, T):
                    lo = mid
                else:
                    hi = mid
            basin[side] = lo
        rows.append(
            {"T": T, "basin_below": basin["below"], "basin_above": basin["above"]}
        )
    return pd.DataFrame(rows), {
        "basin_offsets": offsets.tolist(),
        "basin_refinements": refinements,
    }


# k(T), dk(T)/dc_0, and the elasticity d log k(T) / d log c_0 by forward mode
# through the integrator. Non-finite entries mean the path blew up before T.
@partial(jax.jit, static_argnames=("s",))
def terminal_elasticity(c_0, k_0, T, s, p):
    k_T, dk_T = terminal_capital(c_0, k_0, T, s, p)
    return k_T, dk_T, dk_T * c_0 / k_T


def shooting_gradients(c_offsets, horizons, k_0=1.0, shooting_set=ShootingSettings()):
    p = baseline_parameters(k_0)
    k_ss, c_ss = steady_state(p)
    c_0_true = float(benchmark_paths(p)(jnp.array([0.0]))[1][0])
    # Eigenvalues of the Jacobian of the (k, c) field at the steady state: one
    # unstable lambda_u > 0 and one stable lambda_s < 0, with lambda_u + lambda_s
    # equal to the trace f'(k_ss) - delta = rho_hat.
    eigenvalues = jnp.linalg.eigvals(
        jax.jacfwd(lambda state: field(state, p))(jnp.array([k_ss, c_ss]))
    )
    rows = []
    for T in horizons:
        for offset in c_offsets:
            c_0 = c_0_true * (1.0 + offset)
            k_T, dk_T, elasticity = terminal_elasticity(
                jnp.asarray(c_0), p.k_0, T, shooting_set, p
            )
            rows.append(
                {
                    "T": T,
                    "c_0_offset": offset,
                    "c_0": c_0,
                    "k_T": float(k_T),
                    "d_k_T_d_c_0": float(dk_T),
                    "elasticity_k_T_c_0": float(elasticity),
                }
            )
    return pd.DataFrame(rows), {
        "k_ss": k_ss,
        "c_0_true": c_0_true,
        "unstable_eigenvalue": float(jnp.max(jnp.real(eigenvalues))),
        "stable_eigenvalue": float(jnp.min(jnp.real(eigenvalues))),
        "discount_rate": float(p.rho_hat),
    }


# Derivative seminorm int_0^T (xdot^2 + mudot^2) dt along a path, with
# mu = c^{-gamma}, accumulated as a third state; the quantity the minimum-norm
# theorem is about.
def energy_field(state, p):
    k, c, _ = state
    dk, dc = field(state[:2], p)
    dmu = -p.sigma_crra * c ** (-p.sigma_crra - 1.0) * dc
    return jnp.array([dk, dc, dk**2 + dmu**2])


@partial(jax.jit, static_argnames=("s",))
def integrate_energy(state_0, T, s, p):
    def f(x, t, args, p):
        return energy_field(x, p)

    solution = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        state_0,
        p=p,
        dt_0=s.dt_0,
        controller=PIController(rtol=s.rtol, atol=s.atol),
        max_steps=s.max_steps,
        save_at=SaveAt(t_1=True),
    )
    return jnp.where(solution.ok, solution.xs[2], jnp.nan), solution.ok


# Seminorms by horizon: the saddle path by quadrature along the boundary-value
# benchmark (which satisfies the ODE, so the field gives its derivatives),
# and forward paths from c_0 at relative offsets at or below c_0^*, which
# stay in the positive orthant.
def shooting_seminorms(
    c_offsets,
    horizons,
    k_0=1.0,
    shooting_set=ShootingSettings(),
    quadrature_points=20001,
):
    p = baseline_parameters(k_0)
    benchmark = benchmark_paths(p)
    benchmark_T = BENCHMARK_T
    c_0_true = float(benchmark(jnp.array([0.0]))[1][0])
    rows = []
    for T in horizons:
        if T <= benchmark_T:
            t = jnp.linspace(0.0, T, quadrature_points)
            k, c = benchmark(t)
            dk, dc = jax.vmap(lambda k, c: field(jnp.array([k, c]), p))(k, c).T
            dmu = -p.sigma_crra * c ** (-p.sigma_crra - 1.0) * dc
            saddle = float(jnp.trapezoid(dk**2 + dmu**2, t))
        else:
            saddle = float("nan")
        rows.append(
            {"kind": "benchmark", "c_0_offset": 0.0, "T": T, "seminorm": saddle}
        )
        for offset in c_offsets:
            c_0 = c_0_true * (1.0 + offset)
            energy, _ = integrate_energy(jnp.array([k_0, c_0, 0.0]), T, shooting_set, p)
            rows.append(
                {
                    "kind": "forward",
                    "c_0_offset": offset,
                    "T": T,
                    "seminorm": float(energy),
                }
            )
    return pd.DataFrame(rows), {"benchmark_T": benchmark_T, "c_0_true": c_0_true}


def neoclassical_growth_shooting(
    seed: int = 0,
    a: float = C["a"],
    delta: float = C["delta"],
    rho_hat: float = C["rho_hat"],
    sigma_crra: float = C["sigma_crra"],
    k_0: float = 1.0,
    shooting_set: ShootingSettings = ShootingSettings(),
    data_set: ShootingDataSettings = ShootingDataSettings(),
    verbose: bool = False,
):
    p = baseline_parameters(k_0, a, delta, rho_hat, sigma_crra)
    k_ss, c_ss = steady_state(p)
    benchmark = benchmark_paths(p)
    c_0_true = float(benchmark(jnp.array([0.0]))[1][0])
    # Guesses above this start by decumulating capital.
    c_0_upper_bound = float(production(1.0, p.k_0, p) - p.delta * p.k_0)

    key = jax.random.PRNGKey(seed)
    c_init = c_0_true * (
        1.0 + shooting_set.radius * jax.random.uniform(key, minval=-1.0, maxval=1.0)
    )

    timer = Stopwatch()
    search_failure = None
    if shooting_set.method == "newton":
        c_0, converged, blew_up, n_iterations, _ = newton_shooting(
            c_init, p.k_0, k_ss, shooting_set.T, shooting_set.max_iter, shooting_set, p
        )
        if bool(blew_up):
            search_failure = "blow_up"
        elif not bool(converged):
            search_failure = "max_iter"
    elif shooting_set.method == "bisection":
        c_0, found, n_iterations = bisection_shooting(
            c_init, p.k_0, k_ss, shooting_set.T, shooting_set.max_iter, shooting_set, p
        )
        if not bool(found):
            search_failure = "budget_exhausted"
    else:
        raise ValueError(f"unknown method {shooting_set.method}")
    solve_time = timer.elapsed(c_0)

    state_T, ok_T = integrate(jnp.array([p.k_0, c_0]), shooting_set.T, shooting_set, p)
    terminal_k_miss = float(jnp.abs(state_T[0] - k_ss) / k_ss)
    terminal_c_gap = float(jnp.abs(state_T[1] - c_ss) / c_ss)
    rejection_reasons = []
    if search_failure is not None:
        rejection_reasons.append(search_failure)
    elif not bool(ok_T):
        rejection_reasons.append("terminal_blow_up")
    elif not terminal_k_miss <= shooting_set.terminal_tol:
        rejection_reasons.append("terminal_tolerance")

    # The returned path on the common test window, NaN throughout if its
    # integration fails, against the boundary-value benchmark.
    t_test = np.linspace(0.0, data_set.test_T, data_set.test_points)
    path, _ = integrate(
        jnp.array([p.k_0, c_0]),
        data_set.test_T,
        shooting_set,
        p,
        ts=jnp.asarray(t_test),
    )
    path = np.asarray(path)
    valid = np.all(np.isfinite(path), axis=1)
    k_test = np.where(valid, path[:, 0], np.nan)
    c_test = np.where(valid, path[:, 1], np.nan)
    k_b, c_b = (np.asarray(x) for x in benchmark(jnp.asarray(t_test)))
    test_df = pd.DataFrame(
        {
            "t": t_test,
            "k": k_test,
            "c": c_test,
            "k_benchmark": k_b,
            "c_benchmark": c_b,
            "k_rel_error": np.abs(k_test - k_b) / np.abs(k_b),
            "c_rel_error": np.abs(c_test - c_b) / np.abs(c_b),
        }
    )

    # The same path on its own horizon [0, T], where the error accumulates.
    t_horizon = np.linspace(0.0, shooting_set.T, data_set.test_points)
    horizon, _ = integrate(
        jnp.array([p.k_0, c_0]),
        shooting_set.T,
        shooting_set,
        p,
        ts=jnp.asarray(t_horizon),
    )
    horizon = np.asarray(horizon)
    horizon_valid = np.all(np.isfinite(horizon), axis=1)
    k_hb, c_hb = (np.asarray(x) for x in benchmark(jnp.asarray(t_horizon)))
    horizon_k_rel_error = np.where(
        horizon_valid, np.abs(horizon[:, 0] - k_hb) / np.abs(k_hb), np.nan
    )[1:]
    horizon_c_rel_error = np.where(
        horizon_valid, np.abs(horizon[:, 1] - c_hb) / np.abs(c_hb), np.nan
    )[1:]

    results = {
        "parameters": p.to_dict(),
        "shooting_set": asdict(shooting_set),
        "data_set": {
            **asdict(data_set),
            "k_ss": k_ss,
            "c_ss": c_ss,
            "c_0_true": c_0_true,
            "c_0_upper_bound": c_0_upper_bound,
        },
        "seed": seed,
        "c_0_init": float(c_init),
        "c_0": float(c_0),
        "c_0_rel_error": abs(float(c_0) - c_0_true) / c_0_true,
        "terminal_k_miss": terminal_k_miss,
        "terminal_c_gap": terminal_c_gap,
        "n_iterations": int(n_iterations),
        "test_k_rel_error_abs": float(np.nanmean(test_df["k_rel_error"]))
        if valid.any()
        else np.nan,
        "test_c_rel_error_abs": float(np.nanmean(test_df["c_rel_error"]))
        if valid.any()
        else np.nan,
        "test_points_finite": int(valid.sum()),
        "horizon_k_rel_error_abs": float(np.nanmean(horizon_k_rel_error))
        if horizon_valid[1:].any()
        else np.nan,
        "horizon_c_rel_error_abs": float(np.nanmean(horizon_c_rel_error))
        if horizon_valid[1:].any()
        else np.nan,
        "horizon_points_finite": int(horizon_valid.sum()),
        "solve_time": solve_time,
        "solve_time_success": solve_time if not rejection_reasons else np.nan,
        **acceptance_fields(rejection_reasons),
    }
    if verbose:
        for name, value in results.items():
            if not isinstance(value, dict):
                print(f"  {name}: {value}")
    return results, {"test_df": test_df}


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jsonargparse.CLI(neoclassical_growth_shooting)
