from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from tinydiffeq import solve_bvp

# Reference saddle path via tinydiffeq's solve_bvp (Lobatto IIIA
# residual-controlled collocation), solved in the detrended coordinates
# kappa = k / z^{1/(1-a)}, c_tilde = c / z^{1/(1-a)}: TFP grows at
# (1 - a) g so the labor-augmenting scale z^{1/(1-a)} grows at g, and the
# balanced-growth boundary is stationary. Smooth production (b_1 <= 1) has a
# unique saddle path; the butterfly family (b_1 > 1) has one saddle per
# production branch and the benchmark keeps the DNSS-selected candidate.


@struct.dataclass
class BenchmarkParameters:
    T: float = 200.0
    tol: float = 1e-10
    initial_points: int = struct.field(pytree_node=False, default=201)
    max_nodes: int = struct.field(pytree_node=False, default=2048)


# kappa' = kappa^a - c_tilde - (delta + g) kappa,
# c_tilde' = (c_tilde / sigma)(a kappa^{a-1} - rho_hat - delta - sigma g).
def kappa_field(t, y, _, p):
    kappa, c_tilde = y
    return jnp.array(
        [
            kappa**p.a - c_tilde - (p.delta + p.g) * kappa,
            (c_tilde / p.sigma_crra)
            * (p.a * kappa ** (p.a - 1.0) - p.rho_hat - p.delta - p.sigma_crra * p.g),
        ]
    )


def kappa_bc(ya, yb, _, p):
    marginal_product_ss = p.delta + p.rho_hat + p.sigma_crra * p.g
    kappa_ss = (marginal_product_ss / p.a) ** (1.0 / (p.a - 1.0))
    kappa_0 = p.k_0 / p.z_0 ** (1.0 / (1.0 - p.a))
    return jnp.array([ya[0] - kappa_0, yb[0] - kappa_ss])


@jax.jit
def solve_kappa_bvp(p, bench_set):
    a, delta, rho_hat = p.a, p.delta, p.rho_hat
    sigma_crra, g = p.sigma_crra, p.g
    kappa_0 = p.k_0 / p.z_0 ** (1.0 / (1.0 - a))
    mesh = jnp.linspace(0.0, bench_set.T, bench_set.initial_points)
    marginal_product_ss = delta + rho_hat + sigma_crra * g
    kappa_ss = (marginal_product_ss / a) ** (1.0 / (a - 1.0))
    # Saddle-shaped initial guess
    f_pp = a * (a - 1.0) * kappa_ss ** (a - 2.0)
    c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
    trace = rho_hat + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4.0 * c_tilde_ss * f_pp / sigma_crra
    lambda_stable = 0.5 * (trace - jnp.sqrt(discriminant))
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * jnp.exp(lambda_stable * mesh)
    c_tilde_guess = kappa_guess**a - (delta + g) * kappa_guess
    y_guess = jnp.stack([kappa_guess, c_tilde_guess], axis=-1)
    return solve_bvp(
        kappa_field,
        kappa_bc,
        mesh,
        y_guess,
        args=p,
        tol=bench_set.tol,
        max_nodes=bench_set.max_nodes,
    )


# Butterfly production f(z, k) = z A max(k^a, b_1 k^a - b_2) in the detrended
# coordinates: f/Z = A max(kappa^a, b_1 kappa^a - b_2 Z^{-a}) with
# Z = z_0^{1/(1-a)} e^{g t}, so the branch offset (and with it the switch
# kappa_thr(t) = k_thr z_0^{-1/(1-a)} e^{-g t}) decays at rate a g. A saddle
# candidate whose kappa_0 and terminal steady state share a side of the
# switch never touches it and solves on the true kinked field below; a
# candidate that straddles the switch must cross it, where the f' jump stalls
# single-domain collocation, so crossing candidates use the two-phase solver
# with a free crossing time instead.
def butterfly_kappa_field(t, y, _, args):
    p, _ = args
    kappa, c_tilde = y
    base = kappa**p.a
    b_2_detrended = p.b_2 * p.z_0 ** (-p.a / (1.0 - p.a)) * jnp.exp(-p.a * p.g * t)
    high = p.b_1 * base - b_2_detrended
    f = p.A * jnp.maximum(base, high)
    f_prime = p.A * jnp.where(high > base, p.b_1, 1.0) * p.a * kappa ** (p.a - 1.0)
    return jnp.array(
        [
            f - c_tilde - (p.delta + p.g) * kappa,
            (c_tilde / p.sigma_crra)
            * (f_prime - p.rho_hat - p.delta - p.sigma_crra * p.g),
        ]
    )


def butterfly_kappa_bc(ya, yb, _, args):
    p, kappa_ss = args
    kappa_0 = p.k_0 / p.z_0 ** (1.0 / (1.0 - p.a))
    return jnp.array([ya[0] - kappa_0, yb[0] - kappa_ss])


# Same-side saddle candidate: branch_slope in {1, b_1} picks the steady state
# pinned at the terminal boundary, and the initial guess rides that branch's
# linearized stable manifold. branch_slope must arrive as an array of p's
# dtype so both candidates share one compilation.
@jax.jit
def solve_butterfly_kappa_bvp(p, bench_set, branch_slope):
    a, delta, rho_hat = p.a, p.delta, p.rho_hat
    sigma_crra, g, s = p.sigma_crra, p.g, branch_slope
    kappa_0 = p.k_0 / p.z_0 ** (1.0 / (1.0 - a))
    mesh = jnp.linspace(0.0, bench_set.T, bench_set.initial_points)
    marginal_product_ss = delta + rho_hat + sigma_crra * g
    kappa_ss = (marginal_product_ss / (p.A * a * s)) ** (1.0 / (a - 1.0))
    offset = (
        jnp.where(s > 1.0, 1.0, 0.0)
        * p.b_2
        * p.z_0 ** (-a / (1.0 - a))
        * jnp.exp(-a * g * mesh)
    )
    # Saddle-shaped initial guess on the candidate branch
    f_pp = p.A * s * a * (a - 1.0) * kappa_ss ** (a - 2.0)
    c_tilde_ss = p.A * (s * kappa_ss**a - offset[0]) - (delta + g) * kappa_ss
    trace = rho_hat + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4.0 * c_tilde_ss * f_pp / sigma_crra
    lambda_stable = 0.5 * (trace - jnp.sqrt(discriminant))
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * jnp.exp(lambda_stable * mesh)
    c_tilde_guess = p.A * (s * kappa_guess**a - offset) - (delta + g) * kappa_guess
    y_guess = jnp.stack([kappa_guess, c_tilde_guess], axis=-1)
    return solve_bvp(
        butterfly_kappa_field,
        butterfly_kappa_bc,
        mesh,
        y_guess,
        args=(p, kappa_ss),
        tol=bench_set.tol,
        max_nodes=bench_set.max_nodes,
    )


# Branch-restricted detrended field, slope s in {1, b_1}; only the high
# branch carries the decaying b_2 offset.
def branch_field(t, y, s, p):
    kappa, c_tilde = y[0], y[1]
    offset = (
        jnp.where(s > 1.0, 1.0, 0.0)
        * p.b_2
        * p.z_0 ** (-p.a / (1.0 - p.a))
        * jnp.exp(-p.a * p.g * t)
    )
    f = p.A * (s * kappa**p.a - offset)
    f_prime = p.A * s * p.a * kappa ** (p.a - 1.0)
    return jnp.array(
        [
            f - c_tilde - (p.delta + p.g) * kappa,
            (c_tilde / p.sigma_crra)
            * (f_prime - p.rho_hat - p.delta - p.sigma_crra * p.g),
        ]
    )


# Crossing extremal as a two-phase BVP on s in [0, 1] with the crossing time
# tau a jointly solved unknown (z): y = (kappa_a, c_a, kappa_b, c_b), phase a
# on t = tau s with the initial side's branch, phase b on t = tau + (T - tau) s
# with the terminal side's branch. Each phase's field is smooth, which is
# what lets collocation converge where the single-domain kinked field cannot.
def butterfly_crossing_field(s_node, y, z, args):
    p, T, s_a, s_b, _ = args
    t_a = z * s_node
    t_b = z + (T - z) * s_node
    return jnp.concatenate(
        [
            z * branch_field(t_a, y[:2], s_a, p),
            (T - z) * branch_field(t_b, y[2:], s_b, p),
        ]
    )


# Initial condition, phase a ending on the (moving) switch at tau, state
# continuity across the phases, and the terminal steady state.
def butterfly_crossing_bc(ya, yb, z, args):
    p, T, s_a, s_b, kappa_ss_b = args
    kappa_0 = p.k_0 / p.z_0 ** (1.0 / (1.0 - p.a))
    switch_tau = p.k_thr * p.z_0 ** (-1.0 / (1.0 - p.a)) * jnp.exp(-p.g * z)
    return jnp.array(
        [
            ya[0] - kappa_0,
            yb[0] - switch_tau,
            ya[2] - yb[0],
            ya[3] - yb[1],
            yb[2] - kappa_ss_b,
        ]
    )


# Crossing candidate: terminal_slope names the branch of the terminal steady
# state; phase a rides the other branch from kappa_0 to the switch. Guesses:
# tau from a crossing speed of order 0.1, a linear phase-a path with the
# implied resource-constraint consumption, and phase b on the terminal
# branch's linearized stable manifold.
@jax.jit
def solve_butterfly_crossing_bvp(p, bench_set, terminal_slope):
    a, delta, rho_hat = p.a, p.delta, p.rho_hat
    sigma_crra, g, s_b = p.sigma_crra, p.g, terminal_slope
    T = bench_set.T
    s_a = jnp.where(s_b > 1.0, 1.0, p.b_1)
    kappa_0 = p.k_0 / p.z_0 ** (1.0 / (1.0 - a))
    switch_0 = p.k_thr * p.z_0 ** (-1.0 / (1.0 - a))
    marginal_product_ss = delta + rho_hat + sigma_crra * g
    kappa_ss_b = (marginal_product_ss / (p.A * a * s_b)) ** (1.0 / (a - 1.0))
    mesh = jnp.linspace(0.0, 1.0, bench_set.initial_points)
    tau_guess = jnp.clip(jnp.abs(kappa_0 - switch_0) / 0.1, 1e-3, 0.25 * T)

    kappa_a_guess = kappa_0 + (switch_0 - kappa_0) * mesh
    offset_a = (
        jnp.where(s_a > 1.0, 1.0, 0.0)
        * p.b_2
        * p.z_0 ** (-a / (1.0 - a))
        * jnp.exp(-a * g * tau_guess * mesh)
    )
    c_a_guess = (
        p.A * (s_a * kappa_a_guess**a - offset_a)
        - (delta + g) * kappa_a_guess
        - (switch_0 - kappa_0) / tau_guess
    )

    t_b_guess = tau_guess + (T - tau_guess) * mesh
    offset_b = (
        jnp.where(s_b > 1.0, 1.0, 0.0)
        * p.b_2
        * p.z_0 ** (-a / (1.0 - a))
        * jnp.exp(-a * g * t_b_guess)
    )
    f_pp = p.A * s_b * a * (a - 1.0) * kappa_ss_b ** (a - 2.0)
    c_tilde_ss = p.A * (s_b * kappa_ss_b**a - offset_b[-1]) - (delta + g) * kappa_ss_b
    trace = rho_hat + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4.0 * c_tilde_ss * f_pp / sigma_crra
    lambda_stable = 0.5 * (trace - jnp.sqrt(discriminant))
    kappa_b_guess = kappa_ss_b + (switch_0 - kappa_ss_b) * jnp.exp(
        lambda_stable * (t_b_guess - tau_guess)
    )
    c_b_guess = p.A * (s_b * kappa_b_guess**a - offset_b) - (delta + g) * kappa_b_guess

    y_guess = jnp.stack([kappa_a_guess, c_a_guess, kappa_b_guess, c_b_guess], axis=-1)
    return solve_bvp(
        butterfly_crossing_field,
        butterfly_crossing_bc,
        mesh,
        y_guess,
        tau_guess,
        args=(p, T, s_a, s_b, kappa_ss_b),
        tol=bench_set.tol,
        max_nodes=bench_set.max_nodes,
    )


# One candidate's discounted utility (trapezoid on dense host-side grids, per
# phase for crossing paths since the assembled consumption is kinked at tau)
# and its detrended path t -> (kappa, c_tilde). Returns (-inf, None) when the
# BVP fails, tau leaves (0, T), a crossing phase leaves its branch's side of
# the switch (its restricted field is then not the model's field along the
# path), or the path leaves the positive orthant (a failed solve's iterate
# can be finite-looking garbage, so .ok comes first).
SIDE_SLACK = 1e-6


def butterfly_candidate_value(p, bench_set, branch_slope, num_points=4001):
    crosses = (float(p.k_0) > float(p.k_thr)) != (float(branch_slope) > 1.0)
    slope = jnp.asarray(branch_slope, dtype=p.b_1.dtype)
    T = float(bench_set.T)
    g = float(p.g)
    scale_0 = float(p.z_0) ** (1.0 / (1.0 - float(p.a)))
    if crosses:
        solution = solve_butterfly_crossing_bvp(p, bench_set, slope)
        tau = float(solution.z)
        if not (bool(solution.ok) and 0.0 < tau < T):
            return -np.inf, None
        s_dense = np.linspace(0.0, 1.0, num_points)
        state = np.asarray(solution(jnp.asarray(s_dense)))
        if not (state > 0.0).all():
            return -np.inf, None
        t_a = tau * s_dense
        t_b = tau + (T - tau) * s_dense
        switch_a = float(p.k_thr) / scale_0 * np.exp(-g * t_a)
        switch_b = float(p.k_thr) / scale_0 * np.exp(-g * t_b)
        if float(p.k_0) > float(p.k_thr):
            side_ok = (state[:, 0] >= switch_a - SIDE_SLACK).all() and (
                state[:, 2] <= switch_b + SIDE_SLACK
            ).all()
        else:
            side_ok = (state[:, 0] <= switch_a + SIDE_SLACK).all() and (
                state[:, 2] >= switch_b - SIDE_SLACK
            ).all()
        if not side_ok:
            return -np.inf, None
        segments = [(t_a, state[:, 1]), (t_b, state[:, 3])]

        def path_detrended(t):
            t = jnp.asarray(t)
            state_a = solution(jnp.clip(t / tau, 0.0, 1.0))
            state_b = solution(jnp.clip((t - tau) / (T - tau), 0.0, 1.0))
            in_a = t <= tau
            kappa = jnp.where(in_a, state_a[:, 0], state_b[:, 2])
            c_tilde = jnp.where(in_a, state_a[:, 1], state_b[:, 3])
            return kappa, c_tilde
    else:
        solution = solve_butterfly_kappa_bvp(p, bench_set, slope)
        if not bool(solution.ok):
            return -np.inf, None
        t_dense = np.linspace(0.0, T, num_points)
        state = np.asarray(solution(jnp.asarray(t_dense)))
        if not (state > 0.0).all():
            return -np.inf, None
        segments = [(t_dense, state[:, 1])]

        def path_detrended(t):
            state = solution(jnp.asarray(t))
            return state[:, 0], state[:, 1]

    sigma_crra, rho_hat = float(p.sigma_crra), float(p.rho_hat)
    value = 0.0
    for t_segment, c_tilde in segments:
        c = scale_0 * np.exp(g * t_segment) * c_tilde
        u = (
            np.log(c)
            if sigma_crra == 1.0
            else c ** (1.0 - sigma_crra) / (1.0 - sigma_crra)
        )
        value += float(np.trapezoid(np.exp(-rho_hat * t_segment) * u, t_segment))
    return value, path_detrended


# The crossing solves cannot reach 1e-10 within the node budget (phase b
# spans the full horizon), so the butterfly family defaults to 1e-8; an
# explicitly passed bench_set is used verbatim.
BUTTERFLY_BENCHMARK_TOL = 1e-8


# Returns paths(t) -> (k, c) level arrays, re-trended
def neoclassical_growth_benchmark(p, bench_set=None):
    if bench_set is None:
        bench_set = (
            BenchmarkParameters()
            if float(p.b_1) <= 1.0
            else BenchmarkParameters(tol=BUTTERFLY_BENCHMARK_TOL)
        )
    if float(p.b_1) <= 1.0:
        solution = solve_kappa_bvp(p, bench_set)
        if not solution.ok:
            raise RuntimeError(f"solve_bvp failed: status {int(solution.status)}")

        def path_detrended(t):
            state = solution(jnp.asarray(t))
            return state[:, 0], state[:, 1]
    else:
        # DNSS selection (Skiba; Beyn-Pampel-Semmler): keep the candidate with
        # the higher discounted utility. The two-candidate construction
        # presumes g = 0 (with g > 0 the detrended switch decays and only the
        # high branch is asymptotically stationary); the fields stay general
        # so the crossover diagnostics can probe small g.
        candidates = [
            butterfly_candidate_value(p, bench_set, s) for s in (1.0, float(p.b_1))
        ]
        value, path_detrended = max(candidates, key=lambda pair: pair[0])
        if not np.isfinite(value):
            raise RuntimeError("no butterfly saddle candidate solved")

    def paths(t):
        t = jnp.asarray(t)
        kappa, c_tilde = path_detrended(t)
        scale = p.z_0 ** (1.0 / (1.0 - p.a)) * jnp.exp(p.g * t)
        return scale * kappa, scale * c_tilde

    return paths
