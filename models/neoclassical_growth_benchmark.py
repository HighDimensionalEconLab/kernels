from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp

# Reference saddle path of the smooth model. The interface is the tidy-frame
# benchmark_df below; the engine behind solve_kappa_path is scipy's
# solve_bvp (bvp4c-style residual-controlled collocation) and is the single
# swap point for a different algorithm. Solved in the detrended coordinates
# kappa = k / z^{1/(1-a)}, c_tilde = c / z^{1/(1-a)}: TFP grows at (1 - a) g
# so the labor-augmenting scale z^{1/(1-a)} grows at g, and the
# balanced-growth boundary is stationary. Pure numpy: a jax ODE here would
# inherit the caller's x64 flag, and under float32 solve_bvp can never
# verify BENCHMARK_TOL and exhausts max_nodes.

# Truncation horizon (~20 e-foldings of the calibrated saddle path) with the
# terminal pin kappa(T) = kappa_ss; requested dates beyond it would
# extrapolate. scipy's default tol=1e-3 leaves ~1e-4 errors in the
# "benchmark" itself, polluting every reported solver error.
BENCHMARK_T = 200.0
BENCHMARK_TOL = 1e-10
INITIAL_MESH_POINTS = 801
MAX_MESH_NODES = 100000


@lru_cache(maxsize=None)
def solve_kappa_path(a, delta, rho_hat, sigma_crra, g, kappa_0, times):
    times_array = np.asarray(times, dtype=float)
    if times_array.max() > BENCHMARK_T:
        raise ValueError("extrapolation past the benchmark horizon is not supported")
    mesh = np.linspace(0.0, BENCHMARK_T, INITIAL_MESH_POINTS)
    marginal_product_ss = delta + rho_hat + sigma_crra * g
    kappa_ss = (marginal_product_ss / a) ** (1.0 / (a - 1.0))

    def ODE(t, y):
        kappa = y[0]
        c_tilde = y[1]
        return np.vstack(
            (
                kappa**a - c_tilde - (delta + g) * kappa,
                (c_tilde / sigma_crra)
                * (a * kappa ** (a - 1) - rho_hat - delta - sigma_crra * g),
            )
        )

    def bc(ya, yb):
        return np.array([ya[0] - kappa_0, yb[0] - kappa_ss])

    # Saddle-shaped initial guess: exponential approach to the steady state at
    # the linearized stable eigenvalue, with steady-flow consumption. A naive
    # flat guess sends Newton to a spurious delayed-takeoff branch on long
    # horizons.
    f_pp = a * (a - 1) * kappa_ss ** (a - 2)
    c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
    trace = rho_hat + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4 * c_tilde_ss * f_pp / sigma_crra
    lambda_stable = 0.5 * (trace - np.sqrt(discriminant))
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * np.exp(lambda_stable * mesh)
    c_tilde_guess = kappa_guess**a - (delta + g) * kappa_guess
    solution = solve_bvp(
        ODE,
        bc,
        mesh,
        np.vstack((kappa_guess, c_tilde_guess)),
        tol=BENCHMARK_TOL,
        max_nodes=MAX_MESH_NODES,
    )
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed: {solution.message}")
    # solution.sol is the solver's own C1 collocation spline, accurate to
    # BENCHMARK_TOL everywhere, so requested dates need not be mesh nodes.
    kappa, c_tilde = solution.sol(times_array)
    return kappa, c_tilde


# Re-trended benchmark paths over (z_0, k_0) initial conditions as a tidy
# (trajectory, t) DataFrame with detrended (kappa, c_tilde) and level (k, c)
# columns.
def benchmark_df(initial_conditions, times, a, delta, rho_hat, sigma_crra, g=0.0):
    times = np.asarray(times, dtype=float)
    pieces = []
    for trajectory, (z_0, k_0) in enumerate(
        np.asarray(initial_conditions, dtype=float)
    ):
        scale_0 = z_0 ** (1.0 / (1.0 - a))
        kappa, c_tilde = solve_kappa_path(
            float(a),
            float(delta),
            float(rho_hat),
            float(sigma_crra),
            float(g),
            float(k_0 / scale_0),
            tuple(times.tolist()),
        )
        scale = scale_0 * np.exp(g * times)
        pieces.append(
            pd.DataFrame(
                {
                    "trajectory": trajectory,
                    "t": times,
                    "kappa": kappa,
                    "c_tilde": c_tilde,
                    "k": scale * kappa,
                    "c": scale * c_tilde,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)
