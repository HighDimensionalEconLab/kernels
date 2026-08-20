from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tinydiffeq import hermite_interpolate, solve_bvp

# Reference saddle path of the smooth model. The interface is the tidy-frame
# benchmark_df below; the engine behind solve_kappa_path is tinydiffeq's
# solve_bvp (Lobatto IIIA residual-controlled collocation) and is the single
# swap point for a different algorithm. Solved in the detrended coordinates
# kappa = k / z^{1/(1-a)}, c_tilde = c / z^{1/(1-a)}: TFP grows at (1 - a) g
# so the labor-augmenting scale z^{1/(1-a)} grows at g, and the
# balanced-growth boundary is stationary. The solve must run in float64 --
# under float32 the collocation tolerance clamps at ~1e-5, far above
# BENCHMARK_TOL -- so it relies on the drivers' always-on x64 setup and
# fails loudly on a downgraded dtype.

# Truncation horizon (~20 e-foldings of the calibrated saddle path) with the
# terminal pin kappa(T) = kappa_ss; requested dates beyond it would
# extrapolate. The default tol=1e-3 would leave ~1e-4 errors in the
# "benchmark" itself, polluting every reported solver error. MAX_NODES is the
# static padded-mesh budget and must exceed the refined mesh (~1500 nodes at
# the calibrated tolerance).
BENCHMARK_T = 200.0
BENCHMARK_TOL = 1e-10
INITIAL_MESH_POINTS = 801
MAX_NODES = 2048


# kappa' = kappa^a - c_tilde - (delta + g) kappa,
# c_tilde' = (c_tilde / sigma)(a kappa^{a-1} - rho_hat - delta - sigma g).
def kappa_field(t, y, _, args):
    kappa, c_tilde = y
    a, delta, rho_hat = args["a"], args["delta"], args["rho_hat"]
    sigma_crra, g = args["sigma_crra"], args["g"]
    return jnp.array(
        [
            kappa**a - c_tilde - (delta + g) * kappa,
            (c_tilde / sigma_crra)
            * (a * kappa ** (a - 1.0) - rho_hat - delta - sigma_crra * g),
        ]
    )


def kappa_bc(ya, yb, _, args):
    return jnp.array([ya[0] - args["kappa_0"], yb[0] - args["kappa_ss"]])


@jax.jit
def solve_kappa_bvp(mesh, y_guess, args):
    return solve_bvp(
        kappa_field,
        kappa_bc,
        mesh,
        y_guess,
        args=args,
        tol=BENCHMARK_TOL,
        max_nodes=MAX_NODES,
    )


@lru_cache(maxsize=None)
def solve_kappa_path(a, delta, rho_hat, sigma_crra, g, kappa_0, times):
    times_array = np.asarray(times, dtype=float)
    if times_array.max() > BENCHMARK_T:
        raise ValueError("extrapolation past the benchmark horizon is not supported")
    mesh = np.linspace(0.0, BENCHMARK_T, INITIAL_MESH_POINTS)
    marginal_product_ss = delta + rho_hat + sigma_crra * g
    kappa_ss = (marginal_product_ss / a) ** (1.0 / (a - 1.0))

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
    y_guess = jnp.asarray(np.stack([kappa_guess, c_tilde_guess], axis=-1))
    args = {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in dict(
            a=a,
            delta=delta,
            rho_hat=rho_hat,
            sigma_crra=sigma_crra,
            g=g,
            kappa_0=kappa_0,
            kappa_ss=kappa_ss,
        ).items()
    }
    solution = solve_kappa_bvp(jnp.asarray(mesh), y_guess, args)
    # The Hermite interpolant is the solver's own C1 collocation spline,
    # accurate to BENCHMARK_TOL everywhere, so requested dates need not be
    # mesh nodes.
    paths = hermite_interpolate(
        jnp.asarray(times_array), solution.t, solution.y, solution.yp
    )
    kappa = np.asarray(paths[:, 0])
    c_tilde = np.asarray(paths[:, 1])
    if not bool(solution.ok) or kappa.dtype != np.float64:
        raise RuntimeError(
            f"solve_bvp failed: status {int(solution.status)}, dtype {kappa.dtype}"
        )
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
