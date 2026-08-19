from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import struct
from nlls_gram import LevenbergMarquardt, LMStatus

# Reference saddle path of the smooth model by Hermite-Simpson collocation of
# the detrended ODEs, truncated by pinning kappa = kappa_ss at BENCHMARK_T.
# TFP grows at (1 - a) g so the labor-augmenting scale z^{1/(1-a)} grows at g;
# kappa = k / z^{1/(1-a)} and c_tilde = c / z^{1/(1-a)} make the
# balanced-growth boundary stationary. Solved in float64 regardless of the
# caller's working dtype.


def fhat(kappa, a):
    return kappa**a


def fhat_k(kappa, a):
    return a * kappa ** (a - 1.0)


def detrended_rhs(y, p):
    kappa = y[..., 0]
    c_tilde = y[..., 1]
    dkappa = fhat(kappa, p["a"]) - c_tilde - (p["delta"] + p["g"]) * kappa
    dc_tilde = (c_tilde / p["sigma_crra"]) * (
        fhat_k(kappa, p["a"]) - p["rho_hat"] - p["delta"] - p["sigma_crra"] * p["g"]
    )
    return jnp.stack([dkappa, dc_tilde], axis=-1)


# Endpoint pins of the two-point problem on the collocation mesh.
@struct.dataclass
class BenchmarkArgs:
    t: jax.Array
    kappa_0: jax.Array
    kappa_ss: jax.Array


# Hermite-Simpson defects, fourth-order accurate: on each interval the state
# is a cubic through the endpoints and their slopes, and the defect is the
# Simpson quadrature error of the ODE at the implied midpoint. Unknowns are
# c_tilde at every node and kappa at the interior nodes (2n residuals in 2n
# unknowns for n intervals).
def collocation_defects(theta, args, p):
    kappa_interior, c_tilde = theta
    kappa = jnp.concatenate([args.kappa_0, kappa_interior, args.kappa_ss])
    y = jnp.stack([kappa, c_tilde], axis=-1)
    f = detrended_rhs(y, p)
    h = (args.t[1:] - args.t[:-1])[:, None]
    y_0, y_1 = y[:-1], y[1:]
    f_0, f_1 = f[:-1], f[1:]
    y_mid = 0.5 * (y_0 + y_1) + (h / 8.0) * (f_0 - f_1)
    f_mid = detrended_rhs(y_mid, p)
    return (y_1 - y_0 - (h / 6.0) * (f_0 + 4.0 * f_mid + f_1)).reshape(-1)


# Machine-noise defect sum of squares at which the benchmark solve stops;
# named so the paper can cite the reference solution's stopping rule.
BENCHMARK_DEFECT_SSQ = 1e-22

benchmark_solver = LevenbergMarquardt(
    collocation_defects, init_damping=1e-3, geodesic_acceleration=False
)

# Truncation horizon (~20 e-foldings of the calibrated saddle path) and the
# front-loaded mesh: dense over the transition, coarse over the tail. The
# caller's dates are unioned into the mesh, so requested values are exact
# node values, never interpolated in time. The mesh keeps the truncation
# error of the scheme below ~1e-9 relative on the calibrated family
# (verified against scipy solve_bvp at tol 1e-10).
BENCHMARK_T = 200.0
DENSE_T = 50.0
DENSE_POINTS = 801
TAIL_POINTS = 301


@lru_cache(maxsize=None)
def solve_kappa_path(a, delta, rho_hat, sigma_crra, g, kappa_0, times):
    times_array = np.asarray(times, dtype=float)
    if times_array.max() > BENCHMARK_T:
        raise ValueError("extrapolation past the benchmark horizon is not supported")
    mesh = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, DENSE_T, DENSE_POINTS),
                np.linspace(DENSE_T, BENCHMARK_T, TAIL_POINTS),
                times_array,
            ]
        )
    )
    marginal_product_ss = delta + rho_hat + sigma_crra * g
    kappa_ss = (marginal_product_ss / a) ** (1.0 / (a - 1.0))

    # Saddle-shaped initial guess: exponential approach to the steady state at
    # the linearized stable eigenvalue, with steady-flow consumption. A naive
    # flat guess sends Newton to a spurious delayed-takeoff branch on long
    # horizons.
    f_pp = a * (a - 1.0) * kappa_ss ** (a - 2.0)
    c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
    trace = rho_hat + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4.0 * c_tilde_ss * f_pp / sigma_crra
    lambda_stable = 0.5 * (trace - math.sqrt(discriminant))
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * np.exp(lambda_stable * mesh)
    c_tilde_guess = kappa_guess**a - (delta + g) * kappa_guess

    args = BenchmarkArgs(
        t=jnp.asarray(mesh, dtype=jnp.float64),
        kappa_0=jnp.asarray([kappa_0], dtype=jnp.float64),
        kappa_ss=jnp.asarray([kappa_ss], dtype=jnp.float64),
    )
    theta_0 = (
        jnp.asarray(kappa_guess[1:-1], dtype=jnp.float64),
        jnp.asarray(c_tilde_guess, dtype=jnp.float64),
    )
    p = {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in dict(
            a=a, delta=delta, rho_hat=rho_hat, sigma_crra=sigma_crra, g=g
        ).items()
    }
    result = benchmark_solver.solve(
        theta_0,
        args,
        p=p,
        max_steps=100,
        atol=math.sqrt(BENCHMARK_DEFECT_SSQ),
        max_steps_is_success=False,
    )
    if int(result.status) != LMStatus.CONVERGED:
        raise RuntimeError(f"benchmark collocation failed: status {result.status}")
    kappa_interior, c_tilde = result.x
    kappa = np.concatenate([[kappa_0], np.asarray(kappa_interior), [kappa_ss]])
    index = np.searchsorted(mesh, times_array)
    return kappa[index], np.asarray(c_tilde)[index]


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
