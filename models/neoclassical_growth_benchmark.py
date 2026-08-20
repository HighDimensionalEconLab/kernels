from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from tinydiffeq import solve_bvp

# Reference saddle path of the smooth model via tinydiffeq's solve_bvp
# (Lobatto IIIA residual-controlled collocation), solved in the detrended
# coordinates kappa = k / z^{1/(1-a)}, c_tilde = c / z^{1/(1-a)}: TFP grows at
# (1 - a) g so the labor-augmenting scale z^{1/(1-a)} grows at g, and the
# balanced-growth boundary is stationary. The solve relies on the drivers'
# always-on x64 setup -- under float32 the collocation tolerance clamps at
# ~1e-5, far above tol.


# T is the truncation horizon (~20 e-foldings of the calibrated saddle path)
# with the terminal pin kappa(T) = kappa_ss. The solver's default tol=1e-3
# would leave ~1e-4 errors in the "benchmark" itself, polluting every reported
# solver error. max_nodes is the static padded-mesh budget and must exceed the
# refined mesh (~1600 nodes at the calibrated tolerance).
@struct.dataclass
class BenchmarkParameters:
    T: float = 200.0
    tol: float = 1e-10
    initial_points: int = struct.field(pytree_node=False, default=201)
    max_nodes: int = struct.field(pytree_node=False, default=2048)


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
    a = args["a"]
    marginal_product_ss = (
        args["delta"] + args["rho_hat"] + args["sigma_crra"] * args["g"]
    )
    kappa_ss = (marginal_product_ss / a) ** (1.0 / (a - 1.0))
    return jnp.array([ya[0] - args["kappa_0"], yb[0] - kappa_ss])


@jax.jit
def solve_kappa_bvp(args, bench_set):
    a, delta, rho_hat = args["a"], args["delta"], args["rho_hat"]
    sigma_crra, g, kappa_0 = args["sigma_crra"], args["g"], args["kappa_0"]
    mesh = jnp.linspace(0.0, bench_set.T, bench_set.initial_points)
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
    lambda_stable = 0.5 * (trace - jnp.sqrt(discriminant))
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * jnp.exp(lambda_stable * mesh)
    c_tilde_guess = kappa_guess**a - (delta + g) * kappa_guess
    y_guess = jnp.stack([kappa_guess, c_tilde_guess], axis=-1)
    return solve_bvp(
        kappa_field,
        kappa_bc,
        mesh,
        y_guess,
        args=args,
        tol=bench_set.tol,
        max_nodes=bench_set.max_nodes,
    )


# Returns paths(t) -> (k, c) level arrays, re-trended from the solver's own C1
# collocation spline, accurate to tol everywhere between mesh nodes. paths is
# pure jnp (jit/grad/vmap-safe); the Hermite spline clamps flat past T, so
# callers stay inside the horizon.
def neoclassical_growth_benchmark(
    a,
    delta,
    rho_hat,
    sigma_crra,
    g=0.0,
    z_0=1.0,
    k_0=1.0,
    bench_set=BenchmarkParameters(),
):
    scale_0 = z_0 ** (1.0 / (1.0 - a))
    args = {
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "g": g,
        "kappa_0": k_0 / scale_0,
    }
    solution = solve_kappa_bvp(args, bench_set)
    if not solution.ok:
        raise RuntimeError(f"solve_bvp failed: status {int(solution.status)}")

    def paths(t):
        t = jnp.asarray(t)
        state = solution(t)
        scale = scale_0 * jnp.exp(g * t)
        return scale * state[:, 0], scale * state[:, 1]

    return paths
