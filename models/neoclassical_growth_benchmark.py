from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from tinydiffeq import solve_bvp

# Reference saddle path of the smooth model via tinydiffeq's solve_bvp
# (Lobatto IIIA residual-controlled collocation), solved in the detrended
# coordinates kappa = k / z^{1/(1-a)}, c_tilde = c / z^{1/(1-a)}: TFP grows at
# (1 - a) g so the labor-augmenting scale z^{1/(1-a)} grows at g, and the
# balanced-growth boundary is stationary.


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


# Returns paths(t) -> (k, c) level arrays, re-trended
def neoclassical_growth_benchmark(p, bench_set=BenchmarkParameters()):
    solution = solve_kappa_bvp(p, bench_set)
    if not solution.ok:
        raise RuntimeError(f"solve_bvp failed: status {int(solution.status)}")

    def paths(t):
        t = jnp.asarray(t)
        state = solution(t)
        scale = p.z_0 ** (1.0 / (1.0 - p.a)) * jnp.exp(p.g * t)
        return scale * state[:, 0], scale * state[:, 1]

    return paths
