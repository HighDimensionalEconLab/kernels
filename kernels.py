import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import config
from jax.scipy.special import erf
from quadax import quadgk

# Everything downstream assumes float64 Gram matrices; enabling x64 here (the
# lowest-level module) also keeps quadax from caching float32 quadrature state
# when a consumer touches the kernels before flipping the flag.
config.update("jax_enable_x64", True)

# Stationary Matern kernels with length scale ell (the paper's l), smoothness nu,
# and amplitude sigma, plus their Gaussian/RBF limit (nu = math.inf). The
# integrated kernels K_tilde(t_i, t_j) = int_0^{t_i} K(t, t_j) dt implement the
# integrated RKHS representation x(t) = x_0 + K_tilde alpha.


def matern_kernel_0p5(t_i, t_j, sigma, ell):
    d = jnp.abs(t_i - t_j)
    return sigma**2 * jnp.exp(-d / ell)


def matern_kernel_1p5(t_i, t_j, sigma, ell):
    d = jnp.abs(t_i - t_j)
    exponent = jnp.sqrt(3) * d / ell
    return sigma**2 * (1 + exponent) * jnp.exp(-exponent)


def matern_kernel_2p5(t_i, t_j, sigma, ell):
    d = jnp.abs(t_i - t_j)
    exponent = jnp.sqrt(5) * d / ell
    term = (5 * d**2) / (3 * ell**2)
    return sigma**2 * (1 + exponent + term) * jnp.exp(-exponent)


def gaussian_kernel(t_i, t_j, sigma, ell):
    d = jnp.abs(t_i - t_j)
    return sigma**2 * jnp.exp(-(d**2) / (2 * ell**2))


def integrated_matern_kernel_0p5(t_i, t_j, sigma, ell):
    s = t_i - t_j
    d = jnp.abs(s)

    def if_s_neg():
        return ell * (sigma**2) * (jnp.exp(-d / ell) - jnp.exp(-t_j / ell))

    def if_s_non_neg():
        return ell * (sigma**2) * (2 - jnp.exp(-d / ell) - jnp.exp(-t_j / ell))

    return jax.lax.cond(s < 0, if_s_neg, if_s_non_neg)


def integrated_matern_kernel_1p5(t_i, t_j, sigma, ell, quad_tol=1e-7):
    def matern_integrand(t):
        return matern_kernel_1p5(t, t_j, sigma, ell)

    integral, info = quadgk(
        matern_integrand, [0, t_i], epsabs=quad_tol, epsrel=quad_tol
    )
    return jnp.where(t_i == 0, 0, integral)


def integrated_matern_kernel_2p5(t_i, t_j, sigma, ell, quad_tol=1e-7):
    def matern_integrand(t):
        return matern_kernel_2p5(t, t_j, sigma, ell)

    integral, info = quadgk(
        matern_integrand, [0, t_i], epsabs=quad_tol, epsrel=quad_tol
    )
    return jnp.where(t_i == 0, 0, integral)


def integrated_gaussian_kernel(t_i, t_j, sigma, ell):
    # int_0^{t_i} exp(-(t - t_j)^2 / (2 ell^2)) dt in closed form via erf;
    # the odd symmetry of erf makes the t_i = 0 value exactly zero.
    root_2_ell = jnp.sqrt(2.0) * ell
    scale = sigma**2 * ell * jnp.sqrt(jnp.pi / 2.0)
    return scale * (erf((t_i - t_j) / root_2_ell) + erf(t_j / root_2_ell))


# Adaptive-quadrature cross-check for integrated_gaussian_kernel (tests only).
def integrated_gaussian_kernel_quad(t_i, t_j, sigma, ell, quad_tol=1e-7):
    def gaussian_integrand(t):
        return gaussian_kernel(t, t_j, sigma, ell)

    integral, info = quadgk(
        gaussian_integrand, [0, t_i], epsabs=quad_tol, epsrel=quad_tol
    )
    return jnp.where(t_i == 0, 0, integral)


# nu is static: the kernel pair is selected at trace time.
@partial(jax.jit, static_argnums=(2,))
def integrated_kernel_matrices(t_i, t_j, nu, sigma, ell):
    if math.isinf(nu):
        kernel, integrated_kernel = gaussian_kernel, integrated_gaussian_kernel
    elif nu == 0.5:
        kernel, integrated_kernel = matern_kernel_0p5, integrated_matern_kernel_0p5
    elif nu == 1.5:
        kernel, integrated_kernel = matern_kernel_1p5, integrated_matern_kernel_1p5
    elif nu == 2.5:
        kernel, integrated_kernel = matern_kernel_2p5, integrated_matern_kernel_2p5
    else:
        raise ValueError(f"Unsupported nu: {nu!r} (use 0.5, 1.5, 2.5, or math.inf)")
    K = jax.vmap(
        jax.vmap(kernel, in_axes=(None, 0, None, None)),
        in_axes=(0, None, None, None),
    )(t_i, t_j, sigma, ell)
    K_tilde = jax.vmap(
        jax.vmap(integrated_kernel, in_axes=(None, 0, None, None)),
        in_axes=(0, None, None, None),
    )(t_i, t_j, sigma, ell)
    return K, K_tilde


# Matrix-free O(n) matvecs for the Matern-1/2 kernel and its integrated kernel
# on a sorted strictly increasing grid t. The exponential terms telescope
# through the classic Rybicki-Press forward/backward recurrences with
# e_i = exp(-(t_{i+1} - t_i)/ell):
#   lower_{i+1} = e_i (lower_i + v_i)          lower_i = sum_{j<i} e^{-(t_i-t_j)/ell} v_j
#   upper_i     = e_i (upper_{i+1} + v_{i+1})  upper_i = sum_{j>i} e^{-(t_j-t_i)/ell} v_j
# and the remaining integrated-kernel terms are prefix/suffix sums. Ties (the
# diagonal) fall in the lower/prefix branch, matching integrated_matern_kernel_0p5.
def matern_0p5_exp_recurrences(t, ell, v):
    e = jnp.exp(-jnp.diff(t) / ell)
    zero = jnp.zeros((), dtype=v.dtype)
    edge = jnp.zeros((1,), dtype=v.dtype)

    def step(carry, inputs):
        e_i, v_i = inputs
        out = e_i * (carry + v_i)
        return out, out

    _, lower_rest = jax.lax.scan(step, zero, (e, v[:-1]))
    lower = jnp.concatenate([edge, lower_rest])
    _, upper_head = jax.lax.scan(step, zero, (e, v[1:]), reverse=True)
    upper = jnp.concatenate([upper_head, edge])
    return lower, upper


def matern_kernel_0p5_matvec(t, v, sigma, ell):
    lower, upper = matern_0p5_exp_recurrences(t, ell, v)
    return sigma**2 * (lower + v + upper)


def integrated_matern_0p5_terms(t, ell, v, lower, upper):
    # Closed form split by t_j <= t_i vs t_j > t_i (see
    # integrated_matern_kernel_0p5): (K_tilde v)_i = ell [A - B + C - D]
    # with A the inclusive cumsum of (2 - e^{-t_j/ell}) v_j, B the inclusive
    # forward recurrence, C the strict backward recurrence, and D the strict
    # suffix sum of e^{-t_j/ell} v_j.
    exp_neg = jnp.exp(-t / ell)
    prefix = jnp.cumsum((2.0 - exp_neg) * v)
    weighted = exp_neg * v
    suffix = jnp.cumsum(weighted[::-1])[::-1] - weighted
    return ell * (prefix - (lower + v) + upper - suffix)


def integrated_matern_kernel_0p5_matvec(t, v, sigma, ell):
    lower, upper = matern_0p5_exp_recurrences(t, ell, v)
    return sigma**2 * integrated_matern_0p5_terms(t, ell, v, lower, upper)


# Both matvecs from one recurrence pass: the sequential scans dominate the
# matrix-free residual, and each JVP/VJP inside the solver's CG loop replays
# them, so sharing halves the hot-path work.
def matern_kernel_0p5_matvec_pair(t, v, sigma, ell):
    lower, upper = matern_0p5_exp_recurrences(t, ell, v)
    k_v = sigma**2 * (lower + v + upper)
    k_tilde_v = sigma**2 * integrated_matern_0p5_terms(t, ell, v, lower, upper)
    return k_v, k_tilde_v


# Closed-form tridiagonal precision of the Matern-1/2 Gram on a sorted strictly
# increasing grid (the OU/AR(1) Markov property): with rho_i = e^{-dt_i/ell},
# T = K^{-1} has diag (1/sigma^2)[1_{i=1} + 1_{i>1}/(1-rho_{i-1}^2)
# + 1_{i<n} rho_i^2/(1-rho_i^2)] and off-diagonal -(1/sigma^2) rho_i/(1-rho_i^2).
def matern_0p5_tridiagonal_precision(t, sigma, ell):
    rho = jnp.exp(-jnp.diff(t) / ell)
    one_minus_rho_sq = 1.0 - rho**2
    diag = (
        jnp.concatenate([jnp.ones(1), 1.0 / one_minus_rho_sq])
        + jnp.concatenate([rho**2 / one_minus_rho_sq, jnp.zeros(1)])
    ) / sigma**2
    off_diag = -(rho / one_minus_rho_sq) / sigma**2
    return diag, off_diag
