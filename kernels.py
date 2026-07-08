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
