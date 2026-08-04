import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

# Stationary kernels with length scale ell (the paper's l) and amplitude
# sigma, selected by name; the legacy Matern smoothness values nu in
# {0.5, 1.5, 2.5, inf} are accepted as aliases. The integrated kernels
# K_tilde(t_i, t_j) = int_0^{t_i} K(t, t_j) dt implement the integrated RKHS
# representation x(t) = x_0 + K_tilde alpha; all four have closed forms.

KERNEL_NAMES = ("matern12", "matern32", "matern52", "rbf")


def rkhs_norm_squared(coefficients, gram):
    """Return the RKHS coefficient quadratic form along the last axis."""
    coefficients = jnp.asarray(coefficients)
    gram = jnp.asarray(gram, dtype=coefficients.dtype)
    return jnp.einsum(
        "...i,ij,...j->...",
        coefficients,
        gram,
        coefficients,
    )


def normalize_kernel(kernel):
    if kernel in {"rbf", "gaussian", "squared_exponential", "se", math.inf}:
        return "rbf"
    if kernel in {"matern12", "matern_1_2", "matern0.5", "matern_0.5", 0.5}:
        return "matern12"
    if kernel in {"matern32", "matern_3_2", "matern1.5", "matern_1.5", 1.5}:
        return "matern32"
    if kernel in {"matern52", "matern_5_2", "matern2.5", "matern_2.5", 2.5}:
        return "matern52"
    raise ValueError(
        f"unknown kernel: {kernel!r} (use one of {KERNEL_NAMES} or "
        "nu in {0.5, 1.5, 2.5, inf})"
    )


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


def rbf_state_features(q, centers, sigma, ell):
    difference = q[..., None] - centers
    return sigma**2 * jnp.exp(-(difference**2) / (2 * ell**2))


def rbf_state_gram(centers, sigma, ell):
    return rbf_state_features(centers[:, None], centers, sigma, ell).squeeze(1)


# Vector-state kernels: ARD is the product of per-dimension 1-D correlations
# with one amplitude, additive their sum with per-dimension amplitudes. The
# Matern families keep exponential (rather than Gaussian) tails, so they decay
# to the mean far more slowly off the center support; matern12 is excluded
# because its kink sits exactly at the centers the Euler residual
# differentiates through.
MATERN_STATE_FAMILIES = ("matern32", "matern52")


def matern_correlation(name, scaled_abs):
    x = MATERN_DECAY[name] * scaled_abs
    poly = {"matern32": 1.0 + x, "matern52": 1.0 + x + x**2 / 3.0}[name]
    return poly * jnp.exp(-x)


def ard_matern_state_features(name, x, centers, sigma, lengthscales):
    scaled = jnp.abs(x[..., None, :] - centers) / lengthscales
    return sigma**2 * jnp.prod(matern_correlation(name, scaled), axis=-1)


def ard_matern_state_gram(name, centers, sigma, lengthscales):
    return ard_matern_state_features(name, centers, centers, sigma, lengthscales)


def additive_matern_state_features(name, x, centers, sigmas, lengthscales):
    scaled = jnp.abs(x[..., None, :] - centers) / lengthscales
    return jnp.sum(sigmas**2 * matern_correlation(name, scaled), axis=-1)


def additive_matern_state_gram(name, centers, sigmas, lengthscales):
    return additive_matern_state_features(name, centers, centers, sigmas, lengthscales)


def ard_rbf_state_features(x, centers, sigma, lengthscales):
    scaled = (x[..., None, :] - centers) / lengthscales
    return sigma**2 * jnp.exp(-0.5 * jnp.sum(scaled**2, axis=-1))


def ard_rbf_state_gram(centers, sigma, lengthscales):
    return ard_rbf_state_features(centers, centers, sigma, lengthscales)


def additive_rbf_state_features(x, centers, sigmas, lengthscales):
    scaled = (x[..., None, :] - centers) / lengthscales
    return jnp.sum(sigmas**2 * jnp.exp(-0.5 * scaled**2), axis=-1)


def additive_rbf_state_gram(centers, sigmas, lengthscales):
    return additive_rbf_state_features(centers, centers, sigmas, lengthscales)


# Each half-integer Matern kernel is sigma^2 pi(a d) e^{-a d} with pi a
# degree-p polynomial and a the decay rate below; its antiderivative is
# h(d) = int_0^d k = (sigma^2/a) [c - e^{-a d} eta(a d)] with eta a degree-p
# polynomial satisfying eta(0) = c (so h(0) = 0). Coefficients are in powers
# of (a d).
MATERN_DECAY = {
    "matern12": 1.0,
    "matern32": math.sqrt(3.0),
    "matern52": math.sqrt(5.0),
}
MATERN_ETA = {
    "matern12": (1.0,),
    "matern32": (2.0, 1.0),
    "matern52": (8.0 / 3.0, 5.0 / 3.0, 1.0 / 3.0),
}


def matern_antiderivative(name, d, sigma, ell):
    a = MATERN_DECAY[name] / ell
    eta = MATERN_ETA[name]
    x = a * d
    eta_x = sum(coefficient * x**k for k, coefficient in enumerate(eta))
    return (sigma**2 / a) * (eta[0] - jnp.exp(-x) * eta_x)


def integrated_matern_kernel_0p5(t_i, t_j, sigma, ell):
    return integrated_matern_kernel("matern12", t_i, t_j, sigma, ell)


def integrated_matern_kernel_1p5(t_i, t_j, sigma, ell):
    return integrated_matern_kernel("matern32", t_i, t_j, sigma, ell)


def integrated_matern_kernel_2p5(t_i, t_j, sigma, ell):
    return integrated_matern_kernel("matern52", t_i, t_j, sigma, ell)


def integrated_matern_kernel(name, t_i, t_j, sigma, ell):
    # int_0^{t_i} k(|s - t_j|) ds splits at s = t_j: h(t_j) + h(t_i - t_j)
    # when t_i >= t_j, and h(t_j) - h(t_j - t_i) otherwise. Assumes
    # t_i, t_j >= 0 (the split point must lie at or beyond the lower limit).
    s = t_i - t_j
    h_t_j = matern_antiderivative(name, t_j, sigma, ell)
    h_gap = matern_antiderivative(name, jnp.abs(s), sigma, ell)
    return h_t_j + jnp.where(s < 0, -h_gap, h_gap)


def integrated_gaussian_kernel(t_i, t_j, sigma, ell):
    # int_0^{t_i} exp(-(t - t_j)^2 / (2 ell^2)) dt in closed form via erf;
    # the odd symmetry of erf makes the t_i = 0 value exactly zero.
    root_2_ell = jnp.sqrt(2.0) * ell
    scale = sigma**2 * ell * jnp.sqrt(jnp.pi / 2.0)
    return scale * (erf((t_i - t_j) / root_2_ell) + erf(t_j / root_2_ell))


# The kernel choice is static: the pair is selected at trace time. Accepts a
# name from KERNEL_NAMES or a legacy nu float.
@partial(jax.jit, static_argnums=(2,))
def integrated_kernel_matrices(t_i, t_j, kernel, sigma, ell):
    name = normalize_kernel(kernel)
    kernel_fn, integrated_fn = {
        "matern12": (matern_kernel_0p5, integrated_matern_kernel_0p5),
        "matern32": (matern_kernel_1p5, integrated_matern_kernel_1p5),
        "matern52": (matern_kernel_2p5, integrated_matern_kernel_2p5),
        "rbf": (gaussian_kernel, integrated_gaussian_kernel),
    }[name]
    K = jax.vmap(
        jax.vmap(kernel_fn, in_axes=(None, 0, None, None)),
        in_axes=(0, None, None, None),
    )(t_i, t_j, sigma, ell)
    K_tilde = jax.vmap(
        jax.vmap(integrated_fn, in_axes=(None, 0, None, None)),
        in_axes=(0, None, None, None),
    )(t_i, t_j, sigma, ell)
    return K, K_tilde
