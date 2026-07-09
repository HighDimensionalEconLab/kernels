import dataclasses
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import config
from jax.scipy.special import erf
from tinygp.kernels import quasisep

# Everything downstream assumes float64 Gram matrices; enabling x64 here (the
# lowest-level module) keeps consumers from touching float32 kernels before
# flipping the flag.
config.update("jax_enable_x64", True)

# Stationary kernels with length scale ell (the paper's l) and amplitude
# sigma, selected by name; the legacy Matern smoothness values nu in
# {0.5, 1.5, 2.5, inf} are accepted as aliases. The integrated kernels
# K_tilde(t_i, t_j) = int_0^{t_i} K(t, t_j) dt implement the integrated RKHS
# representation x(t) = x_0 + K_tilde alpha; all four have closed forms.

KERNEL_NAMES = ("matern12", "matern32", "matern52", "rbf")


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


# Each half-integer Matern kernel is sigma^2 pi(a d) e^{-a d} with pi a
# degree-p polynomial and a the decay rate below; its antiderivative is
# h(d) = int_0^d k = (sigma^2/a) [c - e^{-a d} eta(a d)] with eta a degree-p
# polynomial satisfying eta(0) = c (so h(0) = 0). Coefficients are in powers
# of (a d).
MATERN_ORDER = {"matern12": 0, "matern32": 1, "matern52": 2}
MATERN_DECAY = {
    "matern12": 1.0,
    "matern32": math.sqrt(3.0),
    "matern52": math.sqrt(5.0),
}
MATERN_PI = {
    "matern12": (1.0,),
    "matern32": (1.0, 1.0),
    "matern52": (1.0, 1.0, 1.0 / 3.0),
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


# Matrix-free O(n) matvecs for the half-integer Matern kernels and their
# integrated kernels on a sorted strictly increasing grid t. Both K v and
# K_tilde v reduce to the exponential moments
#   M^(k)_i = sum_j (a |t_i - t_j|)^k e^{-a |t_i - t_j|} v_j,  k = 0..p,
# over j < i (forward) and j > i (backward), which obey the affine recurrence
#   M^(k)_{i+1} = e [sum_m C(k, m) delta^{k-m} M^(m)_i + delta^k v_i]
# with delta = a dt_i and e = e^{-delta} -- a (p+1)-dimensional carry per
# step, so the classic Rybicki-Press scalar recurrence is the p = 0 case.
# The remaining K_tilde pieces are prefix/suffix sums plus the row-independent
# scalar sum_j h(t_j) v_j, since K_tilde(t_i, t_j) = h(t_j) +/- h(|t_i - t_j|).
def matern_exp_moments(t, decay_rate, v, order, parallel=None):
    # parallel=None chooses by backend: off-CPU a sequential scan pays a
    # kernel launch per step (~1000x slower at n=1e5 on GPU), so use the
    # associative O(log n)-depth form; on CPU the tight sequential loop wins.
    # The affine maps M -> A M + b compose associatively:
    # (A_1, b_1) then (A_2, b_2) is (A_2 A_1, A_2 b_1 + b_2).
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    size = order + 1
    delta = decay_rate * jnp.diff(t)
    e = jnp.exp(-delta)
    powers = jnp.stack([delta**k for k in range(size)], axis=-1)
    update = jnp.stack(
        [
            jnp.stack(
                [
                    math.comb(k, m) * delta ** (k - m)
                    if m <= k
                    else jnp.zeros_like(delta)
                    for m in range(size)
                ],
                axis=-1,
            )
            for k in range(size)
        ],
        axis=-2,
    )
    update = e[:, None, None] * update
    forward_input = e[:, None] * powers * v[:-1, None]
    backward_input = e[:, None] * powers * v[1:, None]
    edge = jnp.zeros((1, size), dtype=v.dtype)

    if parallel:

        def combine(earlier, later):
            update_1, input_1 = earlier
            update_2, input_2 = later
            return (
                update_2 @ update_1,
                jnp.einsum("...ij,...j->...i", update_2, input_1) + input_2,
            )

        _, lower_rest = jax.lax.associative_scan(combine, (update, forward_input))
        lower = jnp.concatenate([edge, lower_rest])
        _, upper_head = jax.lax.associative_scan(
            combine, (update, backward_input), reverse=True
        )
        upper = jnp.concatenate([upper_head, edge])
        return lower, upper

    def step(carry, inputs):
        update_i, input_i = inputs
        out = update_i @ carry + input_i
        return out, out

    zero = jnp.zeros((size,), dtype=v.dtype)
    _, lower_rest = jax.lax.scan(step, zero, (update, forward_input))
    lower = jnp.concatenate([edge, lower_rest])
    _, upper_head = jax.lax.scan(step, zero, (update, backward_input), reverse=True)
    upper = jnp.concatenate([upper_head, edge])
    return lower, upper


# Both matvecs from one moments pass: the scans dominate the matrix-free
# residual, and each JVP/VJP inside the solver's CG loop replays them, so
# sharing halves the hot-path work. t must be sorted strictly increasing --
# not validated (t may be traced); unsorted input silently returns garbage,
# so callers validate before entering jit.
def matern_kernel_matvec_pair(kernel, t, v, sigma, ell, parallel=None):
    name = normalize_kernel(kernel)
    if name not in MATERN_ORDER:
        raise ValueError(f"matvecs require a Matern kernel, got {name!r}")
    order = MATERN_ORDER[name]
    a = MATERN_DECAY[name] / ell
    pi = MATERN_PI[name]
    eta = MATERN_ETA[name]
    lower, upper = matern_exp_moments(t, a, v, order, parallel=parallel)

    both = lower + upper
    k_v = sigma**2 * (
        v + sum(coefficient * both[:, k] for k, coefficient in enumerate(pi))
    )

    # (K_tilde v)_i = sum_j h(t_j) v_j
    #   + (sigma^2/a) [c (P_i - Q_i) - eta-moments_{j<=i} + eta-moments_{j>i}]
    # with P/Q the inclusive-prefix/strict-suffix sums of v and c = eta(0).
    c = eta[0]
    total = matern_antiderivative(name, t, sigma, ell) @ v
    prefix = jnp.cumsum(v)
    suffix = jnp.cumsum(v[::-1])[::-1] - v
    eta_lower = (
        sum(coefficient * lower[:, k] for k, coefficient in enumerate(eta)) + c * v
    )
    eta_upper = sum(coefficient * upper[:, k] for k, coefficient in enumerate(eta))
    k_tilde_v = total + (sigma**2 / a) * (c * (prefix - suffix) - eta_lower + eta_upper)
    return k_v, k_tilde_v


# Closed-form tridiagonal precision of the Matern-1/2 Gram on a sorted strictly
# increasing grid (not validated; callers validate -- duplicates divide by
# zero, unsorted grids give an indefinite non-precision). With
# rho_i = e^{-dt_i/ell},
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


# Exact O(n) solves with the Matern-3/2 and 5/2 Gram matrices through their
# quasiseparable (celerite/state-space) structure, via tinygp; the matern12
# inverse is exactly tridiagonal (above) and does not need this. The grid must
# be sorted strictly increasing (tinygp silently returns garbage otherwise --
# callers validate). jitter is an absolute nugget added to the diagonal: the
# nugget-free Gram is numerically singular under grid refinement (cond ~ 1e21
# for matern52 at n = 5000), a property of the matrix, not the solver.
def matern_qsm_solve(kernel, t, sigma, ell, jitter):
    name = normalize_kernel(kernel)
    quasisep_kernel = {
        "matern32": quasisep.Matern32,
        "matern52": quasisep.Matern52,
    }[name](scale=ell, sigma=sigma)
    qsm = quasisep_kernel.to_symm_qsm(t)
    qsm = dataclasses.replace(
        qsm, diag=dataclasses.replace(qsm.diag, d=qsm.diag.d + jitter)
    )
    factor = qsm.cholesky()
    factor_transpose = factor.transpose()

    def solve(v):
        return factor_transpose.solve(factor.solve(v))

    return solve
