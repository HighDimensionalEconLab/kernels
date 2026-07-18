"""Time-block structural preconditioner for the multicountry trade-growth dual.

The undamped dual operator ``D = J M^{-1} J'`` of the kernel-collocation LM solve
(see ``multicountry_trade_growth_comparative_statics``/``..._kernel_nlls``) has, with
the shifted-RKHS metric ``M = blockdiag((K+eps I) x 4N, eps I on algebraic)``,
the exact decomposition

    D(lambda) = D_dyn + (1/eps) blockdiag_time(L_t L_t') + (1/eps) U U' + lambda I

* ``D_dyn`` -- the 4N dynamic-coefficient contribution, dense across time through
  the kernel quadratic forms ``Q1=K_tilde(K+eps I)^{-1}K_tilde'``,
  ``Q2=K_tilde(K+eps I)^{-1}K'``, ``Q3=K(K+eps I)^{-1}K'``;
* ``(1/eps) L_t L_t'`` -- the time-local algebraic block (static residuals + rate +
  numeraire), amplified 1e7x;
* ``(1/eps) U U'`` -- the rank-2N global q/marginal-wealth-offset coupling.

The "full C3" preconditioner keeps the EXACT time-diagonal blocks
``D_tt = D_dyn[t,t] + (1/eps)L_t L_t' + (1/eps)U_t U_t' + lambda I`` and the exact
off-time-diagonal global coupling ``(1/eps) U_t U_t''``, dropping only the
off-time-diagonal dynamic tail ``D_dyn[t,t']``. Written as a block-diagonal-plus-
low-rank system it is

    C3 = A + (1/eps) U U',   A = blockdiag_time(D_tt - (1/eps) U_t U_t'),

so ``A_tt = D_dyn[t,t] + (1/eps)L_t L_t' + lambda I`` and the Woodbury identity
inverts it from the per-time-block factorization plus a 2N x 2N capacitance solve.

The time-diagonal blocks are assembled ANALYTICALLY (no operator probing) via a
vmapped ``jax.jacfwd`` of the per-time residual map over its consumed path
values/derivatives and time-local algebraic unknowns. ``lambda`` is kept analytic
by eigendecomposing the undamped blocks ``A_tt(0)`` once and applying
``(Lambda + lambda)^{-1}``, so one build serves ``dual_preconditioner(v, lambda)``
for any damping and ``implicit_preconditioner(v)`` at ``lambda = 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from kernels import integrated_kernel_matrices, normalize_kernel
from multicountry_trade_growth import (
    DYNAMIC_RESIDUAL_NAMES,
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    consumption_basket_log_price,
    multicountry_trade_path_values,
    static_trade_equilibrium,
)

NUM_STATIC_BLOCKS = len(STATIC_BLOCK_NAMES)


def trade_growth_reference_state(parameters, k_0, a_0, dtype=jnp.float64):
    """Reconstruct the reference levels/scales the driver linearizes around.

    Mirrors ``multicountry_trade_growth_comparative_statics.build_...`` exactly:
    the normalized static reference sets ``q_reference``/``marginal_wealth_reference``
    and ``output_scale`` that enter the path-value transforms. Uses the host
    ``static_trade_equilibrium`` solve, so this runs once at build time (not traced).
    """

    # static_trade_equilibrium is a host scipy least-squares solve: force concrete
    # evaluation so this stays valid when a caller (the testbed) wraps prepare in jit.
    with jax.ensure_compile_time_eval():
        k_0_array = jnp.asarray(k_0, dtype=dtype)
        a_0_array = jnp.asarray(a_0, dtype=dtype)
        populations = jnp.asarray(parameters.populations, dtype=dtype)
        num_countries = parameters.num_countries
        base_q = jnp.full(
            num_countries, 1.0 / parameters.installation_elasticity, dtype=dtype
        )
        base_m = jnp.ones(num_countries, dtype=dtype)
        base_static, _ = static_trade_equilibrium(k_0_array, base_q, base_m, parameters)
        basket_price = jnp.exp(
            consumption_basket_log_price(base_static["log_static"], parameters)
        )
        price_scale = 1.0 / basket_price
        q_reference = price_scale * base_q
        marginal_wealth_reference = base_m / price_scale
        static_reference, _ = static_trade_equilibrium(
            k_0_array,
            q_reference,
            marginal_wealth_reference,
            parameters,
            log_initial=base_static["log_static"] + jnp.log(price_scale),
        )
        output_scale = static_reference["world_gdp"] / jnp.sum(populations)
    return {
        "k_0_array": k_0_array,
        "a_0_array": a_0_array,
        "q_reference": q_reference,
        "marginal_wealth_reference": marginal_wealth_reference,
        "output_scale": output_scale,
        "log_static_reference": static_reference["log_static"],
    }


def residual_time_permutation(num_countries, num_train_points):
    """Native-order -> time-major permutation of the (12N+1) n_t residual rows.

    Identical layout to the testbed's ``residual_time_permutations``: per time,
    [4N dynamic (capital, assets, tobin_q, marginal_wealth) | 8N static | 1 numeraire].
    """

    n = num_countries
    nt = num_train_points
    dynamic_size = 4 * n * nt
    static_size = NUM_STATIC_BLOCKS * n * nt
    time_major = []
    for t in range(nt):
        for dynamic_index in range(4):
            start = dynamic_index * nt * n + t * n
            time_major.extend(range(start, start + n))
        static_start = dynamic_size + t * NUM_STATIC_BLOCKS * n
        time_major.extend(range(static_start, static_start + NUM_STATIC_BLOCKS * n))
        time_major.append(dynamic_size + static_size + t)
    time_major = jnp.asarray(time_major, dtype=jnp.int32)
    native = jnp.argsort(time_major)
    return time_major, native


def assemble_time_diagonal_blocks(
    theta,
    K,
    K_tilde,
    reference_state,
    parameters,
    eps,
):
    """Analytic (12N+1) time-diagonal dual blocks + the rank-2N offset columns.

    Returns ``(A0, D_tt, U_blocks, Q1, Q2, Q3)`` where ``A0`` are the undamped
    reduced blocks ``D_dyn[t,t] + (1/eps)L_t L_t'`` (the Woodbury ``A``), ``D_tt``
    the exact time-diagonal blocks of the full dual (``A0 + (1/eps)U_t U_t'``), and
    ``U_blocks`` the per-time rows of the global offset columns (== the q/m level
    Jacobians). All time-batched: shapes ``(n_t, B, B)`` and ``(n_t, B, 2N)``.
    """

    num_countries = parameters.num_countries
    n = num_countries
    dtype = K.dtype
    alpha_k, alpha_a, alpha_q, alpha_m, algebraic = theta
    n_train = K.shape[0]
    block_size = 12 * n + 1

    q_offsets = algebraic[n_train : n_train + n]
    marginal_wealth_offsets = algebraic[n_train + n : n_train + 2 * n]
    rate_coordinates = algebraic[:n_train]
    log_static = algebraic[n_train + 2 * n :].reshape(n_train, NUM_STATIC_BLOCKS, n)

    def kernel_levels(alpha):
        return K_tilde @ alpha.T, K @ alpha.T

    yk, ypk = kernel_levels(alpha_k)
    ya, ypa = kernel_levels(alpha_a)
    yq, ypq = kernel_levels(alpha_q)
    ym, ypm = kernel_levels(alpha_m)

    # w_t = [ y (4N) | y' (4N) | log_static (8N) | rate (1) ], the inputs the
    # time-local residual consumes; jacfwd over w_t yields C=dr/dy (4N cols),
    # D=dr/dy' (4N cols), L=dr/d(log_static, rate) (8N+1 cols) in one shot.
    w = jnp.concatenate(
        [
            yk,
            ya,
            yq,
            ym,
            ypk,
            ypa,
            ypq,
            ypm,
            log_static.reshape(n_train, NUM_STATIC_BLOCKS * n),
            rate_coordinates[:, None],
        ],
        axis=1,
    )

    k_0_array = reference_state["k_0_array"]
    a_0_array = reference_state["a_0_array"]
    q_reference = reference_state["q_reference"]
    marginal_wealth_reference = reference_state["marginal_wealth_reference"]
    output_scale = reference_state["output_scale"]
    rho = parameters.rho

    dynamic_scale = 1.0 / math.sqrt(4 * n * n_train)
    static_scale = 1.0 / math.sqrt(NUM_STATIC_BLOCKS * n * n_train)
    numeraire_scale = 1.0 / math.sqrt(n_train)

    def per_time_residual(w_t):
        yk_t = w_t[0:n]
        ya_t = w_t[n : 2 * n]
        yq_t = w_t[2 * n : 3 * n]
        ym_t = w_t[3 * n : 4 * n]
        ypk_t = w_t[4 * n : 5 * n]
        ypa_t = w_t[5 * n : 6 * n]
        ypq_t = w_t[6 * n : 7 * n]
        ypm_t = w_t[7 * n : 8 * n]
        log_static_t = w_t[8 * n : 16 * n].reshape(NUM_STATIC_BLOCKS, n)
        rate_t = w_t[16 * n]

        k = jnp.exp(jnp.log(k_0_array) + yk_t)
        a = a_0_array + output_scale * ya_t
        tobin_q = jnp.exp(jnp.log(q_reference) + q_offsets + yq_t)
        marginal_wealth = jnp.exp(
            jnp.log(marginal_wealth_reference) + marginal_wealth_offsets + ym_t
        )
        interest_rate = rho * (1.0 + rate_t)
        values = multicountry_trade_path_values(
            k=k[None, :],
            a=a[None, :],
            tobin_q=tobin_q[None, :],
            marginal_wealth=marginal_wealth[None, :],
            interest_rate=interest_rate[None],
            log_static=log_static_t[None, :, :],
            dlog_k_dt=ypk_t[None, :],
            da_dt=output_scale * ypa_t[None, :],
            dlog_q_dt=ypq_t[None, :],
            dlog_marginal_wealth_dt=ypm_t[None, :],
            parameters=parameters,
        )
        dynamic = jnp.concatenate(
            [values[name].reshape(-1) for name in DYNAMIC_RESIDUAL_NAMES]
        )
        static = values["static_residuals"].reshape(-1)
        numeraire = consumption_basket_log_price(
            log_static_t[None, :, :], parameters
        ).reshape(-1)
        return jnp.concatenate(
            [
                dynamic * dynamic_scale,
                static * static_scale,
                numeraire * numeraire_scale,
            ]
        )

    jacobian = jax.vmap(jax.jacfwd(per_time_residual))(w)
    dynamic_c = jacobian[:, :, 0 : 4 * n]
    dynamic_d = jacobian[:, :, 4 * n : 8 * n]
    algebraic_l = jacobian[:, :, 8 * n : 16 * n + 1]
    # q and m level columns (indices 2N..4N of C) are exactly the global offsets.
    offset_u = jacobian[:, :, 2 * n : 4 * n]

    kernel_ridge = K + eps * jnp.eye(n_train, dtype=dtype)
    kernel_inverse = jsp_linalg.solve(kernel_ridge, jnp.eye(n_train, dtype=dtype))
    kernel_inverse = 0.5 * (kernel_inverse + kernel_inverse.T)
    ktilde_kinv = K_tilde @ kernel_inverse
    k_kinv = K @ kernel_inverse
    q1 = jnp.sum(ktilde_kinv * K_tilde, axis=1)
    q2 = jnp.sum(ktilde_kinv * K, axis=1)
    q3 = jnp.sum(k_kinv * K, axis=1)

    cc = jnp.einsum("tik,tjk->tij", dynamic_c, dynamic_c)
    cd = jnp.einsum("tik,tjk->tij", dynamic_c, dynamic_d)
    dd = jnp.einsum("tik,tjk->tij", dynamic_d, dynamic_d)
    ll = jnp.einsum("tik,tjk->tij", algebraic_l, algebraic_l)
    uu = jnp.einsum("tik,tjk->tij", offset_u, offset_u)

    inv_eps = 1.0 / eps
    dynamic_block = (
        q1[:, None, None] * cc
        + q2[:, None, None] * (cd + jnp.swapaxes(cd, 1, 2))
        + q3[:, None, None] * dd
    )
    reduced_block = dynamic_block + inv_eps * ll
    reduced_block = 0.5 * (reduced_block + jnp.swapaxes(reduced_block, 1, 2))
    full_block = reduced_block + inv_eps * uu
    full_block = 0.5 * (full_block + jnp.swapaxes(full_block, 1, 2))
    assert reduced_block.shape[1] == block_size
    return reduced_block, full_block, offset_u, q1, q2, q3


def capacitance_in_eigenbasis(offset_u_eig, inverse_eigenvalues, eps):
    """``S(lambda) = eps I + U' A(lambda)^{-1} U`` from the eigenbasis offset columns.

    ``offset_u_eig = V' U`` (per block) and ``inverse_eigenvalues = 1/(Lambda+lambda)``
    are all that is needed -- no per-block ``A^{-1} U`` matmul with ``V``.
    """

    rank = offset_u_eig.shape[2]
    weighted = offset_u_eig * inverse_eigenvalues[:, :, None]
    capacitance = eps * jnp.eye(rank, dtype=offset_u_eig.dtype) + jnp.einsum(
        "tbk,tbl->kl", offset_u_eig, weighted
    )
    return 0.5 * (capacitance + capacitance.T)


def woodbury_apply_eigenbasis(
    eigenvectors, offset_u_eig, inverse_eigenvalues, capacitance_chol, v_blocks
):
    """Apply ``(A + (1/eps) U U')^{-1}`` block-wise, fully in the eigenbasis.

    ``coeff_y = inv * (V'v - (V'U) S^{-1} (V'U)' inv V'v)``; two ``B x B`` matvecs
    (``V'v`` and the final ``V coeff_y``) plus the rank-2N capacitance solve.
    """

    coefficients = jnp.einsum("tab,ta->tb", eigenvectors, v_blocks)
    offset_residual = jnp.einsum(
        "tbk,tb->k", offset_u_eig, inverse_eigenvalues * coefficients
    )
    correction = jsp_linalg.cho_solve((capacitance_chol, True), offset_residual)
    corrected = coefficients - jnp.einsum("tbk,k->tb", offset_u_eig, correction)
    return jnp.einsum("tab,tb->ta", eigenvectors, inverse_eigenvalues * corrected)


@dataclass(frozen=True)
class TradeGrowthTimeBlockPreconditioner:
    """Full-C3 dual preconditioner exposing both nlls_gram hooks.

    ``dual_preconditioner(v, damping)`` and ``implicit_preconditioner(v)`` apply the
    exact ``(A(damping) + (1/eps) U U')^{-1}`` via the frozen block eigendecomposition
    plus a 2N x 2N Woodbury capacitance solve, entirely in the block eigenbasis
    (``offset_u_eig = V' U`` precomputed). All fields are arrays, so the object is a
    jit-safe pytree; ``num_countries``/``num_train_points`` come from array shapes.
    """

    eigenvectors: jax.Array
    eigenvalues: jax.Array
    offset_u_eig: jax.Array
    time_permutation: jax.Array
    native_permutation: jax.Array
    eps: float
    implicit_damping: float

    @property
    def block_size(self):
        return self.eigenvectors.shape[1]

    @property
    def num_train_points(self):
        return self.eigenvectors.shape[0]

    def dual_preconditioner(self, v, damping):
        eps = jnp.asarray(self.eps, dtype=v.dtype)
        n_t = self.num_train_points
        block = self.block_size
        inverse_eigenvalues = 1.0 / (self.eigenvalues + damping)
        capacitance = capacitance_in_eigenbasis(
            self.offset_u_eig, inverse_eigenvalues, eps
        )
        capacitance_chol = jnp.linalg.cholesky(capacitance)
        v_blocks = v[self.time_permutation].reshape(n_t, block)
        y = woodbury_apply_eigenbasis(
            self.eigenvectors,
            self.offset_u_eig,
            inverse_eigenvalues,
            capacitance_chol,
            v_blocks,
        )
        return y.reshape(-1)[self.native_permutation]

    def implicit_preconditioner(self, v):
        return self.dual_preconditioner(
            v, jnp.asarray(self.implicit_damping, dtype=v.dtype)
        )


jax.tree_util.register_dataclass(
    TradeGrowthTimeBlockPreconditioner,
    data_fields=[
        "eigenvectors",
        "eigenvalues",
        "offset_u_eig",
        "time_permutation",
        "native_permutation",
    ],
    meta_fields=["eps", "implicit_damping"],
)


def trade_growth_preconditioner_apply(state, v, damping):
    """``nlls_gram.PreconditionerFactory`` apply: ``state`` is the preconditioner
    (a ``TradeGrowthTimeBlockPreconditioner``); dispatches to its Woodbury apply.

    Module-level so the ``(prepare, apply)`` factory pair is value-hashable and
    shares one compiled solve loop.
    """

    return state.dual_preconditioner(v, damping)


def build_trade_growth_time_block_preconditioner(
    theta,
    K,
    K_tilde,
    reference_state,
    parameters,
    eps,
    time_permutation,
    native_permutation,
    implicit_damping=1e-10,
    return_blocks=False,
):
    """Assemble the full-C3 time-block preconditioner at linearization point ``theta``.

    ``K``/``K_tilde`` are the (derivative, integrated) kernel matrices on the train
    grid; ``reference_state`` is ``trade_growth_reference_state(...)``. ``eps`` is the
    metric ridge (``metric_eps``). ``time_permutation``/``native_permutation`` map
    native residual order to the per-time blocks. Returns the preconditioner; with
    ``return_blocks=True`` also returns the exact ``D_tt`` for the validation gate.
    """

    reduced_block, full_block, offset_u, _, _, _ = assemble_time_diagonal_blocks(
        theta, K, K_tilde, reference_state, parameters, eps
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(reduced_block)
    offset_u_eig = jnp.einsum("tab,tak->tbk", eigenvectors, offset_u)
    preconditioner = TradeGrowthTimeBlockPreconditioner(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        offset_u_eig=offset_u_eig,
        time_permutation=jnp.asarray(time_permutation, dtype=jnp.int32),
        native_permutation=jnp.asarray(native_permutation, dtype=jnp.int32),
        eps=float(eps),
        implicit_damping=float(implicit_damping),
    )
    if return_blocks:
        return preconditioner, full_block
    return preconditioner


def preconditioner_pieces_from_case(case, kernel="matern52", sigma=1.0, ell=10.0):
    """Rebuild the assembly inputs from a testbed ``LinearizedLLSCase``."""

    train_data = jnp.linspace(
        0.0, case.settings.train_T, case.settings.train_points, dtype=jnp.float64
    )
    return build_trade_growth_preconditioner_pieces(
        parameters=case.parameters,
        k_0=case.k_0,
        a_0=case.a_0,
        train_data=train_data,
        metric_eps=case.settings.metric_eps,
        theta=case.theta,
        kernel=kernel,
        sigma=sigma,
        ell=ell,
    )


@dataclass(frozen=True)
class TradeGrowthTimeBlockAlgorithm:
    """Testbed ``prepare``/``solve`` plugin wrapping the full-C3 preconditioner.

    ``prepare`` assembles the block eigendecomposition and specializes the Woodbury
    pieces (``A^{-1}U`` and the 2N x 2N capacitance factor) for ``case.damping`` so
    the per-CG-iteration apply is one block solve plus a rank-2N correction. The
    reference/kernel build (host ``static_trade_equilibrium``) runs at trace time on
    the argument-free ``prepare`` closure, so it constant-folds under the testbed jit.
    """

    tol: float = 1e-10
    atol: float = 0.0
    maxiter: int = 4000

    def prepare(self, case, settings):
        pieces = preconditioner_pieces_from_case(case)
        reduced_block, _, offset_u, _, _, _ = assemble_time_diagonal_blocks(
            case.theta,
            pieces["K"],
            pieces["K_tilde"],
            pieces["reference_state"],
            case.parameters,
            pieces["eps"],
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(reduced_block)
        offset_u_eig = jnp.einsum("tab,tak->tbk", eigenvectors, offset_u)
        damping = jnp.asarray(case.damping, dtype=reduced_block.dtype)
        eps = jnp.asarray(pieces["eps"], dtype=reduced_block.dtype)
        # fixed damping within a solve: specialize the lambda-dependent pieces once.
        inverse_eigenvalues = 1.0 / (eigenvalues + damping)
        capacitance = capacitance_in_eigenbasis(offset_u_eig, inverse_eigenvalues, eps)
        return {
            "eigenvectors": eigenvectors,
            "offset_u_eig": offset_u_eig,
            "inverse_eigenvalues": inverse_eigenvalues,
            "capacitance_chol": jnp.linalg.cholesky(capacitance),
            "time_permutation": pieces["time_permutation"],
            "native_permutation": pieces["native_permutation"],
        }

    def solve(self, case, prepared, rhs, settings):
        eigenvectors = prepared["eigenvectors"]
        offset_u_eig = prepared["offset_u_eig"]
        inverse_eigenvalues = prepared["inverse_eigenvalues"]
        capacitance_chol = prepared["capacitance_chol"]
        time_permutation = prepared["time_permutation"]
        native_permutation = prepared["native_permutation"]
        n_t, block = eigenvectors.shape[0], eigenvectors.shape[1]

        def msolve(r):
            r_blocks = r[time_permutation].reshape(n_t, block)
            y = woodbury_apply_eigenbasis(
                eigenvectors,
                offset_u_eig,
                inverse_eigenvalues,
                capacitance_chol,
                r_blocks,
            )
            return y.reshape(-1)[native_permutation]

        matvec = case.dual_matvec_with(case.damping)
        rhs_norm = jnp.linalg.norm(rhs)
        tolerance = jnp.maximum(
            jnp.asarray(self.atol, dtype=rhs.dtype),
            jnp.asarray(self.tol, dtype=rhs.dtype) * rhs_norm,
        )
        z0 = msolve(rhs)
        state = (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros_like(rhs),
            rhs,
            z0,
            rhs @ z0,
            rhs_norm,
            jnp.all(jnp.isfinite(rhs)),
        )

        def keep_going(carry):
            iteration, _, _, _, _, residual_norm, finite = carry
            return (iteration < self.maxiter) & (residual_norm > tolerance) & finite

        def step(carry):
            iteration, x, r, p, rz, _, _ = carry
            Ap = matvec(p)
            pAp = p @ Ap
            valid = jnp.isfinite(pAp) & (pAp > 0.0) & (rz >= 0.0)
            safe = jnp.where(valid, pAp, 1.0)
            alpha = rz / safe
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            z_new = msolve(r_new)
            rz_new = r_new @ z_new
            beta = jnp.where(rz > 0.0, rz_new / jnp.where(rz > 0.0, rz, 1.0), 0.0)
            p_new = z_new + beta * p
            residual_norm = jnp.linalg.norm(r_new)
            finite = (
                valid
                & jnp.all(jnp.isfinite(x_new))
                & jnp.all(jnp.isfinite(r_new))
                & jnp.isfinite(rz_new)
            )
            x_out = jnp.where(finite, x_new, x)
            r_out = jnp.where(finite, r_new, r)
            p_out = jnp.where(finite, p_new, p)
            rz_out = jnp.where(finite, rz_new, rz)
            norm_out = jnp.where(finite, residual_norm, jnp.inf)
            return (iteration + 1, x_out, r_out, p_out, rz_out, norm_out, finite)

        final = jax.lax.while_loop(keep_going, step, state)
        iterations, solution, _, _, _, residual_norm, finite = final
        converged = finite & (residual_norm <= tolerance)
        return solution, iterations, converged


time_block_algorithm = TradeGrowthTimeBlockAlgorithm()


def build_trade_growth_preconditioner_pieces(
    parameters: TradeGrowthParameters,
    k_0,
    a_0,
    train_data,
    metric_eps,
    theta,
    kernel="matern52",
    sigma=1.0,
    ell=10.0,
    dtype=jnp.float64,
):
    """Rebuild the kernel matrices / references / permutations from model inputs.

    Convenience for callers holding only the structural description (the testbed
    case, the driver). Recomputes exactly what the comparative-statics builder uses.
    """

    train_data = jnp.asarray(train_data, dtype=dtype)
    n_train = train_data.shape[0]
    kernel_name = normalize_kernel(kernel)
    K, K_tilde = integrated_kernel_matrices(
        train_data, train_data, kernel_name, sigma, ell
    )
    K = 0.5 * (K + K.T)
    reference_state = trade_growth_reference_state(parameters, k_0, a_0, dtype=dtype)
    time_permutation, native_permutation = residual_time_permutation(
        parameters.num_countries, n_train
    )
    return {
        "K": K,
        "K_tilde": K_tilde,
        "reference_state": reference_state,
        "time_permutation": time_permutation,
        "native_permutation": native_permutation,
        "eps": float(metric_eps),
    }
