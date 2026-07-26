"""Whitened LSMR endgame solve for the multicountry trade-growth LM.

The dual operator ``J M^{-1} J'`` is cond ~1e10-1e15 at the LM endgame (the
normal-equations conditioning-squaring), so a dual/CG solve floors at ~eps*cond and
the truncation error corrupts the underdetermined selection (breaks asset_clearing).
Whitened LSMR works on ``B = J S`` with ``S S' = M^{-1}`` (cond = sqrt(cond(dual))),
reaching an ``eps*cond(B)`` step that fixes the endgame.

``B``'s conditioning (~1e8) still needs a preconditioner. This module builds the
parameter-space right-preconditioner ``R`` with ``R'R ~ B'B + damping I`` by a Schur
elimination of the (block-diagonal, per-time) algebraic theta-blocks into the dense
dynamic (alpha + rank-2N offset) block -- the same per-time ``c/d/L`` autodiff the
dual time-block preconditioner uses, transposed to parameter space. Right-precond LSMR
on ``B R^{-1}`` converges in ~63-73 iters (dense S_dd) at the endgame.

nlls_gram-free by construction: ``S = M^{-1/2}`` is built here from ``(K+eps I)^{-1/2}``
(per dynamic function) and ``eps^{-1/2}`` (algebraic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from extra.trade_growth.model import (
    DYNAMIC_RESIDUAL_NAMES,
    STATIC_BLOCK_NAMES,
    consumption_basket_log_price,
    multicountry_trade_path_values,
)

NUM_STATIC_BLOCKS = len(STATIC_BLOCK_NAMES)


def per_time_jacobian(theta, K, K_tilde, reference_state, parameters, eps):
    """Per-time ``(c, d, L)`` blocks (same map the dual time-block precond uses).

    Returns ``dynamic_c`` (n_t, 12N+1, 4N) = dr_t/dy, ``dynamic_d`` (n_t, 12N+1, 4N)
    = dr_t/dy', ``algebraic_l`` (n_t, 12N+1, 8N+1) = dr_t/d(log_static_t, rate_t).
    The q/m columns of ``dynamic_c`` (indices 2N..4N) are the global offset columns.
    """

    n = parameters.num_countries
    n_train = K.shape[0]
    alpha_k, alpha_a, alpha_q, alpha_m, algebraic = theta
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
        yk_t, ya_t, yq_t, ym_t = (
            w_t[0:n],
            w_t[n : 2 * n],
            w_t[2 * n : 3 * n],
            w_t[3 * n : 4 * n],
        )
        ypk_t, ypa_t = w_t[4 * n : 5 * n], w_t[5 * n : 6 * n]
        ypq_t, ypm_t = w_t[6 * n : 7 * n], w_t[7 * n : 8 * n]
        log_static_t = w_t[8 * n : 16 * n].reshape(NUM_STATIC_BLOCKS, n)
        rate_t = w_t[16 * n]
        values = multicountry_trade_path_values(
            k=jnp.exp(jnp.log(k_0_array) + yk_t)[None, :],
            a=(a_0_array + output_scale * ya_t)[None, :],
            tobin_q=jnp.exp(jnp.log(q_reference) + q_offsets + yq_t)[None, :],
            marginal_wealth=jnp.exp(
                jnp.log(marginal_wealth_reference) + marginal_wealth_offsets + ym_t
            )[None, :],
            interest_rate=(rho * (1.0 + rate_t))[None],
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
    return (
        jacobian[:, :, 0 : 4 * n],
        jacobian[:, :, 4 * n : 8 * n],
        jacobian[:, :, 8 * n : 16 * n + 1],
    )


def whitening_root(K, eps):
    """Per-dynamic-function whitening factor ``S_alpha`` with ``S_alpha S_alpha' =
    (K + eps I)^{-1}``.

    MUST match the metric the LM whitens with: the dynamic block of
    ``repeated_shifted_dense_metric`` uses ``chol(K + eps I)`` and its
    ``inv_sqrt`` is ``L^{-T}``, not the symmetric ``M^{-1/2}``. Both satisfy
    ``S S' = M^{-1}`` and give the same minimum-M-norm LM step, but ``B = J S``
    and this preconditioner must use the same ``S`` -- mixing them corrupts
    ``B'B`` in the rank-2N near-null directions.
    """

    n_train = K.shape[0]
    chol = jnp.linalg.cholesky(K + eps * jnp.eye(n_train, dtype=K.dtype))
    return jsp_linalg.solve_triangular(
        chol.T, jnp.eye(n_train, dtype=K.dtype), lower=False
    )


def theta_permutation(num_countries, n_train):
    """theta-order -> [algebraic(per-time) ; dynamic(alpha + offsets)] reindex.

    Returns ``(alg_index, dyn_index)``: ``alg_index`` (n_t, 8N+1) the per-time
    algebraic blocks [log_static_t (8N), rate_t (1)]; ``dyn_index`` (4N n_t + 2N)
    the alpha functions then the 2N offsets.
    """

    n, nt = num_countries, n_train
    dyn = 4 * n * nt
    rate0 = dyn
    off0 = dyn + nt
    ls0 = dyn + nt + 2 * n
    alg_index = jnp.stack(
        [
            jnp.concatenate(
                [
                    jnp.asarray(
                        [
                            ls0 + t * 8 * n + s * n + c
                            for s in range(8)
                            for c in range(n)
                        ],
                        dtype=jnp.int32,
                    ),
                    jnp.asarray([rate0 + t], dtype=jnp.int32),
                ]
            )
            for t in range(nt)
        ]
    )
    dyn_index = jnp.concatenate(
        [
            jnp.arange(dyn, dtype=jnp.int32),
            jnp.arange(off0, off0 + 2 * n, dtype=jnp.int32),
        ]
    )
    return alg_index, dyn_index


@jax.tree_util.register_dataclass
@dataclass
class WhitenedSchurPreconditioner:
    """Right-preconditioner ``R`` (``R'R ~ B'B + damping I``) for whitened LSMR.

    ``solve(v) = R^{-1} v`` and ``solve_transpose(w) = R^{-T} w`` in parameter (theta)
    space, via the block-Cholesky ``L = [[L_aa, 0],[L_da, L_dd]]`` of the
    algebraic-eliminated Hessian (``R = L'``). Arrays only -> jit-safe pytree.
    """

    l_aa: jax.Array  # (n_t, 8N+1, 8N+1) per-time Cholesky of H_aa
    l_da: jax.Array  # (n_t, dyn, 8N+1) block coupling factor
    l_dd: jax.Array  # (dyn, dyn) Cholesky of the dynamic Schur S_dd
    alg_index: jax.Array  # (n_t, 8N+1)
    dyn_index: jax.Array  # (dyn,)
    inverse_permutation: jax.Array  # (n,) [alg;dyn] flat -> theta order

    def _to_blocks(self, v):
        v_alg = v[self.alg_index]  # (n_t, 8N+1)
        v_dyn = v[self.dyn_index]  # (dyn,)
        return v_alg, v_dyn

    def _from_blocks(self, x_alg, x_dyn):
        flat = jnp.concatenate([x_alg.reshape(-1), x_dyn])
        return flat[self.inverse_permutation]

    def solve(self, v):
        # R^{-1} v = L^{-T} v : solve L' x = v_reordered (upper-triangular)
        v_alg, v_dyn = self._to_blocks(v)
        x_dyn = jsp_linalg.solve_triangular(self.l_dd, v_dyn, lower=True, trans="T")
        coupled = v_alg - jnp.einsum("tdk,d->tk", self.l_da, x_dyn)
        x_alg = jax.vmap(
            lambda la, b: jsp_linalg.solve_triangular(la, b, lower=True, trans="T")
        )(self.l_aa, coupled)
        return self._from_blocks(x_alg, x_dyn)

    def solve_transpose(self, w):
        # R^{-T} w = L^{-1} w : solve L x = w_reordered (lower-triangular)
        w_alg, w_dyn = self._to_blocks(w)
        x_alg = jax.vmap(lambda la, b: jsp_linalg.solve_triangular(la, b, lower=True))(
            self.l_aa, w_alg
        )
        coupled = w_dyn - jnp.einsum("tdk,tk->d", self.l_da, x_alg)
        x_dyn = jsp_linalg.solve_triangular(self.l_dd, coupled, lower=True)
        return self._from_blocks(x_alg, x_dyn)


def build_schur_preconditioner(
    theta,
    K,
    K_tilde,
    reference_state,
    parameters,
    eps,
    damping,
    mode="dense",
    return_blocks=False,
):
    """Assemble the whitened parameter-space Schur ``R`` (see module docstring).

    ``mode="dense"`` factors the dynamic Schur ``S_dd`` densely (63-73 endgame iters);
    ``mode="blockfunc"`` uses per-function-alpha block factors of ``S_dd`` (151-175
    iters, no dense factor). ``damping`` is the LM ``lambda`` (fixed per build).
    """

    n = parameters.num_countries
    nt = K.shape[0]
    dtype = K.dtype
    lam = jnp.asarray(damping, dtype=dtype)
    inv_eps = 1.0 / eps
    inv_sqrt_eps = 1.0 / math.sqrt(eps)

    c, d, ell = per_time_jacobian(theta, K, K_tilde, reference_state, parameters, eps)
    W = whitening_root(K, eps)
    ktw = K_tilde @ W  # (n_t, n_t): row t = W K_tilde[t,:]
    kw = K @ W

    # per-time inner products of the (12N+1)-row jacobian columns
    cc = jnp.einsum("trf,trg->tfg", c, c)  # (n_t, 4N, 4N)
    cd = jnp.einsum("trf,trg->tfg", c, d)
    dd = jnp.einsum("trf,trg->tfg", d, d)
    c_ell = jnp.einsum("trf,trl->tfl", c, ell)  # (n_t, 4N, 8N+1)
    d_ell = jnp.einsum("trf,trl->tfl", d, ell)
    ll = jnp.einsum("trk,trl->tkl", ell, ell)  # (n_t, 8N+1, 8N+1)

    four_n = 4 * n
    dyn_size = four_n * nt + 2 * n
    alg_block = 8 * n + 1
    off = slice(2 * n, 4 * n)  # q,m function columns = offsets

    # ---- dynamic-dynamic H_dd (dense: alpha 4N n_t + offsets 2N) + lam I ----
    gaa = (
        jnp.einsum("tfg,ti,tj->figj", cc, ktw, ktw)
        + jnp.einsum("tfg,ti,tj->figj", cd, ktw, kw)
        + jnp.einsum("tgf,ti,tj->figj", cd, kw, ktw)
        + jnp.einsum("tfg,ti,tj->figj", dd, kw, kw)
    ).reshape(four_n * nt, four_n * nt)
    # alpha-offset (whiten alpha side by W (already in ktw/kw), offset side eps^{-1/2})
    gao = (
        jnp.einsum("tfc,ti->fic", cc[:, :, off], ktw)
        + jnp.einsum("tcf,ti->fic", cd[:, off, :], kw)
    ).reshape(four_n * nt, 2 * n) * inv_sqrt_eps
    goo = inv_eps * jnp.sum(cc[:, off, off], axis=0)  # (2N, 2N)
    h_dd = jnp.zeros((dyn_size, dyn_size), dtype=dtype)
    a_end = four_n * nt
    h_dd = h_dd.at[:a_end, :a_end].set(gaa)
    h_dd = h_dd.at[:a_end, a_end:].set(gao)
    h_dd = h_dd.at[a_end:, :a_end].set(gao.T)
    h_dd = h_dd.at[a_end:, a_end:].set(goo)
    h_dd = 0.5 * (h_dd + h_dd.T) + lam * jnp.eye(dyn_size, dtype=dtype)

    # ---- per-time algebraic H_aa = eps^{-1} L_t'L_t + lam I ----
    h_aa = inv_eps * ll + lam * jnp.eye(alg_block, dtype=dtype)  # (n_t, 8N+1, 8N+1)
    l_aa = jnp.linalg.cholesky(0.5 * (h_aa + jnp.swapaxes(h_aa, 1, 2)))

    # ---- per-time dynamic<->algebraic coupling H_da (dyn x 8N+1) ----
    # alpha rows: eps^{-1/2}(WKt[t,i] c_ell[t,f,l] + WK[t,i] d_ell[t,f,l])
    hda_alpha = inv_sqrt_eps * (
        jnp.einsum("ti,tfl->tfil", ktw, c_ell) + jnp.einsum("ti,tfl->tfil", kw, d_ell)
    ).reshape(nt, four_n * nt, alg_block)
    # offset rows: eps^{-1} c_ell for the q,m functions
    hda_off = inv_eps * c_ell[:, off, :]  # (n_t, 2N, 8N+1)
    h_da = jnp.concatenate([hda_alpha, hda_off], axis=1)  # (n_t, dyn, 8N+1)

    # ---- block-Cholesky: L_da = H_da L_aa^{-T}, S_dd = H_dd - sum_t L_da_t L_da_t' ----
    # L_da = H_da L_aa^{-T}: solve L_aa X = H_da' (lower) -> X = L_aa^{-1} H_da', L_da = X'
    l_da = jax.vmap(
        lambda hda, la: jsp_linalg.solve_triangular(la, hda.T, lower=True).T
    )(h_da, l_aa)  # (n_t, dyn, 8N+1)
    s_dd = h_dd - jnp.einsum("tdk,tek->de", l_da, l_da)
    s_dd = 0.5 * (s_dd + s_dd.T)

    if mode == "dense":
        l_dd = jnp.linalg.cholesky(s_dd)
    elif mode == "blockfunc":
        l_dd = jnp.zeros_like(s_dd)
        for f in range(four_n):
            sl = slice(f * nt, (f + 1) * nt)
            l_dd = l_dd.at[sl, sl].set(jnp.linalg.cholesky(s_dd[sl, sl]))
        l_dd = l_dd.at[a_end:, a_end:].set(jnp.linalg.cholesky(s_dd[a_end:, a_end:]))
    else:
        raise ValueError("mode must be 'dense' or 'blockfunc'")

    alg_index, dyn_index = theta_permutation(n, nt)
    permutation = jnp.concatenate([alg_index.reshape(-1), dyn_index])
    inverse_permutation = jnp.argsort(permutation)
    if return_blocks:
        return {"h_aa": h_aa, "h_dd": h_dd, "h_da": h_da, "s_dd": s_dd}
    return WhitenedSchurPreconditioner(
        l_aa=l_aa,
        l_da=l_da,
        l_dd=l_dd,
        alg_index=alg_index,
        dyn_index=dyn_index,
        inverse_permutation=inverse_permutation,
    )


def whitened_operators(
    theta_flat, unravel, residual_fn, K, eps, num_countries, n_train
):
    """Matrix-free whitened ``B``/``B'`` apply: ``B u = J (S u)``, ``B' y = S' (J' y)``.

    ``S`` is the metric whitening (``S S' = M^{-1}``): the Cholesky ``L^{-T}`` per
    dynamic-function alpha block (``L = chol(K+eps I)``, matching ``metric.inv_sqrt``)
    and ``eps^{-1/2}`` on the algebraic block. ``S`` is NOT symmetric, so the adjoint
    ``B'`` applies ``S'`` (``L^{-1}`` on alpha), not ``S``. Returns ``(B_apply,
    Bt_apply, apply_S)``; ``apply_S`` maps the whitened solution ``u`` to the parameter
    step ``delta = S u``.
    """

    n, nt = num_countries, n_train
    W = whitening_root(K, eps)  # S_alpha = L^{-T}
    inv_sqrt_eps = 1.0 / math.sqrt(eps)
    dyn = 4 * n * nt

    def apply_S(u):
        alpha = (W @ u[:dyn].reshape(4 * n, nt).T).T.reshape(-1)
        return jnp.concatenate([alpha, inv_sqrt_eps * u[dyn:]])

    def apply_S_transpose(u):
        alpha = (W.T @ u[:dyn].reshape(4 * n, nt).T).T.reshape(-1)
        return jnp.concatenate([alpha, inv_sqrt_eps * u[dyn:]])

    _, jvp = jax.linearize(lambda th: jnp.ravel(residual_fn(unravel(th))), theta_flat)
    transpose = jax.linear_transpose(jvp, theta_flat)

    def b_apply(u):
        return jvp(apply_S(u))

    def bt_apply(y):
        return apply_S_transpose(transpose(y)[0])

    return b_apply, bt_apply, apply_S
