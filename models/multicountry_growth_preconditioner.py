# Arrow + Woodbury preconditioner for the multicountry growth kernel
# driver's matrix-free CG path.
#
# The whitened normal operator of the ridge subproblem is
# A = J~'J~ + ridge*E + damping*I with J~ = J F_bar^{-1}. Excluding the
# n_train asset-clearing rows, no residual row couples two countries, so the
# off-clearing operator is exactly an ARROWHEAD over
#   - per-country groups: the country's five whitened coefficient blocks
#     (k, a, q, c, mu) plus its q/c/mu anchors (5 n_train + 3 each),
#   - a border: the rate coefficient block plus the rate anchor (n_train + 1),
# and the clearing rows add U U' with U spanning n_train whitened columns
# (rank one across countries -- the population-weighted a-direction -- times
# the time modes). Block elimination of the arrowhead (batched per-country
# Cholesky + border Schur complement) followed by the Woodbury identity over
# U is therefore the EXACT shifted inverse at the build point. The
# factorizations and each apply cost O(N) in the country count; the per-node
# weight assembly below is dense over the 5N+1 rows of a node and costs O(N^3).
#
# Assembly never materializes the (5N+1)n x ((5N+1)n + 3N + 1) Jacobian:
# every residual row is time-local given the per-time levels
# y_b(t) = (K_tilde alpha_b)_t, derivatives (K alpha_b)_t, and the free
# anchors, so one vmapped jacfwd of the pointwise rows yields per-time
# weights C (level columns), D (derivative columns), and U_off (anchor
# columns), and every block of J~'J~ is a kernel quadratic in the two shared
# n_t x n_t matrices K_check = K_tilde F^{-1} and K F^{-1} = F'.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from nlls_gram import AnnealRidge, LMAction, Preconditioner, register_pytree_dataclass

from models.multicountry_growth_shared import (
    capital_drift,
    current_account,
    q_drift,
)


# Flat position of every theta leaf, derived from the pytree itself, so the
# arrowhead groups follow the driver's Theta declaration order rather than a
# hard-coded family stride. Integer indices come from tree_flatten + cumsum,
# not unravel(arange): ravel_pytree's unravel would cast them back to floats.
def leaf_index_tree(theta):
    leaves, structure = jax.tree_util.tree_flatten(theta)
    offsets = np.cumsum([0, *(leaf.size for leaf in leaves)])
    return jax.tree_util.tree_unflatten(
        structure,
        [
            jnp.asarray(np.arange(start, stop).reshape(leaf.shape))
            for start, stop, leaf in zip(offsets[:-1], offsets[1:], leaves)
        ],
    )


# Per-country groups: the country's five coefficient rows plus its q/c/mu
# anchors; the border is the rate coefficient block plus the rate anchor.
def arrow_woodbury_indices(theta):
    index = leaf_index_tree(theta)
    country_index = jnp.concatenate(
        [
            index.alpha_k,
            index.alpha_a,
            index.alpha_q,
            index.alpha_c,
            index.alpha_mu,
            jnp.stack(
                [
                    index.q_0s,
                    index.c_0s,
                    index.mu_0s,
                ],
                axis=1,
            ),
        ],
        axis=1,
    )
    border_index = jnp.concatenate([index.alpha_rate, index.rate_0[None]])
    return country_index, border_index


@dataclass(frozen=True, eq=False)
class ArrowWoodburyPreconditioner(Preconditioner):
    # Exact shifted inverse of the build point's whitened normal operator:
    # arrowhead block elimination (per-country Cholesky blocks + rate-border
    # Schur complement) on the off-clearing operator, then the Woodbury
    # identity adds the clearing columns back. The shift is baked at build
    # time and the live damping is ignored -- spectral closeness is all a
    # preconditioner needs, so requires_positive_damping stays False.
    country_index: jax.Array  # (N, 5*n_train + 3)
    border_index: jax.Array  # (n_train + 1,)
    H_factor: jax.Array  # (N, group, group) lower Cholesky
    C: jax.Array  # (N, border, group)
    S_factor: jax.Array  # (border, border) lower Cholesky
    U: jax.Array  # (n, n_train) whitened clearing columns
    solve_U: jax.Array = field(init=False)  # (n, n_train)
    cap_factor: jax.Array = field(init=False)  # (n_train, n_train) lower Cholesky

    def __post_init__(self):
        solve_U = self.base_apply(self.U)
        capacitance = jnp.eye(self.U.shape[1], dtype=self.U.dtype) + (
            self.U.T @ solve_U
        )
        object.__setattr__(self, "solve_U", solve_U)
        object.__setattr__(
            self, "cap_factor", jnp.linalg.cholesky(0.5 * (capacitance + capacitance.T))
        )

    def base_apply(self, V):
        Vb = V[self.border_index]
        Vg = V[self.country_index]
        solve_H = jax.vmap(lambda L, R: jsp_linalg.cho_solve((L, True), R))
        Yg = solve_H(self.H_factor, Vg)
        rhs_b = Vb - jnp.einsum("jbc,jck->bk", self.C, Yg)
        Zb = jsp_linalg.cho_solve((self.S_factor, True), rhs_b)
        Zg = solve_H(self.H_factor, Vg - jnp.einsum("jbc,bk->jck", self.C, Zb))
        out = jnp.zeros_like(V)
        out = out.at[self.border_index].set(Zb)
        out = out.at[self.country_index.reshape(-1)].set(Zg.reshape(-1, V.shape[1]))
        return out

    def apply(self, v, damping, ctx):
        y = self.base_apply(v[:, None])
        y = y - self.solve_U @ jsp_linalg.cho_solve(
            (self.cap_factor, True), self.U.T @ y
        )
        return y[:, 0]


register_pytree_dataclass(
    ArrowWoodburyPreconditioner,
    data_fields=(
        "country_index",
        "border_index",
        "H_factor",
        "C",
        "S_factor",
        "U",
        "solve_U",
        "cap_factor",
    ),
)


# kernel_args carries the train-grid kernel matrices as traced data:
# K, K_tilde, K_check = K_tilde F^{-1}, and F_t = F' = K F^{-1} (F the
# metric's upper Cholesky factor of K). The driver threads it through the
# solver's args so the rebuild callback below stays a value-hashable static.
def build_arrow_woodbury(theta, kernel_args, p, ridge, damping, num_countries):
    K = kernel_args.K
    K_tilde = kernel_args.K_tilde
    K_check = kernel_args.K_check
    F_t = kernel_args.F_t
    N = num_countries
    n_train = K.shape[0]
    dtype = K.dtype
    num_blocks = 5 * N + 1
    num_offsets = 3 * N + 1
    residual_scale = 1.0 / jnp.sqrt(float(num_blocks * n_train))

    populations_p = p.populations
    a_0_p = p.a_0 - jnp.sum(populations_p * p.a_0) / jnp.sum(populations_p)

    coefficients = jnp.concatenate(
        [
            theta.alpha_k,
            theta.alpha_a,
            theta.alpha_q,
            theta.alpha_c,
            theta.alpha_mu,
            theta.alpha_rate[None, :],
        ],
        axis=0,
    )
    anchors_flat = jnp.concatenate(
        [
            theta.q_0s,
            theta.c_0s,
            theta.mu_0s,
            theta.rate_0[None],
        ]
    )
    levels = K_tilde @ coefficients.T
    derivatives = K @ coefficients.T

    # The rows of one time as a function of that time's levels, derivatives,
    # and anchors. Must apply exactly the transforms of the driver's
    # path_coordinates and match its residual_fn rows exactly, or the
    # arrowhead is no longer the exact off-clearing operator at the build
    # point.
    def per_time_rows(w):
        lv = w[:num_blocks]
        dv = w[num_blocks : 2 * num_blocks]
        anchors = w[2 * num_blocks :]
        k = p.k_0 + lv[0:N]
        a = a_0_p + lv[N : 2 * N]
        q = anchors[:N] + lv[2 * N : 3 * N]
        c = anchors[N : 2 * N] + lv[3 * N : 4 * N]
        mu = anchors[2 * N : 3 * N] + lv[4 * N : 5 * N]
        rate_column = (anchors[-1] + lv[5 * N])[..., None]
        gammas = jnp.asarray(p.gammas, dtype=c.dtype)
        populations = jnp.asarray(p.populations, dtype=a.dtype)
        return residual_scale * jnp.concatenate(
            [
                dv[0:N] - capital_drift(k, q, p),
                dv[N : 2 * N]
                - current_account(k, a, q, c, p.productivities, rate_column, p),
                dv[2 * N : 3 * N] - q_drift(k, q, p.productivities, rate_column, p),
                dv[4 * N : 5 * N] - mu * (p.r - rate_column),
                mu * c**gammas - 1.0,
                (jnp.sum(populations * a, axis=-1) / jnp.sum(populations))[None],
            ]
        )

    w = jnp.concatenate(
        [levels, derivatives, jnp.broadcast_to(anchors_flat, (n_train, num_offsets))],
        axis=1,
    )
    jacobian = jax.vmap(jax.jacfwd(per_time_rows))(w)
    C = jacobian[:, :-1, :num_blocks]
    D = jacobian[:, :-1, num_blocks : 2 * num_blocks]
    U_off = jacobian[:, :-1, 2 * num_blocks :]
    c_clear = jacobian[:, -1, :num_blocks]
    d_clear = jacobian[:, -1, num_blocks : 2 * num_blocks]
    u_clear = jacobian[:, -1, 2 * num_blocks :]

    # Per-time pair weights over the dynamic (off-clearing) rows.
    W_cc = jnp.einsum("trb,trc->tbc", C, C)
    W_cd = jnp.einsum("trb,trc->tbc", C, D)
    W_dd = jnp.einsum("trb,trc->tbc", D, D)
    W_cu = jnp.einsum("trb,trf->tbf", C, U_off)
    W_du = jnp.einsum("trb,trf->tbf", D, U_off)
    W_uu = jnp.einsum("trf,trg->tfg", U_off, U_off)

    country_blocks = jnp.arange(5)[None, :] * N + jnp.arange(N)[:, None]
    country_offsets = jnp.arange(3)[None, :] * N + jnp.arange(N)[:, None]
    rate = 5 * N
    rate_offset = 3 * N

    def contract(left, weights, right):
        return jnp.einsum("ta,tncd,tb->ncadb", left, weights, right)

    def gather(weights):
        return weights[:, country_blocks[:, :, None], country_blocks[:, None, :]]

    # Per-country diagonal groups: coefficient-coefficient, coefficient-
    # offset, and offset-offset couplings assembled into (5n+3) blocks.
    gW_cd = gather(W_cd)
    cc = (
        contract(K_check, gather(W_cc), K_check)
        + contract(K_check, gW_cd, F_t)
        + contract(F_t, jnp.swapaxes(gW_cd, 2, 3), K_check)
        + contract(F_t, gather(W_dd), F_t)
    ).reshape(N, 5 * n_train, 5 * n_train)
    gW_cu = W_cu[:, country_blocks[:, :, None], country_offsets[:, None, :]]
    gW_du = W_du[:, country_blocks[:, :, None], country_offsets[:, None, :]]
    co = (
        jnp.einsum("ta,tncf->ncaf", K_check, gW_cu)
        + jnp.einsum("ta,tncf->ncaf", F_t, gW_du)
    ).reshape(N, 5 * n_train, 3)
    oo = jnp.sum(W_uu[:, country_offsets[:, :, None], country_offsets[:, None, :]], 0)
    H = jnp.concatenate(
        [
            jnp.concatenate([cc, co], axis=2),
            jnp.concatenate([jnp.swapaxes(co, 1, 2), oo], axis=2),
        ],
        axis=1,
    )
    country_shift = jnp.concatenate(
        [
            jnp.full(5 * n_train, ridge + damping, dtype=dtype),
            jnp.full(3, damping, dtype=dtype),
        ]
    )
    H = H + jnp.diag(country_shift)[None]
    H_factor = jnp.linalg.cholesky(0.5 * (H + jnp.swapaxes(H, 1, 2)))

    # Border group: rate coefficient block + rate offset. The rate offset's
    # columns coincide with the rate level's (both add to the rate), so its
    # couplings come out of the same weights.
    def rate_contract(left, weights, right):
        return jnp.einsum("ta,t,tb->ab", left, weights, right)

    rr = (
        rate_contract(K_check, W_cc[:, rate, rate], K_check)
        + rate_contract(K_check, W_cd[:, rate, rate], F_t)
        + rate_contract(F_t, W_cd[:, rate, rate], K_check)
        + rate_contract(F_t, W_dd[:, rate, rate], F_t)
    )
    ro = K_check.T @ W_cu[:, rate, rate_offset] + F_t.T @ W_du[:, rate, rate_offset]
    T = jnp.concatenate(
        [
            jnp.concatenate([rr, ro[:, None]], axis=1),
            jnp.concatenate(
                [ro[None, :], jnp.sum(W_uu[:, rate_offset, rate_offset])[None, None]],
                axis=1,
            ),
        ],
        axis=0,
    )
    border_shift = jnp.concatenate(
        [
            jnp.full(n_train, ridge + damping, dtype=dtype),
            jnp.full(1, damping, dtype=dtype),
        ]
    )
    T = T + jnp.diag(border_shift)

    # Border-country couplings, (N, n+1, 5n+3) each.
    bc = (
        jnp.einsum("ta,tnc,tb->nacb", K_check, W_cc[:, rate, country_blocks], K_check)
        + jnp.einsum("ta,tnc,tb->nacb", K_check, W_cd[:, rate, country_blocks], F_t)
        + jnp.einsum("ta,tnc,tb->nacb", F_t, W_cd[:, country_blocks, rate], K_check)
        + jnp.einsum("ta,tnc,tb->nacb", F_t, W_dd[:, rate, country_blocks], F_t)
    ).reshape(N, n_train, 5 * n_train)
    bo = jnp.einsum(
        "ta,tnf->naf", K_check, W_cu[:, rate, country_offsets]
    ) + jnp.einsum("ta,tnf->naf", F_t, W_du[:, rate, country_offsets])
    oc = jnp.einsum(
        "ta,tnc->nca", K_check, W_cu[:, country_blocks, rate_offset]
    ) + jnp.einsum("ta,tnc->nca", F_t, W_du[:, country_blocks, rate_offset])
    C_border = jnp.concatenate(
        [
            jnp.concatenate([bc, bo], axis=2),
            jnp.concatenate(
                [
                    oc.reshape(N, 1, 5 * n_train),
                    jnp.sum(W_uu[:, rate_offset, country_offsets], 0)[:, None, :],
                ],
                axis=2,
            ),
        ],
        axis=1,
    )

    H_inv_Ct = jax.vmap(lambda L, R: jsp_linalg.cho_solve((L, True), R))(
        H_factor, jnp.swapaxes(C_border, 1, 2)
    )
    S = T - jnp.einsum("jbc,jcd->bd", C_border, H_inv_Ct)
    S_factor = jnp.linalg.cholesky(0.5 * (S + S.T))

    # Whitened clearing columns (linear in the a levels, so the jacfwd
    # weights are exact and constant in theta).
    U_coef = (
        c_clear[:, :, None] * K_check[:, None, :]
        + d_clear[:, :, None] * F_t[:, None, :]
    )
    U = jnp.concatenate([U_coef.reshape(n_train, num_blocks * n_train), u_clear], 1).T

    country_index, border_index = arrow_woodbury_indices(theta)
    return ArrowWoodburyPreconditioner(
        country_index=country_index,
        border_index=border_index,
        H_factor=H_factor,
        C=C_border,
        S_factor=S_factor,
        U=U,
    )


# Per-step solve callback: composes the ridge anneal (when configured) and
# rebuilds the preconditioner from the live iterate at the live ridge. The
# rebuild is unconditional -- a stale build point degrades PCG from ~2
# iterations to hundreds, and the build is O(N).
#
# damping_floor enters only the BUILD shift: the batched Cholesky needs its
# blocks comfortably PD in float64, while the live LM damping must stay free
# to anneal to ~machine level exactly as on the dense path -- clamping it
# leaves the soft ridge-floor modes (curvature ~ ridge) contracting by
# ridge/damping per step and the endgame residual never resolves. The
# preconditioner ignores the live damping, so the mismatch only bounds the
# preconditioned condition number by ~ damping_floor/ridge_floor.
#
# A frozen dataclass rather than a closure: solve() marks the callback a jit
# STATIC argument, so equal configurations must compare equal to share one
# compiled loop. The kernel matrices ride in the solver's args (ctx.args),
# not in fields, keeping the instance hashable.
@dataclass(frozen=True)
class ArrowWoodburyCallback:
    num_countries: int
    damping_floor: float
    anneal: AnnealRidge | None = None

    def __call__(self, ctx):
        action = (
            self.anneal(ctx)
            if self.anneal is not None
            else LMAction(lm_state=ctx.lm_state)
        )
        lm_state = action.lm_state
        preconditioner = build_arrow_woodbury(
            ctx.x,
            ctx.args,
            ctx.p,
            lm_state.ridge,
            jnp.maximum(lm_state.damping, self.damping_floor),
            self.num_countries,
        )
        return dataclasses.replace(
            action,
            lm_state=dataclasses.replace(lm_state, preconditioner=preconditioner),
        )
