"""Block-eigen preconditioner state for the multicountry growth kernel driver.

The whitened normal operator of the ridge solve is
``N = J~'J~ + ridge*E + damping*I`` with ``J~ = J F_bar^{-1}``. Every
residual row is time-local given the per-time path levels
``y_b(t) = (K_tilde alpha_b)_t`` and derivatives ``y_b'(t) = (K alpha_b)_t``,
so the whitened Jacobian of coefficient block ``b`` restricted to the rows
of time ``t`` is

    C[t, row, b] * K_check[t, :]  +  D[t, row, b] * F.T[t, :],

where ``K_check = K_tilde F^{-1}`` and ``K F^{-1} = F'`` are two fixed
n_t x n_t matrices shared by all ``4N + 1`` blocks, and ``C``/``D`` are the
per-time physics Jacobians from one vmapped ``jacfwd`` of the pointwise
residual (the trade-growth C3 assembly pattern, posed on the primal
whitened operator). Exact diagonal blocks of ``J~'J~`` over any grouping of
the coefficient blocks then reduce to kernel quadratics of those two
matrices with per-time pair weights.

The preconditioner keeps three families, block-diagonal:

- per-country ``{k_i, a_i, q_i, m_i}`` groups (all within-country
  cross-equation coupling exact; countries only couple through the
  aggregate rows and the shared rate path),
- the rate block (its derivative never enters the residual, so its weights
  come out one-sided automatically),
- the free initial values ``U'U`` (unpenalized: damping-only shift).

The aggregate rows (resource and resource-drift at t=0, asset clearing at
t>0) are time-local too and enter the diagonal blocks exactly through row
masks. Dropped off-diagonal couplings only change the CG iteration path,
never the converged step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from nlls_gram import block_eigen_state

from multicountry_growth import multicountry_path_values


def block_eigen_permutation(num_countries, n_train):
    """Family-major permutation for the driver's flattened parameter vector.

    Orders the whitened coordinates as per-country (k_i, a_i, q_i, m_i)
    groups, then the rate block, then the free initial values -- the family
    layout ``multicountry_block_eigen_state`` packs.
    """

    order = []
    for country in range(num_countries):
        for equation in range(4):
            start = (equation * num_countries + country) * n_train
            order.extend(range(start, start + n_train))
    rate_start = 4 * num_countries * n_train
    order.extend(range(rate_start, rate_start + n_train))
    free_start = (4 * num_countries + 1) * n_train
    order.extend(range(free_start, free_start + 2 * num_countries + 1))
    return np.asarray(order, dtype=np.int32)


def multicountry_block_eigen_state(
    theta,
    K,
    K_tilde,
    F,
    permutation,
    parameters,
    k_0_array,
    a_0_array,
    q_reference,
    marginal_wealth_reference,
    output_scale,
    residual_scale,
):
    """Assemble the ``nlls_gram.block_eigen_state`` at linearization point
    ``theta``.

    ``K``/``K_tilde`` are the (derivative, integrated) kernel matrices on
    the train grid at the residual dtype, ``F`` the metric's upper Cholesky
    factor of ``K``. Fully traceable, so the driver's solve callback can
    rebuild the state from the live iterate when ridge continuation
    advances a level.
    """

    num_countries = k_0_array.shape[0]
    n_train = K.shape[0]
    dtype = K.dtype
    num_blocks = 4 * num_countries + 1
    free_size = 2 * num_countries + 1

    alpha_k, alpha_a, alpha_q, alpha_m, alpha_r, initial = theta
    coefficients = jnp.concatenate(
        [alpha_k, alpha_a, alpha_q, alpha_m, alpha_r[None, :]], axis=0
    )
    levels = K_tilde @ coefficients.T
    derivatives = K @ coefficients.T

    log_k_0 = jnp.log(k_0_array)
    log_q_reference = jnp.log(q_reference)
    log_marginal_wealth_reference = jnp.log(marginal_wealth_reference)
    log_rho = jnp.log(jnp.asarray(parameters.rho, dtype=dtype))
    total_population = jnp.sum(jnp.asarray(parameters.populations, dtype=dtype))
    N = num_countries

    # w_t = [levels (4N+1) | derivatives (4N+1) | free (2N+1)]: everything
    # the rows of time t consume; jacfwd over w_t yields C (level columns),
    # D (derivative columns), and U (free columns) in one shot.
    def per_time_rows(w):
        lv = w[:num_blocks]
        dv = w[num_blocks : 2 * num_blocks]
        free = w[2 * num_blocks :]
        q_offsets = free[:N]
        marginal_wealth_offsets = free[N : 2 * N]
        rate_offset = free[-1]
        values = multicountry_path_values(
            k=jnp.exp(log_k_0 + lv[0:N])[None, :],
            a=(a_0_array + output_scale * lv[N : 2 * N])[None, :],
            tobin_q=jnp.exp(log_q_reference + q_offsets + lv[2 * N : 3 * N])[None, :],
            marginal_wealth=jnp.exp(
                log_marginal_wealth_reference
                + marginal_wealth_offsets
                + lv[3 * N : 4 * N]
            )[None, :],
            interest_rate=jnp.exp(log_rho + rate_offset + lv[4 * N])[None],
            dlog_k_dt=dv[0:N][None, :],
            da_dt=(output_scale * dv[N : 2 * N])[None, :],
            dlog_q_dt=dv[2 * N : 3 * N][None, :],
            dlog_marginal_wealth_dt=dv[3 * N : 4 * N][None, :],
            parameters=parameters,
        )
        dynamic = jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
            ]
        )
        # Mirrors the driver's per-capita normalization of the aggregate rows.
        aggregates = (
            jnp.concatenate(
                [
                    values["resource"],
                    values["resource_drift"],
                    values["asset_clearing"],
                ]
            )
            / total_population
        )
        return residual_scale * jnp.concatenate([dynamic, aggregates])

    w = jnp.concatenate(
        [levels, derivatives, jnp.broadcast_to(initial, (n_train, free_size))],
        axis=1,
    )
    jacobian = jax.vmap(jax.jacfwd(per_time_rows))(w)
    # The three aggregate candidates exist at every time; the residual keeps
    # resource and resource-drift at t=0 only, asset clearing at t>0 only.
    time_index = jnp.arange(n_train)
    aggregate_mask = jnp.stack(
        [time_index == 0, time_index == 0, time_index > 0], axis=1
    ).astype(dtype)
    mask = jnp.concatenate(
        [jnp.ones((n_train, 4 * N), dtype=dtype), aggregate_mask], axis=1
    )
    jacobian = jacobian * mask[:, :, None]
    C = jacobian[:, :, :num_blocks]
    D = jacobian[:, :, num_blocks : 2 * num_blocks]
    U = jacobian[:, :, 2 * num_blocks :]

    # Whitened structural matrices: level side K_tilde F^{-1}, derivative
    # side K F^{-1} = F'.
    K_check = jsp_linalg.solve_triangular(F.T, K_tilde.T, lower=True).T
    F_t = F.T

    W_cc = jnp.einsum("trb,trc->tbc", C, C)
    W_cd = jnp.einsum("trb,trc->tbc", C, D)
    W_dd = jnp.einsum("trb,trc->tbc", D, D)

    country_blocks = jnp.stack([jnp.arange(4) * N + country for country in range(N)])

    def gather(weights):
        return weights[:, country_blocks[:, :, None], country_blocks[:, None, :]]

    def contract(left, weights, right):
        return jnp.einsum("ta,tncd,tb->ncadb", left, weights, right)

    gathered_cd = gather(W_cd)
    country = (
        contract(K_check, gather(W_cc), K_check)
        + contract(K_check, gathered_cd, F_t)
        + contract(F_t, jnp.swapaxes(gathered_cd, 2, 3), K_check)
        + contract(F_t, gather(W_dd), F_t)
    ).reshape(N, 4 * n_train, 4 * n_train)

    rate_index = 4 * N

    def rate_contract(left, weights, right):
        return jnp.einsum("ta,t,tb->ab", left, weights, right)

    rate_cd = W_cd[:, rate_index, rate_index]
    rate = (
        rate_contract(K_check, W_cc[:, rate_index, rate_index], K_check)
        + rate_contract(K_check, rate_cd, F_t)
        + rate_contract(F_t, rate_cd, K_check)
        + rate_contract(F_t, W_dd[:, rate_index, rate_index], F_t)
    )[None]

    free_block = jnp.einsum("trf,trg->fg", U, U)[None]

    return block_eigen_state(
        [(country, 1.0), (rate, 1.0), (free_block, 0.0)],
        jnp.asarray(permutation),
    )
