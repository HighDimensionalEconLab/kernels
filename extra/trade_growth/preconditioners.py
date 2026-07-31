from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from nlls_gram import Preconditioner, register_pytree_dataclass

from extra.trade_growth.model import (
    STATIC_BLOCK_NAMES,
    consumption_basket_log_price,
    multicountry_trade_path_values,
)


def trade_growth_block_permutation(num_countries, n_train):
    num_country_functions = 4 + len(STATIC_BLOCK_NAMES)
    num_country_initials = 2 + len(STATIC_BLOCK_NAMES)
    num_functions = num_country_functions * num_countries + 1
    free_start = num_functions * n_train
    order = []
    for country in range(num_countries):
        for function in range(num_country_functions):
            start = (function * num_countries + country) * n_train
            order.extend(range(start, start + n_train))
        for initial in range(num_country_initials):
            order.append(free_start + initial * num_countries + country)
    rate_start = (num_functions - 1) * n_train
    order.extend(range(rate_start, rate_start + n_train))
    order.append(free_start + num_country_initials * num_countries)
    return np.asarray(order, dtype=np.int32)


def trade_growth_local_linearization(
    theta,
    K,
    K_tilde,
    F,
    p,
    residual_scale,
    free_scale=1.0,
):
    parameters = p["parameters"]
    populations = parameters.populations
    num_countries = populations.shape[0]
    n_train = K.shape[0]
    num_static = len(STATIC_BLOCK_NAMES)
    num_country_functions = 4 + num_static
    num_country_initials = 2 + num_static
    num_functions = num_country_functions * num_countries + 1
    num_initials = num_country_initials * num_countries + 1
    alpha_k, alpha_a, alpha_q, alpha_m, alpha_static, alpha_r, initial = theta
    coefficients = jnp.concatenate(
        [
            alpha_k,
            alpha_a,
            alpha_q,
            alpha_m,
            alpha_static.reshape(num_static * num_countries, n_train),
            alpha_r[None, :],
        ],
        axis=0,
    )
    levels = K_tilde @ coefficients.T
    derivatives = K @ coefficients.T
    k_0 = p["k_0"]
    a_0 = p["a_0"] - jnp.sum(populations * p["a_0"]) / jnp.sum(populations)
    log_k_0 = jnp.log(k_0)
    N = num_countries

    def per_time_rows(w):
        lv = w[:num_functions]
        dv = w[num_functions : 2 * num_functions]
        free = w[2 * num_functions :]
        log_q_0 = free[:N]
        log_marginal_wealth_0 = free[N : 2 * N]
        log_static_0 = free[2 * N : -1].reshape(num_static, N)
        log_rate_0 = free[-1]
        static_levels = lv[4 * N : -1].reshape(num_static, N)
        values = multicountry_trade_path_values(
            k=jnp.exp(log_k_0 + lv[:N])[None, :],
            a=(a_0 + lv[N : 2 * N])[None, :],
            tobin_q=jnp.exp(log_q_0 + lv[2 * N : 3 * N])[None, :],
            marginal_wealth=jnp.exp(log_marginal_wealth_0 + lv[3 * N : 4 * N])[None, :],
            interest_rate=jnp.exp(log_rate_0 + lv[-1])[None],
            log_static=(log_static_0 + static_levels)[None, :, :],
            dlog_k_dt=dv[:N][None, :],
            da_dt=dv[N : 2 * N][None, :],
            dlog_q_dt=dv[2 * N : 3 * N][None, :],
            dlog_marginal_wealth_dt=dv[3 * N : 4 * N][None, :],
            parameters=parameters,
        )
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
                values["static_residuals"].reshape(-1),
                consumption_basket_log_price(
                    log_static_0 + static_levels,
                    parameters,
                ).reshape(-1),
            ]
        )

    w = jnp.concatenate(
        [levels, derivatives, jnp.broadcast_to(initial, (n_train, num_initials))],
        axis=1,
    )
    jacobian = jax.vmap(jax.jacfwd(per_time_rows))(w)
    C = jacobian[:, :, :num_functions]
    D = jacobian[:, :, num_functions : 2 * num_functions]
    U = jacobian[:, :, 2 * num_functions :]
    # F is the exact upper factor of the coefficient metric.  In the
    # RidgeLM case F'F=K, so the derivative map simplifies to F'.  MetricLM
    # deliberately shifts the metric, F'F=K+epsilon*I, and that shortcut is
    # no longer valid: both maps must be formed as kernel_matrix @ F^{-1}.
    K_check = jsp_linalg.solve_triangular(F.T, K_tilde.T, lower=True).T
    K_derivative = jsp_linalg.solve_triangular(F.T, K.T, lower=True).T
    U_whitened = U / jnp.sqrt(jnp.asarray(free_scale, dtype=U.dtype))
    return C, D, U_whitened, K_check, K_derivative


def trade_growth_time_blocks(
    theta,
    K,
    K_tilde,
    F,
    p,
    residual_scale,
    free_scale=1.0,
):
    """Build the residual-space time blocks for MetricLM GramCG.

    Conditional on the path values at a date, every economic equation is
    local in time.  The RKHS coefficients induce cross-date covariance, but
    the diagonal time block can be contracted exactly from the two shared
    whitened kernel rows.  The free initial values are different: the same
    ``10*N+1`` values enter every date, so their full cross-date low-rank
    contribution is retained by :class:`TradeGrowthTimeBlockPreconditioner`.
    """

    C, D, U, K_check, K_derivative = trade_growth_local_linearization(
        theta,
        K,
        K_tilde,
        F,
        p,
        residual_scale,
        free_scale,
    )
    return assemble_trade_growth_time_blocks(C, D, U, K_check, K_derivative), U


def assemble_trade_growth_time_blocks(C, D, U, K_check, K_derivative):
    q_ll = jnp.einsum("ta,ta->t", K_check, K_check)
    q_ld = jnp.einsum("ta,ta->t", K_check, K_derivative)
    q_dd = jnp.einsum("ta,ta->t", K_derivative, K_derivative)
    cc = jnp.einsum("trb,tsb->trs", C, C)
    cd = jnp.einsum("trb,tsb->trs", C, D)
    dd = jnp.einsum("trb,tsb->trs", D, D)
    coefficient_blocks = (
        q_ll[:, None, None] * cc
        + q_ld[:, None, None] * (cd + jnp.swapaxes(cd, 1, 2))
        + q_dd[:, None, None] * dd
    )
    return 0.5 * (coefficient_blocks + jnp.swapaxes(coefficient_blocks, 1, 2))


def trade_growth_equation_blocks(
    theta,
    K,
    K_tilde,
    F,
    p,
    residual_scale,
    free_scale=1.0,
):
    """Exact temporal Gram block for each economic residual stream."""

    C, D, U, K_check, K_derivative = trade_growth_local_linearization(
        theta,
        K,
        K_tilde,
        F,
        p,
        residual_scale,
        free_scale,
    )
    return assemble_trade_growth_equation_blocks(C, D, U, K_check, K_derivative)


def assemble_trade_growth_equation_blocks(C, D, U, K_check, K_derivative):
    q_ll = K_check @ K_check.T
    q_ld = K_check @ K_derivative.T
    q_dl = K_derivative @ K_check.T
    q_dd = K_derivative @ K_derivative.T
    cc = jnp.einsum("trb,srb->rts", C, C)
    cd = jnp.einsum("trb,srb->rts", C, D)
    dc = jnp.einsum("trb,srb->rts", D, C)
    dd = jnp.einsum("trb,srb->rts", D, D)
    uu = jnp.einsum("trf,srf->rts", U, U)
    blocks = (
        q_ll[None, :, :] * cc
        + q_ld[None, :, :] * cd
        + q_dl[None, :, :] * dc
        + q_dd[None, :, :] * dd
        + uu
    )
    return 0.5 * (blocks + jnp.swapaxes(blocks, 1, 2))


def _residual_to_time_order(v, n_train, num_countries):
    T, N = n_train, num_countries
    dynamic_end = 4 * T * N
    static_end = 12 * T * N
    dynamic = v[:dynamic_end].reshape(4, T, N).transpose(1, 0, 2)
    static = v[dynamic_end:static_end].reshape(T, 8, N)
    price = v[static_end:].reshape(T, 1)
    return jnp.concatenate(
        [dynamic.reshape(T, 4 * N), static.reshape(T, 8 * N), price], axis=1
    )


def _time_to_residual_order(v, n_train, num_countries):
    T, N = n_train, num_countries
    dynamic = v[:, : 4 * N].reshape(T, 4, N).transpose(1, 0, 2)
    static = v[:, 4 * N : 12 * N].reshape(T, 8, N)
    price = v[:, -1]
    return jnp.concatenate([dynamic.reshape(-1), static.reshape(-1), price])


@dataclass(frozen=True, eq=False)
class TradeGrowthEquationBlockPreconditioner(Preconditioner):
    """Full-horizon inverse for each repeated economic equation stream."""

    blocks: InitVar[jax.Array]
    num_countries: int
    eigenvectors: jax.Array = field(init=False)
    eigenvalues: jax.Array = field(init=False)
    n_train: int = field(init=False)

    def __post_init__(self, blocks):
        blocks = jnp.asarray(blocks)
        expected = 12 * self.num_countries + 1
        if blocks.ndim != 3 or blocks.shape[0] != expected:
            raise ValueError("one temporal block is required per residual stream")
        if blocks.shape[1] != blocks.shape[2]:
            raise ValueError("equation blocks must be square")
        values, vectors = jnp.linalg.eigh(0.5 * (blocks + jnp.swapaxes(blocks, 1, 2)))
        scale = jnp.maximum(
            jnp.max(jnp.abs(values), axis=1),
            jnp.ones(blocks.shape[0], dtype=blocks.dtype),
        )
        floor = (
            jnp.maximum(
                100.0 * jnp.finfo(blocks.dtype).eps,
                jnp.asarray(1e-10, dtype=blocks.dtype),
            )
            * scale
        )
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "eigenvalues", jnp.maximum(values, floor[:, None]))
        object.__setattr__(self, "n_train", blocks.shape[1])

    def apply(self, v, damping, ctx):
        del ctx
        streams = _residual_to_time_order(v, self.n_train, self.num_countries).T
        coefficients = jnp.einsum("rji,rj->ri", self.eigenvectors, streams)
        result = jnp.einsum(
            "rji,ri->rj",
            self.eigenvectors,
            coefficients / (self.eigenvalues + damping),
        )
        return _time_to_residual_order(
            result.T, self.n_train, self.num_countries
        ).astype(v.dtype)


register_pytree_dataclass(
    TradeGrowthEquationBlockPreconditioner,
    data_fields=("eigenvectors", "eigenvalues"),
    meta_fields=("num_countries", "n_train"),
)


@dataclass(frozen=True, eq=False)
class TradeGrowthTimeBlockPreconditioner(Preconditioner):
    """Time-local dual inverse with an exact global initial-value update.

    The base matrix is block diagonal over collocation dates and keeps every
    within-date country, trade, price-numeraire, and common-rate coupling.
    A Woodbury update then restores the complete cross-date coupling generated
    by the weakly metric-weighted initial values.  Only off-date coefficient
    covariance is dropped.
    """

    coefficient_blocks: InitVar[jax.Array]
    initial_jacobian: jax.Array
    num_countries: int
    eigenvectors: jax.Array = field(init=False)
    eigenvalues: jax.Array = field(init=False)
    n_train: int = field(init=False)

    def __post_init__(self, coefficient_blocks):
        blocks = jnp.asarray(coefficient_blocks)
        initial_jacobian = jnp.asarray(self.initial_jacobian)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError("coefficient_blocks must be a batch of square matrices")
        expected_rows = 12 * self.num_countries + 1
        expected_initials = 10 * self.num_countries + 1
        if blocks.shape[1] != expected_rows:
            raise ValueError("time-block residual dimension is inconsistent")
        if initial_jacobian.shape != (
            blocks.shape[0],
            expected_rows,
            expected_initials,
        ):
            raise ValueError("initial_jacobian shape is inconsistent")
        values, vectors = jnp.linalg.eigh(0.5 * (blocks + jnp.swapaxes(blocks, 1, 2)))
        scale = jnp.maximum(
            jnp.max(jnp.abs(values), axis=1),
            jnp.ones(blocks.shape[0], dtype=blocks.dtype),
        )
        # At t=0 the integrated level row is identically zero, so the
        # coefficient-only block is structurally rank deficient; the free
        # initial-value update identifies those rows.  A modest relative
        # floor keeps the block inverse and subsequent Woodbury subtraction
        # numerically SPD without pretending to approximate missing economic
        # curvature.
        floor = (
            jnp.maximum(
                100.0 * jnp.finfo(blocks.dtype).eps,
                jnp.asarray(1e-8, dtype=blocks.dtype),
            )
            * scale
        )
        object.__setattr__(self, "initial_jacobian", initial_jacobian)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "eigenvalues", jnp.maximum(values, floor[:, None]))
        object.__setattr__(self, "n_train", blocks.shape[0])

    def _to_time_order(self, v):
        return _residual_to_time_order(v, self.n_train, self.num_countries)

    def _from_time_order(self, v):
        return _time_to_residual_order(v, self.n_train, self.num_countries)

    def _base_inverse(self, v, damping):
        coefficients = jnp.einsum("tji,tj...->ti...", self.eigenvectors, v)
        trailing = (1,) * (v.ndim - 2)
        denominators = (self.eigenvalues + damping).reshape(
            self.eigenvalues.shape + trailing
        )
        return jnp.einsum(
            "tij,tj...->ti...",
            self.eigenvectors,
            coefficients / denominators,
        )

    def apply(self, v, damping, ctx):
        del ctx
        time_v = self._to_time_order(v)
        base_v = self._base_inverse(time_v, damping)
        base_U = self._base_inverse(self.initial_jacobian, damping)
        schur = jnp.eye(self.initial_jacobian.shape[-1], dtype=v.dtype) + jnp.einsum(
            "trf,trg->fg", self.initial_jacobian, base_U
        )
        right = jnp.einsum("trf,tr->f", self.initial_jacobian, base_v)
        correction_weights = jnp.linalg.solve(schur, right)
        result = base_v - jnp.einsum("trf,f->tr", base_U, correction_weights)
        return self._from_time_order(result).astype(v.dtype)


register_pytree_dataclass(
    TradeGrowthTimeBlockPreconditioner,
    data_fields=("initial_jacobian", "eigenvectors", "eigenvalues"),
    meta_fields=("num_countries", "n_train"),
)


@dataclass(frozen=True, eq=False)
class TradeGrowthHybridPreconditioner(Preconditioner):
    """Additive Schwarz combination of local-economy and temporal solves."""

    time_blocks: TradeGrowthTimeBlockPreconditioner
    equation_blocks: TradeGrowthEquationBlockPreconditioner

    def apply(self, v, damping, ctx):
        return 0.5 * (
            self.time_blocks.apply(v, damping, ctx)
            + self.equation_blocks.apply(v, damping, ctx)
        )


register_pytree_dataclass(
    TradeGrowthHybridPreconditioner,
    data_fields=("time_blocks", "equation_blocks"),
)


@dataclass(frozen=True, eq=False)
class TradeGrowthMultiplicativePreconditioner(Preconditioner):
    """Symmetric multiplicative Schwarz on the exact frozen dual operator."""

    time_blocks: TradeGrowthTimeBlockPreconditioner
    equation_blocks: TradeGrowthEquationBlockPreconditioner
    C: jax.Array
    D: jax.Array
    U: jax.Array
    K_check: jax.Array
    K_derivative: jax.Array
    time_first: bool = True

    def _dual_apply(self, v, damping):
        time_v = _residual_to_time_order(
            v, self.time_blocks.n_train, self.time_blocks.num_countries
        )
        coefficient_gradient = jnp.einsum(
            "tr,trb,ta->ba", time_v, self.C, self.K_check
        ) + jnp.einsum("tr,trb,ta->ba", time_v, self.D, self.K_derivative)
        coefficient_result = jnp.einsum(
            "trb,ta,ba->tr", self.C, self.K_check, coefficient_gradient
        ) + jnp.einsum(
            "trb,ta,ba->tr",
            self.D,
            self.K_derivative,
            coefficient_gradient,
        )
        initial_gradient = jnp.einsum("tr,trf->f", time_v, self.U)
        initial_result = jnp.einsum("trf,f->tr", self.U, initial_gradient)
        result = coefficient_result + initial_result + damping * time_v
        return _time_to_residual_order(
            result,
            self.time_blocks.n_train,
            self.time_blocks.num_countries,
        )

    def apply(self, v, damping, ctx):
        first = self.time_blocks if self.time_first else self.equation_blocks
        second = self.equation_blocks if self.time_first else self.time_blocks
        first_v = first.apply(v, damping, ctx)
        residual = v - self._dual_apply(first_v, damping)
        correction = second.apply(residual, damping, ctx)
        return (
            first_v
            + correction
            - first.apply(self._dual_apply(correction, damping), damping, ctx)
        )


register_pytree_dataclass(
    TradeGrowthMultiplicativePreconditioner,
    data_fields=(
        "time_blocks",
        "equation_blocks",
        "C",
        "D",
        "U",
        "K_check",
        "K_derivative",
    ),
    meta_fields=("time_first",),
)


def trade_growth_block_eigen_families(
    theta,
    K,
    K_tilde,
    F,
    p,
    residual_scale,
    ridge,
    free_scale=1.0,
):
    parameters = p["parameters"]
    populations = parameters.populations
    num_countries = populations.shape[0]
    n_train = K.shape[0]
    dtype = K.dtype
    num_static = len(STATIC_BLOCK_NAMES)
    num_country_functions = 4 + num_static
    num_country_initials = 2 + num_static
    num_functions = num_country_functions * num_countries + 1
    num_initials = num_country_initials * num_countries + 1

    alpha_k, alpha_a, alpha_q, alpha_m, alpha_static, alpha_r, initial = theta
    coefficients = jnp.concatenate(
        [
            alpha_k,
            alpha_a,
            alpha_q,
            alpha_m,
            alpha_static.reshape(num_static * num_countries, n_train),
            alpha_r[None, :],
        ],
        axis=0,
    )
    levels = K_tilde @ coefficients.T
    derivatives = K @ coefficients.T

    k_0 = p["k_0"]
    a_0 = p["a_0"] - jnp.sum(populations * p["a_0"]) / jnp.sum(populations)
    log_k_0 = jnp.log(k_0)
    N = num_countries

    # Each collocation row only consumes the path levels, derivatives, and
    # initial offsets at that date. One vmapped local Jacobian therefore
    # contains every weight needed for the block-normal assembly.
    def per_time_rows(w):
        lv = w[:num_functions]
        dv = w[num_functions : 2 * num_functions]
        free = w[2 * num_functions :]
        log_q_0 = free[:N]
        log_marginal_wealth_0 = free[N : 2 * N]
        log_static_0 = free[2 * N : -1].reshape(num_static, N)
        log_rate_0 = free[-1]
        static_levels = lv[4 * N : -1].reshape(num_static, N)
        values = multicountry_trade_path_values(
            k=jnp.exp(log_k_0 + lv[:N])[None, :],
            a=(a_0 + lv[N : 2 * N])[None, :],
            tobin_q=jnp.exp(log_q_0 + lv[2 * N : 3 * N])[None, :],
            marginal_wealth=jnp.exp(log_marginal_wealth_0 + lv[3 * N : 4 * N])[None, :],
            interest_rate=jnp.exp(log_rate_0 + lv[-1])[None],
            log_static=(log_static_0 + static_levels)[None, :, :],
            dlog_k_dt=dv[:N][None, :],
            da_dt=dv[N : 2 * N][None, :],
            dlog_q_dt=dv[2 * N : 3 * N][None, :],
            dlog_marginal_wealth_dt=dv[3 * N : 4 * N][None, :],
            parameters=parameters,
        )
        return residual_scale * jnp.concatenate(
            [
                values["capital"].reshape(-1),
                values["assets"].reshape(-1),
                values["tobin_q"].reshape(-1),
                values["marginal_wealth_euler"].reshape(-1),
                values["static_residuals"].reshape(-1),
                consumption_basket_log_price(
                    log_static_0 + static_levels,
                    parameters,
                ).reshape(-1),
            ]
        )

    w = jnp.concatenate(
        [levels, derivatives, jnp.broadcast_to(initial, (n_train, num_initials))],
        axis=1,
    )
    jacobian = jax.vmap(jax.jacfwd(per_time_rows))(w)
    C = jacobian[:, :, :num_functions]
    D = jacobian[:, :, num_functions : 2 * num_functions]
    U = jacobian[:, :, 2 * num_functions :] / jnp.sqrt(
        jnp.asarray(free_scale, dtype=dtype)
    )

    # Whitening gives K_tilde F^{-1} on levels and K F^{-1} on derivatives.
    # The latter equals F' only in the unshifted RidgeLM metric.
    K_check = jsp_linalg.solve_triangular(F.T, K_tilde.T, lower=True).T
    F_t = jsp_linalg.solve_triangular(F.T, K.T, lower=True).T
    W_cc = jnp.einsum("trb,trc->tbc", C, C)
    W_cd = jnp.einsum("trb,trc->tbc", C, D)
    W_dd = jnp.einsum("trb,trc->tbc", D, D)
    W_cu = jnp.einsum("trb,trf->tbf", C, U)
    W_du = jnp.einsum("trb,trf->tbf", D, U)
    W_uu = jnp.einsum("trf,trg->tfg", U, U)

    country_functions = jnp.stack(
        [jnp.arange(num_country_functions) * N + country for country in range(N)]
    )
    country_initials = jnp.stack(
        [jnp.arange(num_country_initials) * N + country for country in range(N)]
    )

    def gather_square(weights, indices):
        return weights[:, indices[:, :, None], indices[:, None, :]]

    def gather_cross(weights):
        return weights[:, country_functions[:, :, None], country_initials[:, None, :]]

    def kernel_contract(left, weights, right):
        return jnp.einsum("ta,tnij,tb->niajb", left, weights, right)

    gathered_cd = gather_square(W_cd, country_functions)
    country_coefficient = (
        kernel_contract(K_check, gather_square(W_cc, country_functions), K_check)
        + kernel_contract(K_check, gathered_cd, F_t)
        + kernel_contract(F_t, jnp.swapaxes(gathered_cd, 2, 3), K_check)
        + kernel_contract(F_t, gather_square(W_dd, country_functions), F_t)
    ).reshape(
        N,
        num_country_functions * n_train,
        num_country_functions * n_train,
    )
    country_cross = (
        jnp.einsum("ta,tnij->niaj", K_check, gather_cross(W_cu))
        + jnp.einsum("ta,tnij->niaj", F_t, gather_cross(W_du))
    ).reshape(N, num_country_functions * n_train, num_country_initials)
    country_free = jnp.sum(
        gather_square(W_uu, country_initials),
        axis=0,
    )
    country = jnp.concatenate(
        [
            jnp.concatenate([country_coefficient, country_cross], axis=2),
            jnp.concatenate([jnp.swapaxes(country_cross, 1, 2), country_free], axis=2),
        ],
        axis=1,
    )

    rate_function = num_functions - 1
    rate_initial = num_initials - 1

    def scalar_kernel_contract(left, weights, right):
        return jnp.einsum("ta,t,tb->ab", left, weights, right)

    rate_cd = W_cd[:, rate_function, rate_function]
    rate_coefficient = (
        scalar_kernel_contract(K_check, W_cc[:, rate_function, rate_function], K_check)
        + scalar_kernel_contract(K_check, rate_cd, F_t)
        + scalar_kernel_contract(F_t, rate_cd, K_check)
        + scalar_kernel_contract(F_t, W_dd[:, rate_function, rate_function], F_t)
    )
    rate_cross = jnp.sum(
        K_check * W_cu[:, rate_function, rate_initial, None]
        + F_t * W_du[:, rate_function, rate_initial, None],
        axis=0,
    )
    rate_free = jnp.sum(W_uu[:, rate_initial, rate_initial])
    rate = jnp.block(
        [
            [rate_coefficient, rate_cross[:, None]],
            [rate_cross[None, :], rate_free[None, None]],
        ]
    )[None, :, :]

    country_metric_size = num_country_functions * n_train
    country_diagonal = jnp.arange(country_metric_size)
    country = country.at[:, country_diagonal, country_diagonal].add(ridge)
    rate_diagonal = jnp.arange(n_train)
    rate = rate.at[:, rate_diagonal, rate_diagonal].add(ridge)

    # The blocks are theoretically positive definite after the ridge is added
    # to coefficients and the free levels are identified. A roundoff-scale
    # diagonal floor keeps their eigendecompositions SPD without changing the
    # nonlinear target.
    country_scale = jnp.maximum(
        jnp.max(jnp.abs(jnp.diagonal(country, axis1=1, axis2=2))),
        jnp.asarray(1.0, dtype=dtype),
    )
    rate_scale = jnp.maximum(
        jnp.max(jnp.abs(jnp.diagonal(rate, axis1=1, axis2=2))),
        jnp.asarray(1.0, dtype=dtype),
    )
    country_floor = jnp.maximum(
        10.0 * jnp.finfo(dtype).eps * country_scale,
        ridge * jnp.asarray(1e-6, dtype=dtype),
    )
    rate_floor = jnp.maximum(
        10.0 * jnp.finfo(dtype).eps * rate_scale,
        ridge * jnp.asarray(1e-6, dtype=dtype),
    )
    country = country + country_floor * jnp.eye(country.shape[-1], dtype=dtype)
    rate = rate + rate_floor * jnp.eye(rate.shape[-1], dtype=dtype)
    return [(country, 0.0), (rate, 0.0)]


def helmert_country_modes(num_countries, dtype):
    modes = np.zeros((num_countries, num_countries))
    modes[:, 0] = 1.0 / np.sqrt(num_countries)
    for column in range(1, num_countries):
        scale = np.sqrt(column * (column + 1.0))
        modes[:column, column] = 1.0 / scale
        modes[column, column] = -column / scale
    return jnp.asarray(modes, dtype=dtype)


def assemble_mode_blocks(C, D, U, K_check, F_t, ridge):
    dtype = C.dtype
    num_groups = C.shape[2]
    num_functions = C.shape[3]
    num_initials = U.shape[3]
    n_train = K_check.shape[0]
    W_cc = jnp.einsum("trgi,trgj->tgij", C, C)
    W_cd = jnp.einsum("trgi,trgj->tgij", C, D)
    W_dd = jnp.einsum("trgi,trgj->tgij", D, D)
    W_cu = jnp.einsum("trgi,trgf->tgif", C, U)
    W_du = jnp.einsum("trgi,trgf->tgif", D, U)
    W_uu = jnp.einsum("trgf,trgh->tgfh", U, U)

    def contract(left, weights, right):
        return jnp.einsum("ta,tgij,tb->giajb", left, weights, right)

    coefficient = (
        contract(K_check, W_cc, K_check)
        + contract(K_check, W_cd, F_t)
        + contract(F_t, jnp.swapaxes(W_cd, 2, 3), K_check)
        + contract(F_t, W_dd, F_t)
    ).reshape(
        num_groups,
        num_functions * n_train,
        num_functions * n_train,
    )
    cross = (
        jnp.einsum("ta,tgif->giaf", K_check, W_cu)
        + jnp.einsum("ta,tgif->giaf", F_t, W_du)
    ).reshape(num_groups, num_functions * n_train, num_initials)
    free = jnp.sum(W_uu, axis=0)
    blocks = jnp.concatenate(
        [
            jnp.concatenate([coefficient, cross], axis=2),
            jnp.concatenate([jnp.swapaxes(cross, 1, 2), free], axis=2),
        ],
        axis=1,
    )
    metric_size = num_functions * n_train
    metric_diagonal = jnp.arange(metric_size)
    blocks = blocks.at[:, metric_diagonal, metric_diagonal].add(ridge)
    scale = jnp.maximum(
        jnp.max(jnp.abs(jnp.diagonal(blocks, axis1=1, axis2=2)), axis=1),
        jnp.ones(num_groups, dtype=dtype),
    )
    floor = jnp.maximum(
        10.0 * jnp.finfo(dtype).eps * scale,
        ridge * jnp.asarray(1e-6, dtype=dtype),
    )
    return blocks + floor[:, None, None] * jnp.eye(blocks.shape[-1], dtype=dtype)


def assemble_mode_cross_blocks(
    C_left,
    D_left,
    U_left,
    C_right,
    D_right,
    U_right,
    K_check,
    F_t,
):
    num_groups = C_right.shape[2]
    left_functions = C_left.shape[2]
    right_functions = C_right.shape[3]
    left_initials = U_left.shape[2]
    right_initials = U_right.shape[3]
    n_train = K_check.shape[0]
    W_cc = jnp.einsum("tri,trgj->tgij", C_left, C_right)
    W_cd = jnp.einsum("tri,trgj->tgij", C_left, D_right)
    W_dc = jnp.einsum("tri,trgj->tgij", D_left, C_right)
    W_dd = jnp.einsum("tri,trgj->tgij", D_left, D_right)

    def contract(left, weights, right):
        return jnp.einsum("ta,tgij,tb->giajb", left, weights, right)

    coefficient = (
        contract(K_check, W_cc, K_check)
        + contract(K_check, W_cd, F_t)
        + contract(F_t, W_dc, K_check)
        + contract(F_t, W_dd, F_t)
    ).reshape(
        num_groups,
        left_functions * n_train,
        right_functions * n_train,
    )
    W_cu = jnp.einsum("tri,trgf->tgif", C_left, U_right)
    W_du = jnp.einsum("tri,trgf->tgif", D_left, U_right)
    coefficient_free = (
        jnp.einsum("ta,tgif->giaf", K_check, W_cu)
        + jnp.einsum("ta,tgif->giaf", F_t, W_du)
    ).reshape(num_groups, left_functions * n_train, right_initials)
    W_uc = jnp.einsum("trf,trgj->tgfj", U_left, C_right)
    W_ud = jnp.einsum("trf,trgj->tgfj", U_left, D_right)
    free_coefficient = (
        jnp.einsum("tgfj,tb->gfjb", W_uc, K_check)
        + jnp.einsum("tgfj,tb->gfjb", W_ud, F_t)
    ).reshape(num_groups, left_initials, right_functions * n_train)
    free_free = jnp.sum(jnp.einsum("trf,trgh->tgfh", U_left, U_right), axis=0)
    return jnp.concatenate(
        [
            jnp.concatenate([coefficient, coefficient_free], axis=2),
            jnp.concatenate([free_coefficient, free_free], axis=2),
        ],
        axis=1,
    )


def trade_growth_mode_blocks(
    theta,
    K,
    K_tilde,
    F,
    p,
    residual_scale,
    ridge,
    free_scale=1.0,
):
    C, D, U, K_check, F_t = trade_growth_local_linearization(
        theta,
        K,
        K_tilde,
        F,
        p,
        residual_scale,
        free_scale,
    )
    num_countries = p["parameters"].populations.shape[0]
    num_country_functions = 4 + len(STATIC_BLOCK_NAMES)
    num_country_initials = 2 + len(STATIC_BLOCK_NAMES)
    # The common interest rate, world asset balance, and price numeraire load
    # primarily on the equal-weight country mode. Trade reallocations load on
    # country contrasts. Keeping the common block exact captures the global
    # economic closure without making every country block dense in N.
    mode_matrix = helmert_country_modes(num_countries, K.dtype)
    C_country = C[:, :, : num_country_functions * num_countries].reshape(
        C.shape[0], C.shape[1], num_country_functions, num_countries
    )
    D_country = D[:, :, : num_country_functions * num_countries].reshape(
        D.shape[0], D.shape[1], num_country_functions, num_countries
    )
    U_country = U[:, :, : num_country_initials * num_countries].reshape(
        U.shape[0], U.shape[1], num_country_initials, num_countries
    )
    C_modes = jnp.einsum("trbi,im->trmb", C_country, mode_matrix)
    D_modes = jnp.einsum("trbi,im->trmb", D_country, mode_matrix)
    U_modes = jnp.einsum("trfi,im->trmf", U_country, mode_matrix)
    C_aggregate = jnp.concatenate([C_modes[:, :, 0, :], C[:, :, -1, None]], axis=2)[
        :, :, None, :
    ]
    D_aggregate = jnp.concatenate([D_modes[:, :, 0, :], D[:, :, -1, None]], axis=2)[
        :, :, None, :
    ]
    U_aggregate = jnp.concatenate([U_modes[:, :, 0, :], U[:, :, -1, None]], axis=2)[
        :, :, None, :
    ]
    aggregate = assemble_mode_blocks(
        C_aggregate,
        D_aggregate,
        U_aggregate,
        K_check,
        F_t,
        ridge,
    )[0]
    idiosyncratic = assemble_mode_blocks(
        C_modes[:, :, 1:, :],
        D_modes[:, :, 1:, :],
        U_modes[:, :, 1:, :],
        K_check,
        F_t,
        ridge,
    )
    aggregate_cross = assemble_mode_cross_blocks(
        C_aggregate[:, :, 0, :],
        D_aggregate[:, :, 0, :],
        U_aggregate[:, :, 0, :],
        C_modes[:, :, 1:, :],
        D_modes[:, :, 1:, :],
        U_modes[:, :, 1:, :],
        K_check,
        F_t,
    )
    return mode_matrix, aggregate, idiosyncratic, aggregate_cross


@dataclass(frozen=True, eq=False)
class TradeGrowthModePreconditioner(Preconditioner):
    aggregate_block: InitVar[jax.Array]
    idiosyncratic_blocks: InitVar[jax.Array]
    mode_matrix: jax.Array
    aggregate_eigenvectors: jax.Array = field(init=False)
    aggregate_eigenvalues: jax.Array = field(init=False)
    idiosyncratic_eigenvectors: jax.Array = field(init=False)
    idiosyncratic_eigenvalues: jax.Array = field(init=False)
    num_countries: int = field(init=False)
    n_train: int = field(init=False)

    def __post_init__(self, aggregate_block, idiosyncratic_blocks):
        mode_matrix = jnp.asarray(self.mode_matrix)
        aggregate_block = jnp.asarray(aggregate_block)
        idiosyncratic_blocks = jnp.asarray(idiosyncratic_blocks)
        num_countries = mode_matrix.shape[0]
        if mode_matrix.shape != (num_countries, num_countries):
            raise ValueError("mode_matrix must be square")
        if (
            aggregate_block.ndim != 2
            or aggregate_block.shape[0] != aggregate_block.shape[1]
        ):
            raise ValueError("aggregate_block must be square")
        if (
            idiosyncratic_blocks.ndim != 3
            or idiosyncratic_blocks.shape[0] != num_countries - 1
        ):
            raise ValueError("one idiosyncratic block is required per contrast mode")
        n_train, remainder = divmod(aggregate_block.shape[0] - 11, 13)
        if remainder or idiosyncratic_blocks.shape[1:] != (
            12 * n_train + 10,
            12 * n_train + 10,
        ):
            raise ValueError("mode block sizes are inconsistent")
        aggregate_values, aggregate_vectors = jnp.linalg.eigh(
            0.5 * (aggregate_block + aggregate_block.T)
        )
        idiosyncratic_values, idiosyncratic_vectors = jnp.linalg.eigh(
            0.5 * (idiosyncratic_blocks + jnp.swapaxes(idiosyncratic_blocks, 1, 2))
        )
        object.__setattr__(self, "mode_matrix", mode_matrix)
        object.__setattr__(self, "aggregate_eigenvectors", aggregate_vectors)
        object.__setattr__(
            self, "aggregate_eigenvalues", jnp.maximum(aggregate_values, 0.0)
        )
        object.__setattr__(self, "idiosyncratic_eigenvectors", idiosyncratic_vectors)
        object.__setattr__(
            self,
            "idiosyncratic_eigenvalues",
            jnp.maximum(idiosyncratic_values, 0.0),
        )
        object.__setattr__(self, "num_countries", num_countries)
        object.__setattr__(self, "n_train", n_train)

    def apply(self, v, damping, ctx):
        N, T = self.num_countries, self.n_train
        country_coefficient_size = 12 * N * T
        rate_coefficient_start = country_coefficient_size
        free_start = (12 * N + 1) * T
        country_coefficients = v[:country_coefficient_size].reshape(12, N, T)
        rate_coefficients = v[rate_coefficient_start:free_start]
        country_initials = v[free_start : free_start + 10 * N].reshape(10, N)
        rate_initial = v[-1:]
        coefficient_modes = jnp.einsum(
            "tna,nm->tma", country_coefficients, self.mode_matrix
        )
        initial_modes = jnp.einsum("tn,nm->tm", country_initials, self.mode_matrix)
        aggregate = jnp.concatenate(
            [
                coefficient_modes[:, 0, :].reshape(-1),
                rate_coefficients,
                initial_modes[:, 0],
                rate_initial,
            ]
        )
        idiosyncratic = jnp.concatenate(
            [
                jnp.transpose(coefficient_modes[:, 1:, :], (1, 0, 2)).reshape(
                    N - 1, 12 * T
                ),
                initial_modes[:, 1:].T,
            ],
            axis=1,
        )
        aggregate_coefficients = self.aggregate_eigenvectors.T @ aggregate
        aggregate_result = self.aggregate_eigenvectors @ (
            aggregate_coefficients / (self.aggregate_eigenvalues + damping)
        )
        idiosyncratic_coefficients = jnp.einsum(
            "gab,ga->gb", self.idiosyncratic_eigenvectors, idiosyncratic
        )
        idiosyncratic_result = jnp.einsum(
            "gab,gb->ga",
            self.idiosyncratic_eigenvectors,
            idiosyncratic_coefficients / (self.idiosyncratic_eigenvalues + damping),
        )
        aggregate_country_end = 12 * T
        aggregate_rate_end = aggregate_country_end + T
        aggregate_initial_end = aggregate_rate_end + 10
        coefficient_modes_result = jnp.zeros_like(coefficient_modes)
        coefficient_modes_result = coefficient_modes_result.at[:, 0, :].set(
            aggregate_result[:aggregate_country_end].reshape(12, T)
        )
        coefficient_modes_result = coefficient_modes_result.at[:, 1:, :].set(
            idiosyncratic_result[:, :aggregate_country_end]
            .reshape(N - 1, 12, T)
            .transpose(1, 0, 2)
        )
        initial_modes_result = jnp.zeros_like(initial_modes)
        initial_modes_result = initial_modes_result.at[:, 0].set(
            aggregate_result[aggregate_rate_end:aggregate_initial_end]
        )
        initial_modes_result = initial_modes_result.at[:, 1:].set(
            idiosyncratic_result[:, aggregate_country_end:].T
        )
        country_coefficients_result = jnp.einsum(
            "tma,nm->tna", coefficient_modes_result, self.mode_matrix
        )
        country_initials_result = jnp.einsum(
            "tm,nm->tn", initial_modes_result, self.mode_matrix
        )
        return jnp.concatenate(
            [
                country_coefficients_result.reshape(-1),
                aggregate_result[aggregate_country_end:aggregate_rate_end],
                country_initials_result.reshape(-1),
                aggregate_result[aggregate_initial_end:],
            ]
        ).astype(v.dtype)


register_pytree_dataclass(
    TradeGrowthModePreconditioner,
    data_fields=(
        "mode_matrix",
        "aggregate_eigenvectors",
        "aggregate_eigenvalues",
        "idiosyncratic_eigenvectors",
        "idiosyncratic_eigenvalues",
    ),
    meta_fields=("num_countries", "n_train"),
)


@dataclass(frozen=True, eq=False)
class TradeGrowthBalancedPreconditioner(Preconditioner):
    aggregate_block: InitVar[jax.Array]
    idiosyncratic_blocks: InitVar[jax.Array]
    aggregate_cross: InitVar[jax.Array]
    mode_matrix: jax.Array
    aggregate_eigenvectors: jax.Array = field(init=False)
    aggregate_eigenvalues: jax.Array = field(init=False)
    idiosyncratic_eigenvectors: jax.Array = field(init=False)
    idiosyncratic_eigenvalues: jax.Array = field(init=False)
    aggregate_cross_matrix: jax.Array = field(init=False)
    num_countries: int = field(init=False)
    n_train: int = field(init=False)

    def __post_init__(
        self,
        aggregate_block,
        idiosyncratic_blocks,
        aggregate_cross,
    ):
        mode_matrix = jnp.asarray(self.mode_matrix)
        aggregate_block = jnp.asarray(aggregate_block)
        idiosyncratic_blocks = jnp.asarray(idiosyncratic_blocks)
        aggregate_cross = jnp.asarray(aggregate_cross)
        num_countries = mode_matrix.shape[0]
        n_train, remainder = divmod(aggregate_block.shape[0] - 11, 13)
        idiosyncratic_size = 12 * n_train + 10
        if (
            mode_matrix.shape != (num_countries, num_countries)
            or remainder
            or idiosyncratic_blocks.shape
            != (num_countries - 1, idiosyncratic_size, idiosyncratic_size)
            or aggregate_cross.shape
            != (num_countries - 1, aggregate_block.shape[0], idiosyncratic_size)
        ):
            raise ValueError("balanced mode block shapes are inconsistent")
        aggregate_values, aggregate_vectors = jnp.linalg.eigh(
            0.5 * (aggregate_block + aggregate_block.T)
        )
        idiosyncratic_values, idiosyncratic_vectors = jnp.linalg.eigh(
            0.5 * (idiosyncratic_blocks + jnp.swapaxes(idiosyncratic_blocks, 1, 2))
        )
        object.__setattr__(self, "mode_matrix", mode_matrix)
        object.__setattr__(self, "aggregate_eigenvectors", aggregate_vectors)
        object.__setattr__(
            self, "aggregate_eigenvalues", jnp.maximum(aggregate_values, 0.0)
        )
        object.__setattr__(self, "idiosyncratic_eigenvectors", idiosyncratic_vectors)
        object.__setattr__(
            self,
            "idiosyncratic_eigenvalues",
            jnp.maximum(idiosyncratic_values, 0.0),
        )
        object.__setattr__(self, "aggregate_cross_matrix", aggregate_cross)
        object.__setattr__(self, "num_countries", num_countries)
        object.__setattr__(self, "n_train", n_train)

    def apply(self, v, damping, ctx):
        N, T = self.num_countries, self.n_train
        country_coefficient_size = 12 * N * T
        free_start = (12 * N + 1) * T
        country_coefficients = v[:country_coefficient_size].reshape(12, N, T)
        rate_coefficients = v[country_coefficient_size:free_start]
        country_initials = v[free_start : free_start + 10 * N].reshape(10, N)
        rate_initial = v[-1:]
        coefficient_modes = jnp.einsum(
            "tna,nm->tma", country_coefficients, self.mode_matrix
        )
        initial_modes = jnp.einsum("tn,nm->tm", country_initials, self.mode_matrix)
        aggregate = jnp.concatenate(
            [
                coefficient_modes[:, 0, :].reshape(-1),
                rate_coefficients,
                initial_modes[:, 0],
                rate_initial,
            ]
        )
        idiosyncratic = jnp.concatenate(
            [
                jnp.transpose(coefficient_modes[:, 1:, :], (1, 0, 2)).reshape(
                    N - 1, 12 * T
                ),
                initial_modes[:, 1:].T,
            ],
            axis=1,
        )

        def solve_aggregate(rhs):
            coefficients = self.aggregate_eigenvectors.T @ rhs
            return self.aggregate_eigenvectors @ (
                coefficients / (self.aggregate_eigenvalues + damping)
            )

        def solve_idiosyncratic(rhs):
            coefficients = jnp.einsum(
                "gij,gi->gj", self.idiosyncratic_eigenvectors, rhs
            )
            return jnp.einsum(
                "gij,gj->gi",
                self.idiosyncratic_eigenvectors,
                coefficients / (self.idiosyncratic_eigenvalues + damping),
            )

        aggregate_solve = solve_aggregate(aggregate)
        # With A the common block, D the block-diagonal contrast operator,
        # and B their coupling, apply the explicitly SPD balanced inverse
        #
        # [I -A^-1 B; 0 I] diag(A^-1,D^-1) [I 0; -B' A^-1 I].
        #
        # Contrast-to-contrast trade couplings are the only discarded terms.
        idiosyncratic_residual = idiosyncratic - jnp.einsum(
            "gai,a->gi", self.aggregate_cross_matrix, aggregate_solve
        )
        idiosyncratic_result = solve_idiosyncratic(idiosyncratic_residual)
        aggregate_result = aggregate_solve - solve_aggregate(
            jnp.einsum(
                "gai,gi->a",
                self.aggregate_cross_matrix,
                idiosyncratic_result,
            )
        )
        aggregate_country_end = 12 * T
        aggregate_rate_end = aggregate_country_end + T
        aggregate_initial_end = aggregate_rate_end + 10
        coefficient_modes_result = jnp.zeros_like(coefficient_modes)
        coefficient_modes_result = coefficient_modes_result.at[:, 0, :].set(
            aggregate_result[:aggregate_country_end].reshape(12, T)
        )
        coefficient_modes_result = coefficient_modes_result.at[:, 1:, :].set(
            idiosyncratic_result[:, :aggregate_country_end]
            .reshape(N - 1, 12, T)
            .transpose(1, 0, 2)
        )
        initial_modes_result = jnp.zeros_like(initial_modes)
        initial_modes_result = initial_modes_result.at[:, 0].set(
            aggregate_result[aggregate_rate_end:aggregate_initial_end]
        )
        initial_modes_result = initial_modes_result.at[:, 1:].set(
            idiosyncratic_result[:, aggregate_country_end:].T
        )
        country_coefficients_result = jnp.einsum(
            "tma,nm->tna", coefficient_modes_result, self.mode_matrix
        )
        country_initials_result = jnp.einsum(
            "tm,nm->tn", initial_modes_result, self.mode_matrix
        )
        return jnp.concatenate(
            [
                country_coefficients_result.reshape(-1),
                aggregate_result[aggregate_country_end:aggregate_rate_end],
                country_initials_result.reshape(-1),
                aggregate_result[aggregate_initial_end:],
            ]
        ).astype(v.dtype)


register_pytree_dataclass(
    TradeGrowthBalancedPreconditioner,
    data_fields=(
        "mode_matrix",
        "aggregate_eigenvectors",
        "aggregate_eigenvalues",
        "idiosyncratic_eigenvectors",
        "idiosyncratic_eigenvalues",
        "aggregate_cross_matrix",
    ),
    meta_fields=("num_countries", "n_train"),
)
