"""Self-contained tests for the trade-growth time-block preconditioner.

The dense dual reference ``D = J M^{-1} J'`` is built directly from the model
physics and the explicit shifted-RKHS metric -- no solver dependency -- so the
assembly gate is validated against an independent construction of the operator.
"""

from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from kernels import integrated_kernel_matrices
from trade_growth.model import (
    STATIC_BLOCK_NAMES,
    TradeGrowthParameters,
    multicountry_trade_dae_residual,
    multicountry_trade_path_values,
)
from trade_growth.preconditioners import (
    assemble_time_diagonal_blocks,
    build_trade_growth_preconditioner_pieces,
    build_trade_growth_time_block_preconditioner,
    residual_time_permutation,
    trade_growth_reference_state,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

METRIC_EPS = 1e-7
NUM_STATIC = len(STATIC_BLOCK_NAMES)


def heterogeneous_inputs(num_countries, seed=0, heterogeneity=0.06):
    """Deterministic heterogeneous parameters matching the offline random regime."""

    keys = jax.random.split(jax.random.key(seed), 10)

    def positive(key, center=1.0):
        return center * jnp.exp(
            heterogeneity * jax.random.normal(key, (num_countries,))
        )

    def share(key, center):
        logit = math.log(center / (1.0 - center))
        return jax.nn.sigmoid(
            logit + heterogeneity * jax.random.normal(key, (num_countries,))
        )

    populations = positive(keys[0])
    trade_noise = jax.random.normal(keys[7], (num_countries, num_countries))
    center = math.log(math.expm1(0.45))
    off = 1.0 + jax.nn.softplus(center + heterogeneity * trade_noise)
    trade_costs = jnp.where(jnp.eye(num_countries, dtype=bool), 1.0, off)
    k_0 = jnp.exp(0.12 * jax.random.normal(keys[8], (num_countries,)))
    free = 0.10 * jax.random.normal(keys[9], (num_countries - 1,))
    aggregate = jnp.concatenate([free, -jnp.sum(free, keepdims=True)])
    a_0 = aggregate / populations

    def host(v):
        return tuple(float(x) for x in np.asarray(v))

    parameters = TradeGrowthParameters(
        populations=host(populations),
        intermediate_productivities=host(positive(keys[1])),
        consumption_productivities=host(positive(keys[2])),
        investment_productivities=host(positive(keys[3])),
        consumption_value_added_shares=host(share(keys[4], 0.75)),
        intermediate_value_added_shares=host(share(keys[5], 0.50)),
        investment_value_added_shares=host(share(keys[6], 0.50)),
        trade_costs=tuple(host(row) for row in np.asarray(trade_costs)),
    )
    return parameters, host(k_0), host(a_0)


def reference_theta(parameters, n_train, reference_state, dtype=jnp.float64):
    num_countries = parameters.num_countries
    zeros = jnp.zeros((num_countries, n_train), dtype=dtype)
    algebraic = jnp.concatenate(
        [
            jnp.zeros(n_train + 2 * num_countries, dtype=dtype),
            jnp.broadcast_to(
                reference_state["log_static_reference"],
                (n_train, NUM_STATIC, num_countries),
            ).reshape(-1),
        ]
    )
    return (zeros, zeros, zeros, zeros, algebraic)


def perturb_theta(theta, seed=1, scale=0.05):
    flat, unravel = ravel_pytree(theta)
    noise = scale * jax.random.normal(
        jax.random.key(seed), flat.shape, dtype=flat.dtype
    )
    return unravel(flat + noise)


def dense_dual_reference(parameters, k_0, a_0, train_data, theta, eps, reference_state):
    """Independent dense ``D = J M^{-1} J'`` from the model physics + explicit metric."""

    num_countries = parameters.num_countries
    n_train = train_data.shape[0]
    K, K_tilde = integrated_kernel_matrices(
        train_data, train_data, "matern52", 1.0, 10.0
    )
    K = 0.5 * (K + K.T)
    k_0_array = reference_state["k_0_array"]
    a_0_array = reference_state["a_0_array"]
    q_reference = reference_state["q_reference"]
    marginal_wealth_reference = reference_state["marginal_wealth_reference"]
    output_scale = reference_state["output_scale"]
    rho = parameters.rho
    flat, unravel = ravel_pytree(theta)

    def residual_from_flat(theta_flat):
        alpha_k, alpha_a, alpha_q, alpha_m, algebraic = unravel(theta_flat)
        rate = algebraic[:n_train]
        q_off = algebraic[n_train : n_train + num_countries]
        m_off = algebraic[n_train + num_countries : n_train + 2 * num_countries]
        log_static = algebraic[n_train + 2 * num_countries :].reshape(
            n_train, NUM_STATIC, num_countries
        )
        log_k = jnp.log(k_0_array) + K_tilde @ alpha_k.T
        a = a_0_array + output_scale * (K_tilde @ alpha_a.T)
        log_q = jnp.log(q_reference) + q_off + K_tilde @ alpha_q.T
        log_m = jnp.log(marginal_wealth_reference) + m_off + K_tilde @ alpha_m.T
        values = multicountry_trade_path_values(
            k=jnp.exp(log_k),
            a=a,
            tobin_q=jnp.exp(log_q),
            marginal_wealth=jnp.exp(log_m),
            interest_rate=rho * (1.0 + rate),
            log_static=log_static,
            dlog_k_dt=K @ alpha_k.T,
            da_dt=output_scale * (K @ alpha_a.T),
            dlog_q_dt=K @ alpha_q.T,
            dlog_marginal_wealth_dt=K @ alpha_m.T,
            parameters=parameters,
        )
        return multicountry_trade_dae_residual(values, log_static, parameters)

    jacobian = jax.jacfwd(residual_from_flat)(flat)
    n_dyn = 4 * num_countries * n_train
    kernel_inverse = jnp.linalg.inv(K + eps * jnp.eye(n_train))
    metric_inverse_dyn = jnp.kron(jnp.eye(4 * num_countries), kernel_inverse)
    dyn = jacobian[:, :n_dyn]
    alg = jacobian[:, n_dyn:]
    dual = dyn @ metric_inverse_dyn @ dyn.T + (1.0 / eps) * (alg @ alg.T)
    return 0.5 * (dual + dual.T), K, K_tilde


class AssemblyGate(unittest.TestCase):
    def _check(self, num_countries, train_points, theta_kind, dampings):
        parameters, k_0, a_0 = heterogeneous_inputs(num_countries, seed=0)
        train_data = jnp.linspace(0.0, 20.0, train_points, dtype=jnp.float64)
        reference_state = trade_growth_reference_state(parameters, k_0, a_0)
        theta = reference_theta(parameters, train_points, reference_state)
        if theta_kind == "perturbed":
            theta = perturb_theta(theta)
        eps = METRIC_EPS

        dual, K, K_tilde = dense_dual_reference(
            parameters, k_0, a_0, train_data, theta, eps, reference_state
        )
        _, full_block, _, _, _, _ = assemble_time_diagonal_blocks(
            theta, K, K_tilde, reference_state, parameters, eps
        )
        full_block = np.asarray(full_block)
        time_perm = np.asarray(
            residual_time_permutation(num_countries, train_points)[0]
        )
        dual = np.asarray(dual)[np.ix_(time_perm, time_perm)]
        block = 12 * num_countries + 1

        for damping in dampings:
            worst = 0.0
            for t in range(train_points):
                sl = slice(t * block, (t + 1) * block)
                reference_block = dual[sl, sl] + damping * np.eye(block)
                mine = full_block[t] + damping * np.eye(block)
                scale = max(np.max(np.abs(reference_block)), 1e-300)
                worst = max(worst, np.max(np.abs(mine - reference_block)) / scale)
            self.assertLess(
                worst,
                1e-8,
                f"N={num_countries} nt={train_points} {theta_kind} "
                f"damping={damping}: rel err {worst:.2e}",
            )

    def test_lm0_n2_nt5(self):
        self._check(2, 5, "reference", (0.0, 1e-4))

    def test_perturbed_n2_nt5(self):
        self._check(2, 5, "perturbed", (0.0, 1e-4))

    def test_lm0_n3_nt5(self):
        self._check(3, 5, "reference", (0.0, 1e-4))


class PreconditionerApply(unittest.TestCase):
    def _build(self, num_countries=2, train_points=5, theta_kind="reference"):
        parameters, k_0, a_0 = heterogeneous_inputs(num_countries, seed=0)
        train_data = jnp.linspace(0.0, 20.0, train_points, dtype=jnp.float64)
        pieces = build_trade_growth_preconditioner_pieces(
            parameters=parameters,
            k_0=k_0,
            a_0=a_0,
            train_data=train_data,
            metric_eps=METRIC_EPS,
            theta=reference_theta(
                parameters,
                train_points,
                trade_growth_reference_state(parameters, k_0, a_0),
            ),
        )
        reference_state = pieces["reference_state"]
        theta = reference_theta(parameters, train_points, reference_state)
        if theta_kind == "perturbed":
            theta = perturb_theta(theta)
        pre = build_trade_growth_time_block_preconditioner(
            theta,
            pieces["K"],
            pieces["K_tilde"],
            reference_state,
            parameters,
            pieces["eps"],
            pieces["time_permutation"],
            pieces["native_permutation"],
        )
        m = (12 * num_countries + 1) * train_points
        return pre, m

    def test_spd_apply(self):
        pre, m = self._build()
        damping = jnp.asarray(1e-4)
        key = jax.random.key(3)
        for i in range(8):
            v = jax.random.normal(jax.random.fold_in(key, i), (m,), dtype=jnp.float64)
            quad = float(v @ pre.dual_preconditioner(v, damping))
            self.assertGreater(
                quad, 0.0, "preconditioner apply must be positive definite"
            )

    def test_self_adjoint_apply(self):
        pre, m = self._build()
        damping = jnp.asarray(1e-4)
        u = jax.random.normal(jax.random.key(4), (m,), dtype=jnp.float64)
        v = jax.random.normal(jax.random.key(5), (m,), dtype=jnp.float64)
        left = float(u @ pre.dual_preconditioner(v, damping))
        right = float(v @ pre.dual_preconditioner(u, damping))
        self.assertLess(abs(left - right) / max(abs(left), 1e-300), 1e-9)

    def test_implicit_is_lambda_zero_path(self):
        pre, m = self._build()
        v = jax.random.normal(jax.random.key(6), (m,), dtype=jnp.float64)
        implicit = pre.implicit_preconditioner(v)
        explicit = pre.dual_preconditioner(
            v, jnp.asarray(pre.implicit_damping, dtype=v.dtype)
        )
        self.assertLess(
            float(jnp.linalg.norm(implicit - explicit))
            / float(jnp.linalg.norm(explicit)),
            1e-12,
        )

    def test_woodbury_matches_dense_inverse(self):
        num_countries, train_points = 2, 5
        parameters, k_0, a_0 = heterogeneous_inputs(num_countries, seed=0)
        train_data = jnp.linspace(0.0, 20.0, train_points, dtype=jnp.float64)
        reference_state = trade_growth_reference_state(parameters, k_0, a_0)
        theta = reference_theta(parameters, train_points, reference_state)
        eps = METRIC_EPS
        dual, K, K_tilde = dense_dual_reference(
            parameters, k_0, a_0, train_data, theta, eps, reference_state
        )
        pre, full_block = build_trade_growth_time_block_preconditioner(
            theta,
            K,
            K_tilde,
            reference_state,
            parameters,
            eps,
            *residual_time_permutation(num_countries, train_points),
            return_blocks=True,
        )
        # dense C3 = blockdiag_time(full D_tt) + off-time-diagonal (1/eps) U_t U_t''
        block = 12 * num_countries + 1
        m = block * train_points
        time_perm = np.asarray(
            residual_time_permutation(num_countries, train_points)[0]
        )
        _, _, offset_u, _, _, _ = assemble_time_diagonal_blocks(
            theta, K, K_tilde, reference_state, parameters, eps
        )
        offset_u = np.asarray(offset_u)
        damping = 1e-4
        c3 = np.zeros((m, m))
        u_full = np.zeros((m, offset_u.shape[2]))
        for t in range(train_points):
            sl = slice(t * block, (t + 1) * block)
            c3[sl, sl] = np.asarray(full_block[t]) + damping * np.eye(block)
            u_full[sl] = offset_u[t]
        global_g = (1.0 / eps) * (u_full @ u_full.T)
        for t in range(train_points):
            sl = slice(t * block, (t + 1) * block)
            global_g[sl, sl] = 0.0
        c3 = c3 + global_g
        c3_inverse = np.linalg.inv(0.5 * (c3 + c3.T))

        native_perm = np.asarray(
            residual_time_permutation(num_countries, train_points)[1]
        )
        probe = jax.random.normal(jax.random.key(7), (m,), dtype=jnp.float64)
        mine = np.asarray(pre.dual_preconditioner(probe, jnp.asarray(damping)))
        # c3 is in time-major order; map the native-order probe/result through the perms
        exact = (c3_inverse @ np.asarray(probe)[time_perm])[native_perm]
        rel = np.linalg.norm(mine - exact) / np.linalg.norm(exact)
        self.assertLess(rel, 1e-6, f"woodbury apply vs dense C3^-1 rel err {rel:.2e}")


if __name__ == "__main__":
    unittest.main()
