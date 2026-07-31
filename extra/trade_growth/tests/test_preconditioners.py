from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax.flatten_util import ravel_pytree

from extra.trade_growth.kernel import (
    DataSettings,
    SolverSettings,
    multicountry_trade_growth_kernel,
)
from extra.trade_growth.preconditioners import (
    TradeGrowthEquationBlockPreconditioner,
    TradeGrowthHybridPreconditioner,
    TradeGrowthTimeBlockPreconditioner,
    helmert_country_modes,
    trade_growth_equation_blocks,
    trade_growth_mode_blocks,
    trade_growth_time_blocks,
)
from extra.trade_growth.scaling import draw_trade_growth_parameters
from multicountry_growth import InitialConditionSettings, draw_initial_conditions

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class BalancedModePreconditioner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_countries = 3
        cls.n_train = 6
        parameters = draw_trade_growth_parameters(cls.num_countries, seed=0)
        k_0, a_0 = draw_initial_conditions(
            InitialConditionSettings(seed=0),
            parameters,
            dtype=jnp.float64,
        )
        results, extra = multicountry_trade_growth_kernel(
            k_0=k_0,
            a_0=a_0,
            parameters=parameters,
            data_set=DataSettings(
                train_T=15.0,
                train_points=cls.n_train,
                sampling="front_loaded",
                dense_T=7.5,
                tail_points=2,
                test_T=7.5,
                test_points=13,
                validation_points=25,
                quality_T=15.0,
            ),
            lm_set=SolverSettings(linear_solver="auto", max_steps=100),
        )
        if not results["solution_accepted"]:
            raise RuntimeError(results["rejection_reason"])
        cls.results = results
        cls.extra = extra
        cls.ridge = jnp.asarray(results["final_ridge"])
        cls.preconditioner = extra["build_preconditioner"](
            extra["theta"],
            extra["p"],
            cls.ridge,
        )
        cls.context = SimpleNamespace(lm_state=SimpleNamespace(ridge=cls.ridge))

        theta_flat, unravel_theta = ravel_pytree(extra["theta"])
        jacobian = np.asarray(
            jax.jacrev(
                lambda flat: extra["residual_fn"](
                    unravel_theta(flat),
                    None,
                    extra["p"],
                )
            )(theta_flat)
        )
        metric_size = extra["metric"].size
        jacobian_transpose = jnp.asarray(jacobian.T)
        whitened_jacobian_transpose = np.asarray(
            jnp.concatenate(
                [
                    extra["metric"].factor_solve_transpose(
                        jacobian_transpose[:metric_size],
                        None,
                    ),
                    jacobian_transpose[metric_size:],
                ],
                axis=0,
            )
        )
        normal = whitened_jacobian_transpose @ whitened_jacobian_transpose.T
        normal[np.arange(metric_size), np.arange(metric_size)] += float(cls.ridge)
        cls.normal = 0.5 * (normal + normal.T)
        (
            cls.mode_matrix,
            cls.aggregate_block,
            cls.idiosyncratic_blocks,
            cls.aggregate_cross,
        ) = trade_growth_mode_blocks(
            extra["theta"],
            extra["K"],
            extra["K_tilde"],
            extra["metric"].F,
            extra["p"],
            extra["residual_scale"],
            cls.ridge,
        )

        num_countries = cls.num_countries
        n_train = cls.n_train
        mode_matrix = cls.mode_matrix

        def native_to_modes(v):
            country_size = 12 * num_countries * n_train
            free_start = (12 * num_countries + 1) * n_train
            country_coefficients = v[:country_size].reshape(
                12,
                num_countries,
                n_train,
            )
            rate_coefficients = v[country_size:free_start]
            country_initials = v[free_start : free_start + 10 * num_countries].reshape(
                10, num_countries
            )
            coefficient_modes = jnp.einsum(
                "tna,nm->tma",
                country_coefficients,
                mode_matrix,
            )
            initial_modes = jnp.einsum(
                "tn,nm->tm",
                country_initials,
                mode_matrix,
            )
            aggregate = jnp.concatenate(
                [
                    coefficient_modes[:, 0, :].reshape(-1),
                    rate_coefficients,
                    initial_modes[:, 0],
                    v[-1:],
                ]
            )
            idiosyncratic = jnp.concatenate(
                [
                    jnp.transpose(
                        coefficient_modes[:, 1:, :],
                        (1, 0, 2),
                    ).reshape(num_countries - 1, 12 * n_train),
                    initial_modes[:, 1:].T,
                ],
                axis=1,
            )
            return jnp.concatenate([aggregate, idiosyncratic.reshape(-1)])

        cls.mode_transform = np.asarray(
            jax.vmap(native_to_modes, in_axes=1, out_axes=1)(
                jnp.eye(cls.normal.shape[0], dtype=jnp.float64)
            )
        )

    def inverse_matrix(self, damping):
        identity = jnp.eye(self.normal.shape[0], dtype=jnp.float64)
        inverse = jax.vmap(
            lambda column: self.preconditioner.apply(
                column,
                jnp.asarray(damping),
                self.context,
            ),
            in_axes=1,
            out_axes=1,
        )(identity)
        return np.asarray(inverse)

    def test_helmert_modes_are_orthonormal_and_country_symmetric(self):
        modes = np.asarray(helmert_country_modes(self.num_countries, jnp.float64))
        np.testing.assert_allclose(modes.T @ modes, np.eye(3), atol=2e-15)
        np.testing.assert_allclose(
            modes[:, 0],
            np.ones(3) / np.sqrt(3.0),
            atol=2e-15,
        )
        np.testing.assert_allclose(
            np.sum(modes[:, 1:], axis=0),
            0.0,
            atol=2e-15,
        )

    def test_local_assembly_matches_dense_whitened_normal_blocks(self):
        transform = self.mode_transform
        np.testing.assert_allclose(
            transform @ transform.T,
            np.eye(transform.shape[0]),
            atol=5e-15,
        )
        normal_modes = transform @ self.normal @ transform.T
        aggregate_size = self.aggregate_block.shape[0]
        idiosyncratic_size = self.idiosyncratic_blocks.shape[1]
        np.testing.assert_allclose(
            self.aggregate_block,
            normal_modes[:aggregate_size, :aggregate_size],
            rtol=3e-10,
            atol=3e-11,
        )
        for mode in range(self.num_countries - 1):
            selection = slice(
                aggregate_size + mode * idiosyncratic_size,
                aggregate_size + (mode + 1) * idiosyncratic_size,
            )
            np.testing.assert_allclose(
                self.idiosyncratic_blocks[mode],
                normal_modes[selection, selection],
                rtol=3e-10,
                atol=3e-11,
            )
            np.testing.assert_allclose(
                self.aggregate_cross[mode],
                normal_modes[:aggregate_size, selection],
                rtol=3e-10,
                atol=3e-11,
            )

    def test_apply_is_spd_and_self_adjoint_at_live_dampings(self):
        for damping in (0.0, 1e-8, 1e-4):
            with self.subTest(damping=damping):
                inverse = self.inverse_matrix(damping)
                scale = max(float(np.max(np.abs(inverse))), 1.0)
                self.assertLess(
                    float(np.max(np.abs(inverse - inverse.T))) / scale,
                    1e-7 if damping == 0.0 else 3e-9,
                )
                eigenvalues = np.linalg.eigvalsh(0.5 * (inverse + inverse.T))
                self.assertGreater(eigenvalues[0], 0.0)

    def test_apply_includes_damping_in_every_mode_block(self):
        damping = 1e-4
        aggregate = np.asarray(self.aggregate_block)
        idiosyncratic = np.asarray(self.idiosyncratic_blocks)
        cross = np.concatenate(np.asarray(self.aggregate_cross), axis=1)
        aggregate_inverse = np.linalg.inv(
            aggregate + damping * np.eye(aggregate.shape[0])
        )
        idiosyncratic_inverse = scipy.linalg.block_diag(
            *(
                np.linalg.inv(block + damping * np.eye(block.shape[0]))
                for block in idiosyncratic
            )
        )
        upper_right = -aggregate_inverse @ cross @ idiosyncratic_inverse
        balanced_modes = np.block(
            [
                [
                    aggregate_inverse
                    + aggregate_inverse
                    @ cross
                    @ idiosyncratic_inverse
                    @ cross.T
                    @ aggregate_inverse,
                    upper_right,
                ],
                [upper_right.T, idiosyncratic_inverse],
            ]
        )
        expected = self.mode_transform.T @ balanced_modes @ self.mode_transform
        actual = self.inverse_matrix(damping)
        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        self.assertLess(relative_error, 2e-10)

    def test_preconditioner_reduces_the_positive_spectrum_condition_number(self):
        normal_eigenvalues = np.linalg.eigvalsh(self.normal)
        normal_floor = (
            np.finfo(self.normal.dtype).eps
            * self.normal.shape[0]
            * normal_eigenvalues[-1]
        )
        normal_positive = normal_eigenvalues[normal_eigenvalues > normal_floor]
        normal_condition = normal_positive[-1] / normal_positive[0]
        inverse = self.inverse_matrix(0.0)
        inverse = 0.5 * (inverse + inverse.T)
        inverse_factor = scipy.linalg.cholesky(inverse, lower=True)
        preconditioned = inverse_factor.T @ self.normal @ inverse_factor
        preconditioned_eigenvalues = scipy.linalg.eigvalsh(
            0.5 * (preconditioned + preconditioned.T)
        )
        preconditioned_floor = (
            np.finfo(preconditioned.dtype).eps
            * preconditioned.shape[0]
            * preconditioned_eigenvalues[-1]
        )
        preconditioned_positive = preconditioned_eigenvalues[
            preconditioned_eigenvalues > preconditioned_floor
        ]
        preconditioned_condition = (
            preconditioned_positive[-1] / preconditioned_positive[0]
        )
        self.assertLess(preconditioned_condition, 0.01 * normal_condition)


class MetricDualPreconditioners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_countries = 2
        cls.n_train = 6
        parameters = draw_trade_growth_parameters(cls.num_countries, seed=0)
        k_0, a_0 = draw_initial_conditions(
            InitialConditionSettings(seed=0), parameters, dtype=jnp.float64
        )
        results, extra = multicountry_trade_growth_kernel(
            k_0=k_0,
            a_0=a_0,
            parameters=parameters,
            data_set=DataSettings(
                train_T=15.0,
                train_points=cls.n_train,
                sampling="front_loaded",
                dense_T=7.5,
                tail_points=2,
                test_T=7.5,
                test_points=13,
                validation_points=25,
                quality_T=15.0,
            ),
            lm_set=SolverSettings(
                optimizer="metric_lm",
                metric_eps=1e-7,
                linear_solver="auto",
                preconditioner="hybrid_blocks",
                init_damping=1e-4,
                geodesic_acceleration=False,
                asset_homotopy_steps=0,
                max_steps=100,
            ),
        )
        if not results["solution_accepted"]:
            raise RuntimeError(results["rejection_reason"])
        cls.extra = extra
        metric = extra["metric"]
        theta_flat, unravel_theta = ravel_pytree(extra["theta"])
        jacobian = jax.jacrev(
            lambda flat: extra["residual_fn"](unravel_theta(flat), None, extra["p"])
        )(theta_flat)
        jacobian_transpose = jacobian.T
        whitened_transpose = jnp.concatenate(
            [
                metric.factor_solve_transpose(jacobian_transpose[: metric.size], None),
                jacobian_transpose[metric.size :] / jnp.sqrt(metric.free_scale),
            ],
            axis=0,
        )
        cls.dual = np.asarray(whitened_transpose.T @ whitened_transpose)
        coefficient_blocks, initial_jacobian = trade_growth_time_blocks(
            extra["theta"],
            extra["K"],
            extra["K_tilde"],
            metric.F,
            extra["p"],
            extra["residual_scale"],
            metric.free_scale,
        )
        equation_blocks = trade_growth_equation_blocks(
            extra["theta"],
            extra["K"],
            extra["K_tilde"],
            metric.F,
            extra["p"],
            extra["residual_scale"],
            metric.free_scale,
        )
        cls.coefficient_blocks = np.asarray(coefficient_blocks)
        cls.initial_jacobian = np.asarray(initial_jacobian)
        cls.equation_blocks = np.asarray(equation_blocks)
        cls.time_preconditioner = TradeGrowthTimeBlockPreconditioner(
            coefficient_blocks=coefficient_blocks,
            initial_jacobian=initial_jacobian,
            num_countries=cls.num_countries,
        )
        cls.equation_preconditioner = TradeGrowthEquationBlockPreconditioner(
            blocks=equation_blocks,
            num_countries=cls.num_countries,
        )
        cls.hybrid = TradeGrowthHybridPreconditioner(
            time_blocks=cls.time_preconditioner,
            equation_blocks=cls.equation_preconditioner,
        )

    def test_time_and_equation_assemblies_match_dense_dual_blocks(self):
        size = self.dual.shape[0]
        transform = np.asarray(
            jax.vmap(
                self.time_preconditioner._to_time_order,
                in_axes=1,
                out_axes=2,
            )(jnp.eye(size, dtype=jnp.float64))
        ).reshape(size, size)
        np.testing.assert_allclose(transform @ transform.T, np.eye(size), atol=0.0)
        dual_time = transform @ self.dual @ transform.T
        rows = 12 * self.num_countries + 1
        dual_four = dual_time.reshape(self.n_train, rows, self.n_train, rows)
        for time_index in range(self.n_train):
            expected = self.coefficient_blocks[time_index] + (
                self.initial_jacobian[time_index] @ self.initial_jacobian[time_index].T
            )
            np.testing.assert_allclose(
                expected,
                dual_four[time_index, :, time_index, :],
                rtol=3e-10,
                atol=3e-11,
            )
        for row in range(rows):
            np.testing.assert_allclose(
                self.equation_blocks[row],
                dual_four[:, row, :, row],
                rtol=3e-10,
                atol=3e-11,
            )

    def test_hybrid_apply_is_spd_and_improves_the_damped_spectrum(self):
        size = self.dual.shape[0]
        identity = jnp.eye(size, dtype=jnp.float64)
        context = SimpleNamespace(lm_state=SimpleNamespace(ridge=None))
        damping = 1e-8
        inverse = np.asarray(
            jax.vmap(
                lambda column: self.hybrid.apply(column, jnp.asarray(damping), context),
                in_axes=1,
                out_axes=1,
            )(identity)
        )
        inverse = 0.5 * (inverse + inverse.T)
        eigenvalues = np.linalg.eigvalsh(inverse)
        self.assertGreater(eigenvalues[0], 0.0)
        factor = scipy.linalg.cholesky(inverse, lower=True)
        system = self.dual + damping * np.eye(size)
        preconditioned = factor.T @ system @ factor
        preconditioned_eigenvalues = scipy.linalg.eigvalsh(
            0.5 * (preconditioned + preconditioned.T)
        )
        preconditioned_condition = (
            preconditioned_eigenvalues[-1] / preconditioned_eigenvalues[0]
        )
        self.assertLess(preconditioned_condition, 2e4)
        self.assertLess(preconditioned_condition, 1e-4 * np.linalg.cond(system))


if __name__ == "__main__":
    unittest.main()
