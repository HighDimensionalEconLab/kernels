import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from tinydiffeq import SaveAt

from neoclassical_growth_nn_recursive import (
    DataSettings,
    InvestmentPolicySettings,
    StoppingCriteria,
    StructuralParameters,
    batched_rollout,
    collocation_log_states,
    collocation_residuals,
    euler_residuals,
    neoclassical_growth_nn_recursive,
    production,
    resolve_adaptive_tolerances,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


@struct.dataclass
class SharePolicy:
    logit: jax.Array
    slope: jax.Array

    def __call__(self, log_state, p):
        share = jax.nn.sigmoid(self.logit + self.slope * (log_state[1] - log_state[0]))
        z, k = jnp.exp(log_state)
        return production(z, k, p) * share


@struct.dataclass
class BalancedGrowthPolicy:
    investment_growth: jax.Array

    def __call__(self, log_state, p):
        del p
        return self.investment_growth * jnp.exp(log_state[1])


def parameters(*, z_0=(1.0,), k_0=(1.0,), g=0.02, dtype=jnp.float64):
    return StructuralParameters.create(
        dtype,
        a=1 / 3,
        delta=0.1,
        rho_hat=0.11,
        sigma_crra=1.0,
        g=g,
        z_0=jnp.asarray(z_0),
        k_0=jnp.asarray(k_0),
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        kappa_thr=jnp.inf,
    )


class RecursiveGrowthTsit5Tests(unittest.TestCase):
    def test_small_nnx_lm_adaptive_smoke(self):
        results, extras = neoclassical_growth_nn_recursive(
            g=0.02,
            verbose=False,
            data_set=DataSettings(
                ode_solver="tsit5",
                train_T=8.0,
                train_steps=16,
                train_points=17,
                adaptive_attempt_budget=32,
                train_tvc_T=20.0,
                tvc_steps=32,
                test_T=4.0,
                num_test_trajectories=2,
                benchmark_T=40.0,
                benchmark_points=120,
                policy_grid_size=20,
            ),
            nn_set=InvestmentPolicySettings(width=4, depth=1),
            stop_set=StoppingCriteria(
                max_steps=80,
                train_loss_threshold=1e-8,
            ),
        )

        self.assertTrue(results["solution_accepted"], results["stopping_reason"])
        self.assertTrue(results["collocation_path_ok"])
        self.assertTrue(results["test_path_ok"])
        self.assertEqual(results["point_rule"], "adaptive_accepted_steps")
        self.assertLess(
            results["num_collocation_states"],
            results["collocation_capacity_per_trajectory"],
        )
        self.assertTrue(extras["train_df"]["accepted"].all())

    def test_initial_step_must_respect_controller_bounds(self):
        with self.assertRaisesRegex(ValueError, "adaptive_dt_0 must lie"):
            neoclassical_growth_nn_recursive(
                verbose=False,
                data_set=DataSettings(
                    ode_solver="tsit5",
                    adaptive_dt_0=1.0,
                    adaptive_dt_max=0.5,
                ),
            )

    def test_float32_uses_safe_tolerances_without_dtype_promotion(self):
        p = parameters(dtype=jnp.float32)
        policy = SharePolicy(
            jnp.asarray(-1.0, jnp.float32), jnp.asarray(0.15, jnp.float32)
        )
        data_set = DataSettings(
            ode_solver="tsit5",
            train_T=10.0,
            adaptive_attempt_budget=32,
        )
        resolved = resolve_adaptive_tolerances(data_set, jnp.float32)

        residual = collocation_residuals(policy, p, data_set)
        _, _, solution = collocation_log_states(policy, p, data_set)

        self.assertIsNone(data_set.adaptive_rtol)
        self.assertEqual(resolved.adaptive_rtol, 1e-4)
        self.assertEqual(resolved.adaptive_atol, 1e-6)
        self.assertEqual(residual.dtype, jnp.float32)
        self.assertTrue(bool(solution.ok[0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(residual))))

    def test_accepted_step_contract_and_vmap_parity(self):
        p = parameters(z_0=(1.0, 1.2), k_0=(0.8, 1.4))
        policy = SharePolicy(jnp.asarray(-1.0), jnp.asarray(0.15))
        data_set = DataSettings(
            ode_solver="tsit5",
            train_T=8.0,
            adaptive_attempt_budget=32,
            adaptive_dt_0=0.5,
        )

        states, active, solution = collocation_log_states(policy, p, data_set)
        jax.block_until_ready(solution)
        self.assertEqual(states.shape, (2, 33, 2))
        self.assertEqual(active.shape, (2, 33))
        self.assertTrue(np.all(np.asarray(solution.ok)))

        initial = jnp.log(jnp.stack([p.z_0, p.k_0], axis=1))
        loop_solutions = [
            batched_rollout(
                policy,
                initial[index : index + 1],
                data_set.train_T,
                data_set.adaptive_attempt_budget,
                data_set.adaptive_dt_0,
                p,
                data_set,
                SaveAt(steps=True),
                training=True,
            )
            for index in range(initial.shape[0])
        ]
        for index, loop_solution in enumerate(loop_solutions):
            lane_active = np.asarray(active[index])
            self.assertTrue(np.all(lane_active[:-1] >= lane_active[1:]))
            self.assertEqual(
                int(lane_active.sum()), int(solution.num_accepted[index]) + 1
            )
            active_times = np.asarray(solution.ts[index])[lane_active]
            self.assertEqual(active_times[0], 0.0)
            self.assertAlmostEqual(active_times[-1], data_set.train_T)
            self.assertTrue(np.all(np.diff(active_times) > 0.0))
            self.assertGreater(np.ptp(np.diff(active_times)), 1e-8)
            np.testing.assert_array_equal(
                np.asarray(loop_solution.accepted[0]), lane_active
            )
            np.testing.assert_allclose(
                np.asarray(loop_solution.ts[0]), np.asarray(solution.ts[index])
            )
            np.testing.assert_allclose(
                np.asarray(loop_solution.xs[0]),
                np.asarray(solution.xs[index]),
                rtol=2e-6,
                atol=1e-10,
            )

    def test_padding_is_inert_and_normalization_is_global(self):
        p = parameters(z_0=(1.0, 1.2), k_0=(0.8, 1.4))
        policy = SharePolicy(jnp.asarray(-1.0), jnp.asarray(0.15))
        data_set = DataSettings(
            ode_solver="tsit5",
            train_T=8.0,
            adaptive_attempt_budget=32,
            adaptive_dt_0=0.5,
        )

        states, active, _ = collocation_log_states(policy, p, data_set)
        residual = collocation_residuals(policy, p, data_set)
        raw = euler_residuals(policy, states.reshape(-1, 2), p).reshape(active.shape)

        np.testing.assert_array_equal(np.asarray(residual[~active]), 0.0)
        expected_squared_norm = jnp.sum(jnp.where(active, raw**2, 0.0)) / jnp.sum(
            active
        )
        np.testing.assert_allclose(
            np.asarray(jnp.sum(residual**2)),
            np.asarray(expected_squared_norm),
            rtol=2e-14,
            atol=2e-14,
        )

    def test_budget_starvation_is_a_loud_failure(self):
        p = parameters()
        policy = SharePolicy(jnp.asarray(-1.0), jnp.asarray(0.15))
        data_set = DataSettings(
            ode_solver="tsit5",
            train_T=40.0,
            adaptive_attempt_budget=1,
            adaptive_dt_0=0.5,
        )

        _, _, solution = collocation_log_states(policy, p, data_set)
        residual = collocation_residuals(policy, p, data_set)
        self.assertFalse(bool(solution.ok[0]))
        self.assertTrue(np.all(np.isinf(np.asarray(residual))))

    def test_accepted_states_support_jvp_vjp_and_reverse_over_forward(self):
        p = parameters()
        data_set = DataSettings(
            ode_solver="tsit5",
            train_T=4.0,
            adaptive_attempt_budget=24,
            adaptive_dt_0=0.5,
        )

        def residual_from_logit(logit):
            policy = SharePolicy(logit, jnp.asarray(0.15))
            return collocation_residuals(policy, p, data_set).reshape(-1)

        logit = jnp.asarray(-1.0)
        direction = jnp.asarray(0.3)
        primal, tangent = jax.jvp(residual_from_logit, (logit,), (direction,))
        cotangent = jnp.linspace(-0.5, 0.5, primal.size)
        _, pullback = jax.vjp(residual_from_logit, logit)
        transpose_direction = pullback(cotangent)[0]
        np.testing.assert_allclose(
            np.asarray(jnp.vdot(cotangent, tangent)),
            np.asarray(direction * transpose_direction),
            rtol=2e-12,
            atol=2e-12,
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(tangent))))

        def objective(value):
            residual = residual_from_logit(value)
            return 0.5 * jnp.vdot(residual, residual)

        _, curvature = jax.jvp(jax.grad(objective), (logit,), (direction,))
        self.assertTrue(bool(jnp.isfinite(curvature)))

        def states_from_initial_conditions(log_initial_conditions):
            local_p = p.replace(
                z_0=jnp.exp(log_initial_conditions[:1]),
                k_0=jnp.exp(log_initial_conditions[1:]),
            )
            policy = SharePolicy(logit, jnp.asarray(0.15))
            states, _, _ = collocation_log_states(policy, local_p, data_set)
            return states

        _, initial_condition_tangent = jax.jvp(
            states_from_initial_conditions,
            (jnp.asarray([0.0, 0.0]),),
            (jnp.asarray([0.2, -0.1]),),
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(initial_condition_tangent))))

    def test_balanced_growth_is_exact_on_the_adaptive_mesh(self):
        p = parameters()
        a, delta, rho_hat, g = 1 / 3, 0.1, 0.11, 0.02
        kappa_ss = ((delta + rho_hat + g) / a) ** (1.0 / (a - 1.0))
        p = parameters(z_0=(1.3,), k_0=(1.3 * kappa_ss,), g=g)
        policy = BalancedGrowthPolicy(jnp.asarray(delta + g))
        data_set = DataSettings(
            ode_solver="tsit5",
            train_T=20.0,
            adaptive_attempt_budget=24,
            adaptive_dt_0=0.25,
        )

        states, active, solution = collocation_log_states(policy, p, data_set)
        active_states = states[0][active[0]]
        z, k = jnp.moveaxis(jnp.exp(active_states), -1, 0)
        residual = euler_residuals(policy, active_states, p)

        self.assertTrue(bool(solution.ok[0]))
        np.testing.assert_allclose(np.asarray(k / z), kappa_ss, rtol=2e-13)
        np.testing.assert_allclose(np.asarray(residual), 0.0, atol=3e-14)


if __name__ == "__main__":
    unittest.main()
