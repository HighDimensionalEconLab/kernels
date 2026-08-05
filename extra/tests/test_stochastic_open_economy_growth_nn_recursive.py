import math
import pathlib
import tempfile
import unittest
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from nlls_gram import LMContext, LMState, LMStatus

import open_economy_growth_nn_recursive as deterministic
import stochastic_open_economy_growth_nn_recursive as stochastic
from open_economy_growth import ModelParameters, marginal_product

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def structural_parameters(dtype, *, sigma_z=0.0, sigma_r=0.0, corr_zr=0.0):
    parameters = ModelParameters()
    return stochastic.StructuralParameters.create(
        dtype,
        alpha=parameters.alpha,
        gamma=parameters.gamma,
        delta=parameters.delta,
        preference_rate=parameters.r,
        investment_eta=parameters.eta,
        interest_rate_mean=parameters.interest_rate,
        chi=parameters.chi,
        kappa_z=0.2,
        sigma_z=sigma_z,
        log_z_bar=0.0,
        kappa_r=0.55,
        sigma_r=sigma_r,
        corr_zr=corr_zr,
        output_scale=1.0,
    )


def capital_policy(k):
    return jnp.stack([0.08 + 0.015 * k, 0.01 * (k - 1.0)])


def lifted_capital_policy(state):
    return capital_policy(jnp.exp(state[1]))


def state_policy(state):
    log_z, log_k, asset, rate_gap = state
    k = jnp.exp(log_k)
    investment = 0.08 + 0.015 * k + 0.005 * jnp.tanh(log_z) - 0.02 * rate_gap
    current_account = 0.01 * (k - 1.0) - 0.015 * asset + 0.02 * log_z - 0.1 * rate_gap
    return jnp.stack([investment, current_account])


class StochasticOpenEconomyRecursiveTests(unittest.TestCase):
    def test_default_policy_is_overparameterized_and_asset_flat(self):
        dtype = jnp.float64
        settings = stochastic.SequentialPolicySettings(
            log_z_scale=0.1,
            rate_scale=0.02,
            investment_anchor=0.06,
        )
        model = stochastic.SequentialPolicyNN(
            settings,
            dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        _, theta, _ = nnx.split(model, nnx.Param, ...)
        parameter_count = sum(leaf.size for leaf in jax.tree.leaves(theta))
        states = jnp.asarray(
            [
                [-0.1, math.log(0.5), -1.0, -0.02],
                [0.0, math.log(1.0), 0.0, 0.0],
                [0.1, math.log(2.0), 1.0, 0.02],
            ],
            dtype=dtype,
        )
        policies = jax.vmap(model)(states)

        self.assertEqual(parameter_count, 2_338)
        np.testing.assert_array_equal(policies[:, 1], jnp.zeros(3, dtype=dtype))
        # The pre-scaled investment head keeps the near-zero draw flat at the
        # anchor on the output_scale scale, not the anchor scale.
        self.assertLess(float(jnp.ptp(policies[:, 0]) / settings.output_scale), 0.02)

    def test_world_rate_ar1_mapping_matches_stationary_variance(self):
        rho = stochastic.WORLD_RATE_AR1_RHO
        innovation_sd = stochastic.WORLD_RATE_AR1_INNOVATION_SD
        kappa = stochastic.WORLD_RATE_KAPPA
        sigma = stochastic.WORLD_RATE_SIGMA

        self.assertAlmostEqual(math.exp(-kappa), rho)
        self.assertAlmostEqual(
            sigma**2 / (2.0 * kappa), innovation_sd**2 / (1.0 - rho**2)
        )

    def test_zero_diffusion_nests_deterministic_sequential_residual(self):
        dtype = jnp.float64
        model = ModelParameters()
        deterministic_parameters = deterministic.DynamicModelParameters.from_model(
            model, dtype
        )
        stochastic_parameters = structural_parameters(dtype)

        for capital, asset in ((0.8, -0.2), (1.0, 0.0), (1.2, 0.3)):
            with self.subTest(capital=capital, asset=asset):
                old = deterministic.policy_values(
                    capital_policy,
                    jnp.asarray(capital, dtype=dtype),
                    jnp.asarray(asset, dtype=dtype),
                    deterministic_parameters,
                )
                state = jnp.asarray([0.0, math.log(capital), asset, 0.0], dtype=dtype)
                residual = stochastic.euler_residual_at_state(
                    lifted_capital_policy, state, stochastic_parameters
                )[0]
                np.testing.assert_allclose(
                    residual,
                    jnp.stack([old["c_euler"] / old["c"], old["q_euler"]]),
                    rtol=2e-13,
                    atol=2e-13,
                )

    def test_directional_generator_matches_dense_hessian(self):
        p = structural_parameters(
            jnp.float64, sigma_z=0.05, sigma_r=0.025, corr_zr=-0.3
        )
        state = jnp.asarray([0.03, -0.05, 0.1, 0.01], dtype=jnp.float64)
        residual, diffusion_terms, _ = stochastic.euler_residual_at_state(
            state_policy, state, p
        )
        drift = stochastic.state_drift(state_policy, state, p)
        directions = stochastic.diffusion_directions(p)
        covariance_matrix = directions.T @ directions

        def logs(candidate):
            return stochastic.log_policy_quantities(state_policy, candidate, p)

        gradient = jax.jacrev(logs)(state)
        hessian = jax.jacfwd(jax.jacrev(logs))(state)
        generator = gradient @ drift + 0.5 * jnp.einsum(
            "ij,kij->k", covariance_matrix, hessian
        )
        log_covariance = gradient @ covariance_matrix @ gradient.T
        values = stochastic.policy_values(state_policy, state, p)
        investment_rate = values["x"] / values["k"]
        q_rhs = (
            values["interest_rate"]
            + p.delta
            - (1.0 - p.investment_eta) * p.chi * investment_rate**p.investment_eta
            - marginal_product(values["k"], values["z"], p) / values["q"]
        )
        dense_residual = jnp.stack(
            [
                generator[0]
                - 0.5 * p.gamma * log_covariance[0, 0]
                - (values["interest_rate"] - p.preference_rate) / p.gamma,
                generator[1]
                + 0.5 * log_covariance[1, 1]
                - p.gamma * log_covariance[1, 0]
                - q_rhs,
            ]
        )
        drift_generator = gradient @ drift
        dense_diffusion_terms = dense_residual - jnp.stack(
            [
                drift_generator[0]
                - (values["interest_rate"] - p.preference_rate) / p.gamma,
                drift_generator[1] - q_rhs,
            ]
        )

        np.testing.assert_allclose(residual, dense_residual, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(
            diffusion_terms, dense_diffusion_terms, rtol=2e-13, atol=2e-13
        )

    def test_correlated_noise_transform_has_requested_covariance(self):
        key = jax.random.key(4)
        increments = jax.random.normal(key, (200_000, 4), dtype=jnp.float64)
        transformed = stochastic.correlate_noise(increments, jnp.asarray(-0.4))
        sample_correlation = np.corrcoef(
            np.asarray(transformed[:, 0]), np.asarray(transformed[:, 3])
        )[0, 1]

        self.assertAlmostEqual(sample_correlation, -0.4, delta=0.01)

    def test_transversality_conditions_use_path_expectations(self):
        dtype = jnp.float64
        p = structural_parameters(dtype)
        consumption = jnp.asarray(1.5, dtype=dtype)
        investment_rate = jnp.asarray(0.1, dtype=dtype)
        times = jnp.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=dtype)
        q = investment_rate ** (1.0 - p.investment_eta) / (p.chi * p.investment_eta)
        levels = {
            "c": jnp.full_like(times, consumption),
            "a": jnp.asarray([[0.0, 1.0], [0.0, -1.0]], dtype=dtype),
            "q": jnp.full_like(times, q),
            "k": jnp.ones_like(times),
        }
        values = {**levels, **stochastic.transversality_terms(levels, times, p)}
        statistics = stochastic.transversality_statistics(values)
        discount_mu = jnp.exp(-2.0 * p.preference_rate) * consumption ** (-p.gamma)

        np.testing.assert_allclose(
            statistics["expected_absolute_asset_tvc"],
            discount_mu,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            statistics["expected_capital_tvc"],
            discount_mu * q,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            statistics["signed_expected_asset_tvc"], 0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            statistics["absolute_asset_tvc_mc_se"], 0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            statistics["signed_asset_tvc_mc_se"], discount_mu, atol=2e-14
        )
        np.testing.assert_allclose(statistics["capital_tvc_mc_se"], 0.0, atol=2e-14)
        self.assertTrue(
            bool(
                jnp.isnan(
                    stochastic.monte_carlo_standard_error(jnp.ones(1, dtype=dtype))
                )
            )
        )
        self.assertEqual(values["absolute_asset_tvc"].dtype, dtype)

    def test_single_ic_default_and_additive_asset_spread(self):
        dtype = jnp.float64
        key = jax.random.key(11)
        center = jnp.asarray([0.0, 0.0, -0.8, 0.0], dtype=dtype)
        pinned = stochastic.standardized_initial_states(
            key, center, jnp.zeros(4, dtype=dtype), 4
        )
        np.testing.assert_array_equal(pinned, jnp.tile(center, (4, 1)))

        # Additive semantics: a positive asset spread at a zero asset center
        # must produce a nondegenerate cloud (a multiplicative spread would
        # collapse it to zero and leave the CA head unidentified).
        zero_center = jnp.zeros(4, dtype=dtype)
        scales = jnp.asarray([0.0, 0.0, 0.5, 0.0], dtype=dtype)
        spread = stochastic.standardized_initial_states(key, zero_center, scales, 8)
        np.testing.assert_array_equal(spread[:, 0], jnp.zeros(8, dtype=dtype))
        np.testing.assert_array_equal(spread[:, 1], jnp.zeros(8, dtype=dtype))
        np.testing.assert_array_equal(spread[:, 3], jnp.zeros(8, dtype=dtype))
        np.testing.assert_allclose(
            jnp.sqrt(jnp.mean(spread[:, 2] ** 2)), 0.5, rtol=1e-12
        )

    def test_save_load_round_trip_is_bit_identical(self):
        dtype = jnp.float64
        settings = stochastic.SequentialPolicySettings(
            width=8,
            depth=2,
            log_z_scale=0.11,
            log_k_center=0.2,
            asset_center=-0.5,
            asset_scale=1.3,
            rate_scale=0.02,
            output_scale=1.4,
            investment_anchor=0.06,
        )
        model = stochastic.SequentialPolicyNN(settings, dtype=dtype, rngs=nnx.Rngs(3))
        states = jnp.asarray(
            [
                [-0.05, -0.2, -1.5, -0.01],
                [0.0, 0.0, -0.5, 0.0],
                [0.08, 0.3, 0.7, 0.02],
            ],
            dtype=dtype,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "policy.npz"
            stochastic.save_sequential_policy(
                path,
                model,
                settings,
                structural={"alpha": 0.33},
                initial_state={"k_0": 1.0, "a_0": -0.8, "z_0": 1.0, "rate_gap_0": 0.0},
                extra={"model": "test", "solver_accepted": True},
            )
            loaded, meta = stochastic.load_sequential_policy(path)

        self.assertEqual(meta["settings"], asdict(settings))
        self.assertEqual(meta["dtype"], "float64")
        self.assertTrue(meta["solver_accepted"])
        original_params = jax.tree.leaves(nnx.state(model, nnx.Param).to_pure_dict())
        loaded_params = jax.tree.leaves(nnx.state(loaded, nnx.Param).to_pure_dict())
        for original, reloaded in zip(original_params, loaded_params, strict=True):
            np.testing.assert_array_equal(original, reloaded)
        np.testing.assert_array_equal(jax.vmap(model)(states), jax.vmap(loaded)(states))

    def test_problem_cache_key_tracks_the_problem_only(self):
        parameters = ModelParameters()

        def config(**overrides):
            values = dict(
                kappa_z=0.2,
                sigma_z=0.03,
                z_bar=1.0,
                kappa_r=0.55,
                sigma_r=0.01,
                corr_zr=0.0,
                k_0=1.0,
                a_0=-0.8,
                z_0=1.0,
                rate_gap_0=0.0,
                use_float64=True,
                num_train_trajectories=4,
                train_variances=(0.0, 0.0, 0.0, 0.0),
            )
            values.update(overrides)
            return stochastic.problem_cache_config(parameters, **values)

        base_key = stochastic.policy_cache_key(config())
        self.assertEqual(base_key, stochastic.policy_cache_key(config()))
        self.assertNotEqual(base_key, stochastic.policy_cache_key(config(sigma_r=0.02)))
        self.assertNotEqual(
            base_key,
            stochastic.policy_cache_key(config(train_variances=(0.0, 0.0, 0.25, 0.0))),
        )
        # Acceptance tolerances are not inputs of the problem hash, so a
        # tolerance change can never invalidate a cached policy.
        self.assertNotIn("tolerance", repr(config()))

    def test_float32_collocation_vjp_and_epoch_boundary_status(self):
        dtype = jnp.float32
        data_set = stochastic.DataSettings(
            train_points=3, test_points=3, train_T=0.5, test_T=0.5
        )
        p = structural_parameters(dtype, sigma_z=0.02, sigma_r=0.01)
        settings = stochastic.SequentialPolicySettings(
            width=4,
            depth=1,
            hidden_activation="tanh",
            log_z_scale=0.1,
            rate_scale=0.02,
            investment_anchor=0.06,
        )
        model = stochastic.SequentialPolicyNN(
            settings,
            dtype=dtype,
            rngs=nnx.Rngs(0),
        )
        graphdef, theta, fixed_state = nnx.split(model, nnx.Param, ...)
        solver = stochastic.SDE_SOLVERS["sra1"]
        initial_states = jnp.zeros((2, 4), dtype=dtype)
        noise = stochastic.sample_trajectory_noise(
            solver,
            jax.random.key(1),
            initial_states.shape[0],
            data_set.train_points - 1,
            data_set.train_T / (data_set.train_points - 1),
            dtype,
            4,
        )
        args = stochastic.ResidualArgs(
            graphdef=graphdef,
            fixed_state=fixed_state,
            initial_states=initial_states,
            initial_center=jnp.zeros(4, dtype=dtype),
            initial_scales=jnp.zeros(4, dtype=dtype),
            noise=noise,
            epoch_key=jax.random.key(2),
            restart_key=jax.random.key(5),
            epoch=jnp.asarray(0, dtype=jnp.int32),
            shock_scales=jnp.asarray([0.5, 1.0], dtype=dtype),
            shock_scale=jnp.asarray(0.5, dtype=dtype),
            train_loss_threshold=jnp.asarray(1e-6, dtype=dtype),
            epoch_steps=jnp.asarray(2, dtype=jnp.int32),
            total_steps=jnp.asarray(4, dtype=jnp.int32),
            solver=solver,
            data_set=data_set,
        )
        residual = stochastic.residual_fn(theta, args, p)
        _, pullback = jax.vjp(
            lambda value: stochastic.residual_fn(value, args, p), theta
        )
        gradient = pullback(jnp.ones_like(residual))[0]

        self.assertEqual(residual.shape, (12,))
        self.assertEqual(residual.dtype, dtype)
        self.assertEqual(
            {leaf.dtype for leaf in jax.tree.leaves(gradient)}, {jnp.dtype(dtype)}
        )
        residual_jaxpr = str(
            jax.make_jaxpr(lambda value: stochastic.residual_fn(value, args, p))(theta)
        )
        vjp_jaxpr = str(
            jax.make_jaxpr(
                lambda value, cotangent: jax.vjp(
                    lambda candidate: stochastic.residual_fn(candidate, args, p),
                    value,
                )[1](cotangent)[0]
            )(theta, jnp.ones_like(residual))
        )
        self.assertNotIn("f64", residual_jaxpr)
        self.assertNotIn("f64", vjp_jaxpr)

        # Interior epoch boundary: the callback steps the ladder to full
        # shock scale and declares convergence on the fresh batch under the
        # (accepting) threshold.
        accepting_args = args.replace(
            train_loss_threshold=jnp.asarray(1e6, dtype=dtype)
        )
        lm_state = LMState(damping=jnp.asarray(1e-3, dtype=dtype))
        context = LMContext(
            step=jnp.asarray(2, dtype=jnp.int32),
            x=theta,
            x_old=theta,
            lm_state=lm_state,
            lm_state_old=lm_state,
            initial_lm_state=lm_state,
            args=accepting_args,
            p=p,
            user_state=None,
            info=None,
        )
        action = stochastic.EpochCallback(
            num_train_trajectories=initial_states.shape[0],
            n_steps=data_set.train_points - 1,
            dt=data_set.train_T / (data_set.train_points - 1),
            epoch_early_stopping=True,
            resample_initial_states=False,
        )(context)

        self.assertTrue(bool(action.stop))
        self.assertEqual(int(action.status), LMStatus.CONVERGED)
        self.assertEqual(float(action.args.shock_scale), 1.0)


if __name__ == "__main__":
    unittest.main()
