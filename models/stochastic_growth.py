from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from nlls_gram import DrawNNXModule
from tinydiffeq import SaveAt, solve_sde

from models.kernels import rbf_state_features, rbf_state_gram, state_features
from models.stochastic_growth_benchmark import stochastic_growth_benchmark
from models.utilities import (
    ACTIVATIONS,
    DOMAIN_EPS,
    unit_softplus,
)
from models.utilities import (
    NEOCLASSICAL_GROWTH_CONSTANTS as CONSTANTS,
)

# Stochastic neoclassical growth on the state (log z, log k): OU log-TFP
# deviation around log z_bar with smooth Cobb-Douglas technology, exactly the
# paper's F(k, z) = e^z k^a written here with z in levels.


@struct.dataclass
class StructuralParameters:
    a: jax.Array
    delta: jax.Array
    rho_hat: jax.Array
    sigma_crra: jax.Array
    eta: jax.Array
    sigma_z: jax.Array
    log_z_bar: jax.Array
    z_0: jax.Array
    k_0: jax.Array

    @classmethod
    def create(cls, dtype, **kwargs):
        return cls(
            **{key: jnp.asarray(value, dtype=dtype) for key, value in kwargs.items()}
        )


@dataclass(frozen=True)
class DataSettings:
    # Fixed-step SDE simulation on the integrator's own states: n points from
    # n - 1 steps including the initial condition, nothing interpolated.
    train_points: int = 65
    test_points: int = 41
    tvc_steps: int = 256
    train_T: float = CONSTANTS.train_T
    train_tvc_T: float = CONSTANTS.transversality_T
    test_T: float = CONSTANTS.test_T
    policy_grid_size: int = 100
    benchmark_z_points: int = 161
    benchmark_k_points: int = 1201
    benchmark_z_sd_span: float = 5.0
    benchmark_k_lo: float = 0.4
    benchmark_k_hi: float | None = None


def production(z, k, p):
    return z * k**p.a


def mpk(z, k, p):
    return p.a * z * k ** (p.a - 1.0)


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return production(z, k, p) - policy(log_state, p)


def log_state_drift(policy, log_state, p):
    log_z, log_k = log_state
    k = jnp.exp(log_k)
    investment = policy(log_state, p)
    return jnp.stack([p.eta * (p.log_z_bar - log_z), investment / k - p.delta])


def log_state_diffusion(log_state, p):
    return jnp.stack([p.sigma_z, jnp.zeros_like(p.sigma_z)])


# The problem half of the two-part artifact key, shared verbatim with any
# consumer that looks up a saved policy for the same problem (e.g. the
# stochastic kernel-recursive collocation generator).
def problem_cache_config(
    a, delta, rho_hat, sigma_crra, eta, sigma_z, z_bar, z_0, k_0, use_float64, ic_set
):
    return {
        "technology": "hicks",
        "a": a,
        "delta": delta,
        "rho_hat": rho_hat,
        "sigma_crra": sigma_crra,
        "eta": eta,
        "sigma_z": sigma_z,
        "z_bar": z_bar,
        "z_0": z_0,
        "k_0": k_0,
        "use_float64": use_float64,
        "train_ic": {
            "num_train_trajectories": ic_set.num_train_trajectories,
            "train_log_z_variance": ic_set.train_log_z_variance,
            "train_log_k_variance": ic_set.train_log_k_variance,
        },
    }


# Stochastic Euler equation (the differentiated HJBE) along d log z =
# eta (log_z_bar - log z) dt + sigma_z dW, with l = log c:
#   grad(l) . mu + (sigma_z^2/2)(l_zz - sigma_crra l_z^2)
#     = (mpk - delta - rho_hat) / sigma_crra
# The Ito terms come from the drift of u'(c) = exp(-sigma_crra l); sigma_z = 0
# recovers the deterministic residual exactly.
def euler_residual_terms(policy, log_states, p):
    e_z = jnp.asarray([1.0, 0.0], dtype=log_states.dtype)

    def scalar_terms(log_state):
        state_drift = log_state_drift(policy, log_state, p)

        def log_consumption(state):
            return jnp.log(consumption(policy, state, p))

        log_c, dlog_c_dt = jax.jvp(log_consumption, (log_state,), (state_drift,))
        l_z, l_zz = jax.jvp(
            lambda state: jax.jvp(log_consumption, (state,), (e_z,))[1],
            (log_state,),
            (e_z,),
        )
        diffusion_term = 0.5 * p.sigma_z**2 * (l_zz - p.sigma_crra * l_z**2)
        z, k = jnp.exp(log_state)
        # The zero-scaled primal keeps the residual NaN outside the model's
        # domain (c <= 0) instead of quietly extending the Euler equation.
        domain_guard = 0.0 * log_c
        residual = (
            dlog_c_dt
            + diffusion_term
            + domain_guard
            - (mpk(z, k, p) - p.delta - p.rho_hat) / p.sigma_crra
        )
        return residual, diffusion_term

    return jax.vmap(scalar_terms)(log_states)


def euler_residuals(policy, log_states, p):
    return euler_residual_terms(policy, log_states, p)[0]


def log_state_drift_field(log_state, _, args, p):
    policy, structural = p
    return log_state_drift(policy, log_state, structural)


def log_state_diffusion_field(log_state, _, args, p):
    _, structural = p
    return log_state_diffusion(log_state, structural)


@partial(jax.jit, static_argnames=("T", "n_steps"))
def simulate(solver, policy, initial_log_states, noise, T, n_steps, p):
    def solve_one(initial_log_state, trajectory_noise):
        return solve_sde(
            log_state_drift_field,
            log_state_diffusion_field,
            solver,
            0.0,
            T,
            initial_log_state,
            noise=trajectory_noise,
            n_steps=n_steps,
            save_at=SaveAt(steps=True),
            p=(policy, p),
        )

    return jax.vmap(solve_one)(initial_log_states, noise)


@partial(jax.jit, static_argnames=("data_set",))
def simulate_train(solver, policy, initial_log_states, noise, p, data_set):
    return simulate(
        solver,
        policy,
        initial_log_states,
        noise,
        data_set.train_T,
        data_set.train_points - 1,
        p,
    )


# Natural-log transversality certificate at the terminal date T, normalized
# by the initial-date certificate, per initial state: the saddle path sits
# far negative, a capital bubble drives it positive. A failed simulation is a
# loud +inf.
def log_tvc_residual(solver, policy, initial_log_states, noise, T, n_steps, p):
    solution = simulate(solver, policy, initial_log_states, noise, T, n_steps, p)
    terminal_log_states = solution.xs[:, -1, :]
    c_T = jax.vmap(lambda state: consumption(policy, state, p))(terminal_log_states)
    k_T = jnp.exp(terminal_log_states[:, 1])
    c_0 = jax.vmap(lambda state: consumption(policy, state, p))(initial_log_states)
    k_0 = jnp.exp(initial_log_states[:, 1])
    log_tvc = -p.rho_hat * T - p.sigma_crra * jnp.log(c_T) + jnp.log(k_T)
    initial_log_tvc = -p.sigma_crra * jnp.log(c_0) + jnp.log(k_0)
    return jnp.where(solution.ok, log_tvc - initial_log_tvc, jnp.inf)


# The FD consumption policy as a simulation-compatible callable: bilinear
# interpolation on the benchmark grid, edge-clamped, i = f - c.
@struct.dataclass
class FDInterpolatedPolicy:
    log_z_grid: jax.Array
    log_k_grid: jax.Array
    consumption_grid: jax.Array

    def __call__(self, log_state, p):
        index_z = (log_state[0] - self.log_z_grid[0]) / (
            self.log_z_grid[1] - self.log_z_grid[0]
        )
        index_k = (log_state[1] - self.log_k_grid[0]) / (
            self.log_k_grid[1] - self.log_k_grid[0]
        )
        c = jax.scipy.ndimage.map_coordinates(
            self.consumption_grid,
            [index_z[None], index_k[None]],
            order=1,
            mode="nearest",
        )[0]
        z, k = jnp.exp(log_state)
        return production(z, k, p) - c


@lru_cache(maxsize=None)
def cached_stochastic_benchmark(
    a,
    delta,
    rho_hat,
    sigma_crra,
    eta,
    sigma_z,
    log_z_bar,
    num_z_points,
    num_k_points,
    z_sd_span,
    k_lo,
    k_hi,
):
    return stochastic_growth_benchmark(
        a,
        delta,
        rho_hat,
        sigma_crra,
        eta,
        sigma_z,
        log_z_bar=log_z_bar,
        num_z_points=num_z_points,
        num_k_points=num_k_points,
        z_sd_span=z_sd_span,
        k_lo=k_lo,
        k_hi=k_hi,
    )


# Common-noise pathwise errors against the FD policy: the exogenous z path is
# identical under both policies, so the comparison isolates the capital and
# consumption responses.
def add_fd_path_diagnostics(test_df, test_values, fd_values):
    k_benchmark = np.asarray(fd_values["k"])
    c_benchmark = np.asarray(fd_values["c"])
    k_error = (np.asarray(test_values["k"]) - k_benchmark) / np.abs(k_benchmark).clip(
        min=DOMAIN_EPS
    )
    c_error = (np.asarray(test_values["c"]) - c_benchmark) / np.abs(c_benchmark).clip(
        min=DOMAIN_EPS
    )
    columns = {
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "k_rel_error": k_error,
        "c_rel_error": c_error,
    }
    for name, value in columns.items():
        test_df[name] = np.asarray(value).reshape(-1)
    return k_error, c_error


# nnx.jit so one compiled evaluation serves both the nnx policy and the FD
# interpolated policy pytree.
@nnx.jit
def trajectory_values(policy, log_states, p):
    shape = log_states.shape[:-1]
    flat_states = log_states.reshape(-1, 2)
    investment = jax.vmap(lambda state: policy(state, p))(flat_states).reshape(shape)
    c = jax.vmap(lambda state: consumption(policy, state, p))(flat_states).reshape(
        shape
    )
    euler, diffusion_term = euler_residual_terms(policy, flat_states, p)
    z, k = jnp.moveaxis(jnp.exp(log_states), -1, 0)
    return {
        "z": z,
        "k": k,
        "kappa": k / z ** (1.0 / (1.0 - p.a)),
        "i": investment,
        "c": c,
        "dlog_z": p.eta * (p.log_z_bar - jnp.log(z)),
        "dk": investment - p.delta * k,
        "euler_residual": euler.reshape(shape),
        "diffusion_term": diffusion_term.reshape(shape),
    }


@dataclass(frozen=True)
class InitialConditionSettings:
    # One data_seed fixes independent train and test draws around (z_0, k_0),
    # separate from the NN seed; a zero log variance pins its axis at the base.
    data_seed: int = 0
    num_train_trajectories: int = 1
    num_test_trajectories: int = 50
    train_log_z_variance: float = 0.0
    train_log_k_variance: float = 0.0
    # None inherits the training variances; results record the resolved values.
    test_log_z_variance: float | None = None
    test_log_k_variance: float | None = None
    # Recenter the held-out draws when set (out-of-distribution testing).
    test_z_0: float | None = None
    test_k_0: float | None = None


def draw_initial_conditions(
    z_0, k_0, num_trajectories, log_z_variance, log_k_variance, key, dtype
):
    base = jnp.asarray([z_0, k_0], dtype=dtype)
    if log_z_variance == 0.0 and log_k_variance == 0.0:
        return jnp.tile(base, (num_trajectories, 1))
    if num_trajectories < 2:
        raise ValueError("positive log variances require at least two trajectories")
    draws = jax.random.normal(key, (num_trajectories, 2), dtype=dtype)
    # Center and standardize each column so every axis carries exactly the
    # requested population variance; a zero variance pins its axis at the base.
    centered = draws - jnp.mean(draws, axis=0, keepdims=True)
    standardized = centered / jnp.sqrt(jnp.mean(centered**2, axis=0, keepdims=True))
    scales = jnp.sqrt(jnp.asarray([log_z_variance, log_k_variance], dtype=dtype))
    return jnp.exp(jnp.log(base)[None, :] + scales[None, :] * standardized)


# Train and test draws from one data_seed; the resolved test design rides
# along for the results dicts.
@dataclass(frozen=True)
class InitialConditionDraw:
    train_ic_key: jax.Array
    test_ic_key: jax.Array
    train_initial_conditions: jax.Array
    test_initial_conditions: jax.Array
    test_z_center: float
    test_k_center: float
    test_log_z_variance: float
    test_log_k_variance: float


def growth_initial_conditions(z_0, k_0, ic_set, dtype):
    train_ic_key, test_ic_key = jax.random.split(jax.random.key(ic_set.data_seed))
    train_initial_conditions = draw_initial_conditions(
        z_0,
        k_0,
        ic_set.num_train_trajectories,
        ic_set.train_log_z_variance,
        ic_set.train_log_k_variance,
        train_ic_key,
        dtype,
    )
    test_z_center = z_0 if ic_set.test_z_0 is None else ic_set.test_z_0
    test_k_center = k_0 if ic_set.test_k_0 is None else ic_set.test_k_0
    test_log_z_variance = (
        ic_set.train_log_z_variance
        if ic_set.test_log_z_variance is None
        else ic_set.test_log_z_variance
    )
    test_log_k_variance = (
        ic_set.train_log_k_variance
        if ic_set.test_log_k_variance is None
        else ic_set.test_log_k_variance
    )
    test_initial_conditions = draw_initial_conditions(
        test_z_center,
        test_k_center,
        ic_set.num_test_trajectories,
        test_log_z_variance,
        test_log_k_variance,
        test_ic_key,
        dtype,
    )
    return InitialConditionDraw(
        train_ic_key=train_ic_key,
        test_ic_key=test_ic_key,
        train_initial_conditions=train_initial_conditions,
        test_initial_conditions=test_initial_conditions,
        test_z_center=test_z_center,
        test_k_center=test_k_center,
        test_log_z_variance=test_log_z_variance,
        test_log_k_variance=test_log_k_variance,
    )


@dataclass(frozen=True)
class InvestmentPolicySettings:
    mlp_width: int = 32
    mlp_depth: int = 3
    hidden_activation: str = "gelu"
    head_init_scale: float = 1e-3
    homothetic_representation: bool = True


# i = anchor * U(h), U the unit softplus with near-zero head: homothetic is
# h(log kappa) with kappa = k / z^{1/(1-a)} the detrended capital, anchored at
# delta * k; non-homothetic h(centered log z, log k) anchored at the constant
# delta * k_scale; consumption Y - i stays unclamped.
class InvestmentPolicyNN(nnx.Module):
    def __init__(
        self,
        settings: InvestmentPolicySettings,
        *,
        z_scale=1.0,
        k_scale=1.0,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.homothetic = nnx.static(settings.homothetic_representation)
        self.log_z_scale = nnx.Variable(jnp.asarray(math.log(z_scale), dtype=dtype))
        self.log_k_scale = nnx.Variable(jnp.asarray(math.log(k_scale), dtype=dtype))
        self.k_scale = nnx.Variable(jnp.asarray(k_scale, dtype=dtype))
        self.hidden_activation = nnx.static(ACTIVATIONS[settings.hidden_activation])
        in_features = 1 if self.homothetic else 2
        self.layers = nnx.List(
            [
                nnx.Linear(
                    in_features,
                    settings.mlp_width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(settings.mlp_depth - 1):
            self.layers.append(
                nnx.Linear(
                    settings.mlp_width,
                    settings.mlp_width,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
        self.head = nnx.Linear(
            settings.mlp_width,
            1,
            kernel_init=nnx.initializers.normal(settings.head_init_scale),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, log_state, p):
        log_z, log_k = log_state
        if self.homothetic:
            h = jnp.atleast_1d(log_k - log_z / (1.0 - p.a))
            anchor = p.delta * jnp.exp(log_k)
        else:
            h = jnp.stack(
                [log_z - self.log_z_scale[...], log_k - self.log_k_scale[...]]
            )
            anchor = p.delta * self.k_scale[...]
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        return anchor * unit_softplus(self.head(h)[0])


# DrawNNXModule redraws theta but passes args through unchanged; multi-start
# chains each attempt's initial args from the previous draw, so re-chaining
# restart_key/epoch_key here (and resetting epoch to 0) is what keeps each
# retry's epoch-boundary resampling independent of the others.
class DrawInvestmentPolicy:
    def __init__(self, settings, *, z_scale, k_scale, dtype):
        self.draw_theta = DrawNNXModule(
            InvestmentPolicyNN, settings, z_scale=z_scale, k_scale=k_scale, dtype=dtype
        )

    def __call__(self, key, _, args):
        theta, _ = self.draw_theta(key, _, args)
        restart_key, epoch_key = jax.random.split(args.restart_key)
        return theta, args.replace(
            epoch_key=epoch_key,
            restart_key=restart_key,
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def __hash__(self):
        return hash((DrawInvestmentPolicy, self.draw_theta))

    def __eq__(self, other):
        return (
            isinstance(other, DrawInvestmentPolicy)
            and self.draw_theta == other.draw_theta
        )


# The load template only needs the architecture from meta: the scale
# Variables ride in the artifact's nondiff block and overwrite the defaults.
def make_investment_policy(meta, dtype):
    return InvestmentPolicyNN(
        InvestmentPolicySettings(**meta["settings"]), dtype=dtype, rngs=nnx.Rngs(0)
    )


@jax.tree_util.register_static
@dataclass(frozen=True)
class StateKernelSettings:
    amplitude: float = 1.0
    lengthscale: float = 0.6
    metric_epsilon: float = 3e-5
    # With the free constant, theta is (alpha, h_const) and only alpha is
    # penalized; without it the policy is the pure RKHS expansion.
    free_constant: bool = True
    # Homothetic: 1-D RBF on log(k/z). Non-homothetic: a 2-D expansion on the
    # centered (log z, log k) with product (ard) or sum (additive) composition
    # of RBF or Matern 1-D kernels; lengthscales default per axis to
    # `lengthscale`, and linear_mean appends two unpenalized coefficients.
    homothetic_representation: bool = True
    state_kernel: Literal[
        "ard_rbf",
        "additive_rbf",
        "ard_matern32",
        "ard_matern52",
        "additive_matern32",
        "additive_matern52",
    ] = "ard_rbf"
    log_z_lengthscale: float | None = None
    log_k_lengthscale: float | None = None
    linear_mean: bool = False


@struct.dataclass
class KernelResidualArgs:
    centers: jax.Array
    train_log_states: jax.Array
    gram: jax.Array
    state_offset: jax.Array
    k_scale: jax.Array
    # 1/(1-a): log kappa = log k - z_detrend * log z for Hicks-neutral z.
    z_detrend: jax.Array


def state_lengthscales(kernel_set):
    ell_z = kernel_set.log_z_lengthscale
    ell_k = kernel_set.log_k_lengthscale
    return (
        kernel_set.lengthscale if ell_z is None else ell_z,
        kernel_set.lengthscale if ell_k is None else ell_k,
    )


def build_residual_args(train_log_states, kernel_set, a, z_0=1.0, k_0=1.0):
    flat_states = train_log_states.reshape(-1, 2)
    dtype = flat_states.dtype
    state_offset = jnp.log(jnp.asarray([z_0, k_0], dtype))
    z_detrend = jnp.asarray(1.0 / (1.0 - a), dtype)
    if kernel_set.homothetic_representation:
        centers = flat_states[:, 1] - z_detrend * flat_states[:, 0]
        gram = rbf_state_gram(
            centers,
            jnp.asarray(kernel_set.amplitude, dtype),
            jnp.asarray(kernel_set.lengthscale, dtype),
        )
    else:
        centers = flat_states - state_offset
        gram = state_features(
            kernel_set.state_kernel,
            centers,
            centers,
            kernel_set.amplitude,
            state_lengthscales(kernel_set),
        )
    gram = (gram + gram.T) / 2.0
    return KernelResidualArgs(
        centers=centers,
        train_log_states=flat_states,
        gram=gram,
        state_offset=state_offset,
        k_scale=jnp.asarray(k_0, dtype),
        z_detrend=z_detrend,
    )


def raw_policy(theta, args, log_state, kernel_set):
    alpha = theta[0]
    if kernel_set.homothetic_representation:
        q = log_state[1] - args.z_detrend * log_state[0]
        features = rbf_state_features(
            q,
            args.centers,
            jnp.asarray(kernel_set.amplitude, q.dtype),
            jnp.asarray(kernel_set.lengthscale, q.dtype),
        )
        h = features @ alpha
    else:
        x = log_state - args.state_offset
        features = state_features(
            kernel_set.state_kernel,
            x,
            args.centers,
            kernel_set.amplitude,
            state_lengthscales(kernel_set),
        )
        h = features @ alpha
        # The linear-mean coefficients are the last theta leaf, after the free
        # constant when both are present; like h_const they are unpenalized.
        if kernel_set.linear_mean:
            h = h + x @ theta[-1]
    if kernel_set.free_constant and len(theta) > 1:
        h = h + theta[1]
    return h


# Same output maps as the recursive NN: unit softplus of the raw policy
# anchored at delta * k (homothetic) or the constant delta * k_0.
def investment(theta, args, log_state, p, kernel_set):
    raw = raw_policy(theta, args, log_state, kernel_set)
    if kernel_set.homothetic_representation:
        k = jnp.exp(log_state[1])
        return p.delta * k * unit_softplus(raw)
    return p.delta * args.k_scale * unit_softplus(raw)


@struct.dataclass
class KernelInvestmentPolicy:
    theta: tuple
    args: KernelResidualArgs
    kernel_set: StateKernelSettings = struct.field(pytree_node=False)

    def __call__(self, log_state, p):
        return investment(self.theta, self.args, log_state, p, self.kernel_set)
