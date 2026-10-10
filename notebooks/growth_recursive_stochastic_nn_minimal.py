import math
import time

import jax
import jax.numpy as jnp
from flax import nnx
from nlls_gram import LevenbergMarquardt
from tinydiffeq import EulerMaruyama, SaveAt, solve_sde

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
# Run twice for jit-free execution time.
jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

a = 1.0 / 3.0  # capital share in f(z, k) = z k^a
delta = 0.1  # depreciation rate
rho_hat = -4.0 * math.log(0.9896)  # continuous discount rate (quarterly beta anchor)
sigma_crra = 2.0  # CRRA curvature
# OU log TFP: d log z = eta (log z_bar - log z) dt + sigma_z dW, matched to a
# quarterly AR(1) with persistence 0.95 and innovation sd 0.035.
eta = -4.0 * math.log(0.95)
sigma_z = math.sqrt(2.0 * eta * 0.035**2 / (1.0 - 0.95**2))
log_z_bar = 0.0
z_0 = 1.0  # initial TFP
k_0 = 1.0  # initial capital
p = {
    "a": a,
    "delta": delta,
    "rho_hat": rho_hat,
    "sigma_crra": sigma_crra,
    "eta": eta,
    "sigma_z": sigma_z,
    "log_z_bar": log_z_bar,
}

WIDTH = 16
DEPTH = 2  # hidden silu layers
HEAD_INIT_SCALE = 1e-3  # near-zero head: the policy starts at the anchor
SEED = 0  # network initialization
DATA_SEED = 0  # Brownian paths: training epochs, test, transversality
TRAIN_T = 60.0
TRAIN_STEPS = 40  # Euler-Maruyama steps; residuals sit on the 41 states per path
TRAIN_PATHS = 16  # Brownian paths from the single initial state
TEST_T = 40.0
TEST_STEPS = 40
TEST_PATHS = 50
TVC_T = 200.0
TVC_STEPS = 256
EPOCHS = 4  # each epoch redraws the Brownian paths
STEPS_PER_EPOCH = 125
LOSS_THRESHOLD = 5e-13  # mean squared Euler residual on a fresh batch
INIT_DAMPING = 3.26e-5
DAMPING_DECREASE = 0.482
DAMPING_INCREASE = 2.71
GEODESIC_ACCEPTANCE_RATIO = 0.541


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale.
def unit_softplus(raw):
    return jax.nn.softplus(2.0 * jnp.log(2.0) * raw) / jnp.log(2.0)


# Investment policy on the log state (log z, log k): a silu MLP of x - x_0 with
# x_0 = (log z_bar, log k_0), anchored at delta k_0 so the near-zero head
# starts at replacement investment at the initial capital.
log_state_0 = jnp.array([log_z_bar, math.log(k_0)])
log_state_initial = jnp.array([math.log(z_0), math.log(k_0)])


class InvestmentPolicy(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        layers = [nnx.Linear(2, WIDTH, param_dtype=jnp.float64, rngs=rngs)]
        for _ in range(DEPTH - 1):
            layers.append(nnx.Linear(WIDTH, WIDTH, param_dtype=jnp.float64, rngs=rngs))
        self.layers = nnx.List(layers)
        self.head = nnx.Linear(
            WIDTH,
            1,
            kernel_init=nnx.initializers.normal(HEAD_INIT_SCALE),
            param_dtype=jnp.float64,
            rngs=rngs,
        )

    def __call__(self, log_state):
        h = log_state - log_state_0
        for layer in self.layers:
            h = jax.nn.silu(layer(h))
        return delta * k_0 * unit_softplus(self.head(h))


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return z * k ** p["a"] - policy(log_state)[0]


def log_state_drift(policy, log_state, p):
    log_z, log_k = log_state
    return jnp.stack(
        [
            p["eta"] * (p["log_z_bar"] - log_z),
            policy(log_state)[0] / jnp.exp(log_k) - p["delta"],
        ]
    )


def log_state_diffusion(log_state, p):
    return jnp.stack([p["sigma_z"], 0.0])


# Stochastic Euler equation with l = log c: the drift of l along the state
# plus the Ito correction (sigma_z^2 / 2)(l_zz - sigma l_z^2) from the drift of
# u'(c) = exp(-sigma l), against the Euler rate (f_k - delta - rho_hat) / sigma.
# Both derivatives of l in z come from nested jax.jvp in the e_z direction.
def euler_residual(policy, log_state, p):
    def log_consumption(state):
        return jnp.log(consumption(policy, state, p))

    e_z = jnp.array([1.0, 0.0])
    drift = log_state_drift(policy, log_state, p)
    log_c, dlog_c_dt = jax.jvp(log_consumption, (log_state,), (drift,))
    l_z, l_zz = jax.jvp(
        lambda state: jax.jvp(log_consumption, (state,), (e_z,))[1],
        (log_state,),
        (e_z,),
    )
    z, k = jnp.exp(log_state)
    # 0 * log_c is NaN where c <= 0; the tangents alone would stay finite there.
    return (
        dlog_c_dt
        + 0.5 * p["sigma_z"] ** 2 * (l_zz - p["sigma_crra"] * l_z**2)
        + 0.0 * log_c
        - (p["a"] * z * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
        / p["sigma_crra"]
    )


# Fixed-step Euler-Maruyama closed-loop simulation of one path from the initial
# state, saving the integrator's own states; the key fixes the Brownian path.
def simulate(policy, key, T, n_steps, p):
    return solve_sde(
        lambda log_state: log_state_drift(policy, log_state, p),
        lambda log_state: log_state_diffusion(log_state, p),
        EulerMaruyama(),
        0.0,
        T,
        log_state_initial,
        key=key,
        n_steps=n_steps,
        save_at=SaveAt(steps=True),
    )


def euler_residuals_on_paths(policy, keys, T, n_steps, p):
    solution = jax.vmap(lambda key: simulate(policy, key, T, n_steps, p))(keys)
    return jax.vmap(jax.vmap(lambda s: euler_residual(policy, s, p)))(solution.xs)


graphdef, theta_0 = nnx.split(InvestmentPolicy(rngs=nnx.Rngs(SEED)), nnx.Param)


# Sample-average approximation: Euler residuals on the closed-loop paths the
# current policy generates under the Brownian paths in args["keys"]. Jitted
# because the solver also evaluates it eagerly once per solve to size buffers.
@jax.jit
def residual_fn(theta, args, p):
    policy = nnx.merge(args["graphdef"], theta)
    euler = euler_residuals_on_paths(policy, args["keys"], TRAIN_T, TRAIN_STEPS, p)
    return (euler / jnp.sqrt(euler.size)).reshape(-1)


@jax.jit
def loss(theta, args, p):
    return jnp.sum(residual_fn(theta, args, p) ** 2)


solver = LevenbergMarquardt(
    residual_fn,
    init_damping=INIT_DAMPING,
    damping_decrease=DAMPING_DECREASE,
    damping_increase=DAMPING_INCREASE,
    geodesic_acceptance_ratio=GEODESIC_ACCEPTANCE_RATIO,
)
theta = theta_0
epoch_key, test_key, tvc_key = jax.random.split(jax.random.key(DATA_SEED), 3)
total_steps = 0
start = time.perf_counter()
for epoch in range(EPOCHS):
    epoch_key, batch_key = jax.random.split(epoch_key)
    args = {"graphdef": graphdef, "keys": jax.random.split(batch_key, TRAIN_PATHS)}
    # The loss on the fresh batch is the generalization test: stop when the
    # policy already fits paths it was not trained on.
    fresh_loss = float(loss(theta, args, p))
    print(f"epoch {epoch}: fresh-batch loss {fresh_loss:.3e}")
    if fresh_loss < LOSS_THRESHOLD:
        break
    result = solver.solve(theta, args, p=p, max_steps=STEPS_PER_EPOCH)
    theta = result.x
    total_steps += int(result.steps)
solve_time = time.perf_counter() - start
policy = nnx.merge(graphdef, theta)

# Held-out Brownian paths, never seen in training.
euler_test = euler_residuals_on_paths(
    policy, jax.random.split(test_key, TEST_PATHS), TEST_T, TEST_STEPS, p
)

# Log transversality diagnostic at T on each of many long paths, normalized by
# its initial-date value: (-rho_hat T - sigma log c(T) + log k(T)) - (-sigma
# log c(0) + log k(0)). On the saddle path every realization sits far negative;
# a capital bubble drives it up, so the worst path is reported.
terminal_log_states = jax.vmap(
    lambda key: simulate(policy, key, TVC_T, TVC_STEPS, p).xs[-1]
)(jax.random.split(tvc_key, TEST_PATHS))
c_T = jax.vmap(lambda s: consumption(policy, s, p))(terminal_log_states)
c_init = consumption(policy, log_state_initial, p)
log_tvc = (-rho_hat * TVC_T - sigma_crra * jnp.log(c_T) + terminal_log_states[:, 1]) - (
    -sigma_crra * jnp.log(c_init) + math.log(k_0)
)

print(f"solve: {total_steps} steps in {solve_time:.2f}s (incl. jit)")
print(
    f"test |Euler residual| over {TEST_PATHS} fresh paths: "
    f"mean {float(jnp.abs(euler_test).mean()):.3e}, max {float(jnp.abs(euler_test).max()):.3e}"
)
print(
    f"max log-TVC at t={TVC_T:.0f} over {TEST_PATHS} paths: {float(log_tvc.max()):.1f}   (should be << 0)"
)
