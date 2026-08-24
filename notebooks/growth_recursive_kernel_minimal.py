import math
import time

import jax
import jax.numpy as jnp
from nlls_gram import RepeatedFactorMetric, RidgeLevenbergMarquardt
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_bvp, solve_ode

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

a = 1.0 / 3.0  # capital share in f(z, k) = z k^a
delta = 0.1  # depreciation rate
rho_hat = -4.0 * math.log(0.9896)  # continuous discount rate (quarterly beta anchor)
sigma_crra = 2.0  # CRRA curvature
g = 0.02  # balanced growth rate; TFP z grows at (1 - a) g
z_0 = 1.0  # initial TFP
k_0 = 1.0  # initial capital
p = {"a": a, "delta": delta, "rho_hat": rho_hat, "sigma_crra": sigma_crra, "g": g}

AMPLITUDE = 1.0  # RBF amplitude
LENGTHSCALE = 0.6  # RBF length scale in log kappa
METRIC_EPSILON = 3e-5  # diagonal shift before the Cholesky metric factor
TRAIN_T = 100.0
TRAIN_POINTS = 65  # collocation states (and RBF centers) on the benchmark path
TEST_T = 40.0
TEST_STEPS = 40
TVC_T = 200.0
TVC_STEPS = 256
RIDGE = 1e-12  # disappearing penalty weight
ATOL = 1e-5
GTOL = 1e-7
INIT_DAMPING = 1e-8
MAX_STEPS = 100
BVP_T = 200.0
BVP_POINTS = 201
BVP_TOL = 1e-10
BVP_MAX_NODES = 2048

# Reference saddle path in the detrended coordinates kappa = k / z^{1/(1-a)},
# c_tilde = c / z^{1/(1-a)}: kappa' = kappa^a - c_tilde - (delta + g) kappa,
# c_tilde' = (c_tilde / sigma)(a kappa^{a-1} - rho_hat - delta - sigma g),
# with kappa(0) = kappa_0 and the terminal pin kappa(T) = kappa_ss.
kappa_0 = k_0 / z_0 ** (1.0 / (1.0 - a))
kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1.0 / (a - 1.0))


def kappa_field(t, y):
    kappa, c_tilde = y
    return jnp.array(
        [
            kappa**a - c_tilde - (delta + g) * kappa,
            (c_tilde / sigma_crra)
            * (a * kappa ** (a - 1.0) - rho_hat - delta - sigma_crra * g),
        ]
    )


def kappa_bc(ya, yb):
    return jnp.array([ya[0] - kappa_0, yb[0] - kappa_ss])


# Saddle-shaped initial guess: exponential approach at the linearized stable
# eigenvalue, with steady-flow consumption.
mesh = jnp.linspace(0.0, BVP_T, BVP_POINTS)
c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
f_pp = a * (a - 1.0) * kappa_ss ** (a - 2.0)
trace = rho_hat + (sigma_crra - 1.0) * g
lambda_stable = 0.5 * (
    trace - jnp.sqrt(trace**2 - 4.0 * c_tilde_ss * f_pp / sigma_crra)
)
kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * jnp.exp(lambda_stable * mesh)
y_guess = jnp.stack([kappa_guess, kappa_guess**a - (delta + g) * kappa_guess], axis=-1)
benchmark = solve_bvp(
    kappa_field, kappa_bc, mesh, y_guess, tol=BVP_TOL, max_nodes=BVP_MAX_NODES
)

# Collocation states: the benchmark path sampled on the training dates,
# re-trended to (log z, log k); their log kappa values double as RBF centers.
t_train = jnp.linspace(0.0, TRAIN_T, TRAIN_POINTS)
log_z_train = jnp.log(z_0) + (1.0 - a) * g * t_train
log_k_train = jnp.log(benchmark(t_train)[:, 0]) + log_z_train / (1.0 - a)
train_log_states = jnp.stack([log_z_train, log_k_train], axis=-1)
centers = log_k_train - log_z_train / (1.0 - a)


def rbf_features(q):
    return AMPLITUDE**2 * jnp.exp(
        -((q[..., None] - centers) ** 2) / (2.0 * LENGTHSCALE**2)
    )


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale.
def unit_softplus(raw):
    return jax.nn.softplus(2.0 * jnp.log(2.0) * raw) / jnp.log(2.0)


# Homothetic investment policy on the detrended state: an RBF expansion in
# log kappa = log k - log z / (1 - a) plus the unpenalized free constant b,
# anchored at replacement investment delta k.
def investment(theta, log_state, p):
    alpha, b = theta["alpha"], theta["b"]
    log_kappa = log_state[1] - log_state[0] / (1.0 - p["a"])
    return (
        p["delta"]
        * jnp.exp(log_state[1])
        * unit_softplus(rbf_features(log_kappa) @ alpha + b)
    )


def consumption(theta, log_state, p):
    z, k = jnp.exp(log_state)
    return z * k ** p["a"] - investment(theta, log_state, p)


def log_state_drift(theta, log_state, p):
    log_k = log_state[1]
    return jnp.stack(
        [
            (1.0 - p["a"]) * p["g"],
            investment(theta, log_state, p) / jnp.exp(log_k) - p["delta"],
        ]
    )


# d log c / dt along the state drift via jax.jvp, against the Euler rate
# (f_k - delta - rho_hat) / sigma.
def euler_residual(theta, log_state, p):
    drift = log_state_drift(theta, log_state, p)
    _, dlog_c_dt = jax.jvp(
        lambda state: jnp.log(consumption(theta, state, p)), (log_state,), (drift,)
    )
    z, k = jnp.exp(log_state)
    return (
        dlog_c_dt
        - (p["a"] * z * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
        / p["sigma_crra"]
    )


def residual_fn(theta, args, p):
    euler = jax.vmap(lambda s: euler_residual(theta, s, p))(args)
    return euler / jnp.sqrt(euler.size)


gram = rbf_features(centers)
metric = RepeatedFactorMetric(
    jnp.linalg.cholesky(gram + METRIC_EPSILON * jnp.eye(TRAIN_POINTS), upper=True),
    repeats=1,
)
# Flattened by sorted key: "alpha" (penalized by the metric) precedes the free "b".
theta_0 = {"alpha": jnp.zeros(TRAIN_POINTS), "b": jnp.zeros(())}

solver = RidgeLevenbergMarquardt(
    residual_fn, metric=metric, ridge=RIDGE, init_damping=INIT_DAMPING
)
start = time.perf_counter()
result = solver.solve(
    theta_0,
    train_log_states,
    p=p,
    max_steps=MAX_STEPS,
    atol=ATOL,
    gtol=GTOL,
    max_steps_is_success=False,
)
solve_time = time.perf_counter() - start
theta = result.x


def drift_field(log_state, _, args, p):
    theta, structural = p
    return log_state_drift(theta, log_state, structural)


# Fixed-step RK4 closed-loop simulation, saving the integrator's own states.
def simulate(theta, initial_log_state, T, n_steps, p):
    return solve_ode(
        drift_field,
        RK4(),
        0.0,
        T,
        initial_log_state,
        dt_0=T / n_steps,
        controller=ConstantStepSize(),
        max_steps=n_steps,
        save_at=SaveAt(steps=True),
        p=(theta, p),
    )


initial_log_state = jnp.log(jnp.array([z_0, k_0]))
test_solution = simulate(theta, initial_log_state, TEST_T, TEST_STEPS, p)
euler_test = jax.vmap(lambda s: euler_residual(theta, s, p))(test_solution.xs)

# Log transversality certificate at T, normalized by its initial-date value:
# (-rho_hat T - sigma log c(T) + log k(T)) - (-sigma log c(0) + log k(0)).
tvc_solution = simulate(theta, initial_log_state, TVC_T, TVC_STEPS, p)
terminal_log_state = tvc_solution.xs[-1]
c_T = consumption(theta, terminal_log_state, p)
k_T = jnp.exp(terminal_log_state[1])
c_init = consumption(theta, initial_log_state, p)
log_tvc = (-rho_hat * TVC_T - sigma_crra * jnp.log(c_T) + jnp.log(k_T)) - (
    -sigma_crra * jnp.log(c_init) + jnp.log(k_0)
)

print(f"solve: {int(result.steps)} steps in {solve_time:.2f}s (incl. jit)")
print(f"train loss: {float(result.info.resid_loss):.3e}")
print(f"mean |Euler residual| test roll: {float(jnp.abs(euler_test).mean()):.3e}")
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")
