import math
import time

import jax
import jax.numpy as jnp
from nlls_gram import LMStatus, RepeatedFactorMetric, RidgeLevenbergMarquardt
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_bvp, solve_ode

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
g = 0.02  # balanced growth rate; TFP z grows at (1 - a) g
z_0 = 1.0  # initial TFP
k_0 = 1.0  # initial capital
p = {"a": a, "delta": delta, "rho_hat": rho_hat, "sigma_crra": sigma_crra, "g": g}

RAY_SLOPE = 1.51  # slope s of the ray coordinate q = log k - s log z
RAY_AMPLITUDE = 0.412  # RBF amplitude in q
RAY_LENGTHSCALE = 0.634  # RBF length scale in q
AMPLITUDE = 5.71  # anisotropic RBF amplitude in (log z, log k)
LOG_Z_LENGTHSCALE = 1.0
LOG_K_LENGTHSCALE = 4.27
METRIC_EPSILON = 2e-7  # diagonal shift before the Cholesky metric factor
TRAIN_T = 60.0
TRAIN_POINTS = 61  # collocation states (and RBF centers) on the benchmark path
TEST_T = 40.0
TEST_STEPS = 40
TVC_T = 200.0
TVC_STEPS = 256
RIDGE = 3.9e-12  # disappearing penalty weight
ATOL = 6.75e-6
GTOL = 1e-7
INIT_DAMPING = 1e-6
DAMPING_DECREASE = 0.25
MAX_STEPS = 150
BVP_T = 200.0
BVP_POINTS = 201
BVP_TOL = 1e-10
BVP_MAX_NODES = 2048
TRAIN_EULER_MAX = 1e-4
TEST_EULER_MEAN_ABS = 3e-5
TEST_EULER_MAX = 3e-4

# Reference saddle path in the detrended coordinates kappa = k / z^{1/(1-a)},
# c_tilde = c / z^{1/(1-a)}: kappa' = kappa^a - c_tilde - (delta + g) kappa,
# c_tilde' = (c_tilde / sigma)(a kappa^{a-1} - rho_hat - delta - sigma g),
# with kappa(0) = kappa_0 and the terminal pin kappa(T) = kappa_ss.
kappa_0 = k_0 / z_0 ** (1.0 / (1.0 - a))
kappa_ss = ((delta + rho_hat + sigma_crra * g) / a) ** (1.0 / (a - 1.0))


def kappa_field(_, y):
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
# re-trended to (log z, log k); the states are the kernel centers, with their
# ray coordinates q = log k - RAY_SLOPE log z the centers of the ray component.
# Nothing is integrated during training.
t_train = jnp.linspace(0.0, TRAIN_T, TRAIN_POINTS)
log_z_train = jnp.log(z_0) + (1.0 - a) * g * t_train
log_k_train = jnp.log(benchmark(t_train)[:, 0]) + log_z_train / (1.0 - a)
train_log_states = jnp.stack([log_z_train, log_k_train], axis=-1)
q_centers = log_k_train - RAY_SLOPE * log_z_train


# Additive state kernel: an anisotropic RBF in (log z, log k) plus an RBF in
# the ray coordinate q; the slope is a tuned hyperparameter like the
# lengthscales.
def rbf_features(log_state):
    scaled = (log_state[..., None, :] - train_log_states) / jnp.array(
        [LOG_Z_LENGTHSCALE, LOG_K_LENGTHSCALE]
    )
    ard = AMPLITUDE**2 * jnp.exp(-0.5 * jnp.sum(scaled**2, axis=-1))
    q = log_state[..., 1] - RAY_SLOPE * log_state[..., 0]
    ray = RAY_AMPLITUDE**2 * jnp.exp(
        -((q[..., None] - q_centers) ** 2) / (2.0 * RAY_LENGTHSCALE**2)
    )
    return ard + ray


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale.
def unit_softplus(raw):
    return jax.nn.softplus(2.0 * jnp.log(2.0) * raw) / jnp.log(2.0)


# Kernel expansion in the full log state plus an unpenalized linear term and
# constant, anchored at replacement investment at the initial capital.
def investment(theta, log_state, p):
    raw = rbf_features(log_state) @ theta["alpha"] + log_state @ theta["linear"]
    return p["delta"] * k_0 * unit_softplus(raw + theta["b"])


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
    log_c, dlog_c_dt = jax.jvp(
        lambda state: jnp.log(consumption(theta, state, p)), (log_state,), (drift,)
    )
    z, k = jnp.exp(log_state)
    # 0 * log_c is NaN where c <= 0; the tangent alone would stay finite there.
    return (
        dlog_c_dt
        + 0.0 * log_c
        - (p["a"] * z * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
        / p["sigma_crra"]
    )


# Fixed-step RK4 closed-loop simulation, saving the integrator's own states.
def simulate(theta, initial_log_state, T, n_steps, p):
    return solve_ode(
        lambda log_state: log_state_drift(theta, log_state, p),
        RK4(),
        0.0,
        T,
        initial_log_state,
        dt_0=T / n_steps,
        controller=ConstantStepSize(),
        max_steps=n_steps,
        save_at=SaveAt(steps=True),
    )


def residual_fn(theta, args, p):
    euler = jax.vmap(lambda s: euler_residual(theta, s, p))(args)
    return euler / jnp.sqrt(euler.size)


gram = rbf_features(train_log_states)
metric = RepeatedFactorMetric(
    jnp.linalg.cholesky(gram + METRIC_EPSILON * jnp.eye(TRAIN_POINTS), upper=True),
    repeats=1,
)
# Flattened by sorted key: "alpha" (penalized by the metric) precedes the free
# "b" and "linear".
theta_0 = {
    "alpha": jnp.zeros(TRAIN_POINTS),
    "b": jnp.zeros(()),
    "linear": jnp.zeros(2),
}

solver = RidgeLevenbergMarquardt(
    residual_fn,
    metric=metric,
    ridge=RIDGE,
    init_damping=INIT_DAMPING,
    damping_decrease=DAMPING_DECREASE,
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

initial_log_state = jnp.log(jnp.array([z_0, k_0]))
test_solution = simulate(theta, initial_log_state, TEST_T, TEST_STEPS, p)
euler_test = jax.vmap(lambda s: euler_residual(theta, s, p))(test_solution.xs)
z_test, k_test = jnp.exp(test_solution.xs).T
c_test = jax.vmap(lambda s: consumption(theta, s, p))(test_solution.xs)

# Benchmark saddle path re-trended to levels: k*(t) = z_0^{1/(1-a)} e^{g t} kappa(t).
scale_test = z_0 ** (1.0 / (1.0 - a)) * jnp.exp(g * test_solution.ts)
kappa_benchmark, c_tilde_benchmark = benchmark(test_solution.ts).T
k_error = jnp.abs(k_test - scale_test * kappa_benchmark) / (
    scale_test * kappa_benchmark
)
c_error = jnp.abs(c_test - scale_test * c_tilde_benchmark) / (
    scale_test * c_tilde_benchmark
)

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
print(
    f"mean relative error vs benchmark: k {float(k_error.mean()):.3e}, "
    f"c {float(c_error.mean()):.3e}"
)
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")

# The production driver reports the ridge LM status but does not gate on it;
# acceptance is an ex-post check of the max training Euler residual and the
# held-out bars.
euler_train = jax.vmap(lambda s: euler_residual(theta, s, p))(train_log_states)
train_euler_max = float(jnp.max(jnp.abs(euler_train)))
train_ok = train_euler_max <= TRAIN_EULER_MAX
test_euler_mean_abs = float(jnp.abs(euler_test).mean())
test_euler_max = float(jnp.abs(euler_test).max())
test_ok = (
    test_euler_mean_abs <= TEST_EULER_MEAN_ABS and test_euler_max <= TEST_EULER_MAX
)
print(f"status: {LMStatus(result.status).name}")
print(f"max train |Euler| {train_euler_max:.3e} <= {TRAIN_EULER_MAX:.0e}: {train_ok}")
print(
    f"test mean|Euler| {test_euler_mean_abs:.3e} <= {TEST_EULER_MEAN_ABS:.0e} and "
    f"max|Euler| {test_euler_max:.3e} <= {TEST_EULER_MAX:.0e}: {test_ok}"
)
print(f"accepted: {train_ok and test_ok}")
