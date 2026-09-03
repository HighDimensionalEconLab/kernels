import math
import time

import jax
import jax.numpy as jnp
from flax import nnx
from nlls_gram import LevenbergMarquardt
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
p = {
    "a": a,
    "delta": delta,
    "rho_hat": rho_hat,
    "sigma_crra": sigma_crra,
    "g": g,
    "z_0": z_0,
    "k_0": k_0,
}

WIDTH = 8
DEPTH = 2  # hidden tanh layers
HEAD_INIT_SCALE = 4.7e-3  # near-zero head: the policy starts at the anchor
SEED = 0
TRAIN_T = 100.0
TRAIN_STEPS = 64  # RK4 steps; residuals sit on the 65 integrator states
TEST_T = 40.0
TEST_STEPS = 40
TVC_T = 200.0
TVC_STEPS = 256
MAX_STEPS = 125
LOSS_THRESHOLD = 2.15e-14  # sum of squared residuals; atol is its square root
INIT_DAMPING = 4.74e-4
DAMPING_DECREASE = 0.511
DAMPING_INCREASE = 2.82
GEODESIC_ACCEPTANCE_RATIO = 0.892
BVP_T = 200.0
BVP_POINTS = 201
BVP_TOL = 1e-10
BVP_MAX_NODES = 2048


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale.
def unit_softplus(raw):
    return jax.nn.softplus(2.0 * jnp.log(2.0) * raw) / jnp.log(2.0)


# Investment policy on the full log state: a tanh MLP of x - x_0 with
# x_0 = (log z_0, log k_0), anchored at delta k_0 so the near-zero head starts
# at replacement investment at the initial capital. No structural constant
# enters the map beyond that anchor; nothing detrends the state.
log_state_0 = jnp.log(jnp.array([z_0, k_0]))


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
            h = jnp.tanh(layer(h))
        return delta * k_0 * unit_softplus(self.head(h))


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return z * k ** p["a"] - policy(log_state)[0]


def log_state_drift(policy, log_state, p):
    log_k = log_state[1]
    return jnp.stack(
        [(1.0 - p["a"]) * p["g"], policy(log_state)[0] / jnp.exp(log_k) - p["delta"]]
    )


# d log c / dt along the state drift via jax.jvp, against the Euler rate
# (f_k - delta - rho_hat) / sigma.
def euler_residual(policy, log_state, p):
    drift = log_state_drift(policy, log_state, p)
    log_c, dlog_c_dt = jax.jvp(
        lambda state: jnp.log(consumption(policy, state, p)), (log_state,), (drift,)
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
def simulate(policy, initial_log_state, T, n_steps, p):
    return solve_ode(
        lambda log_state: log_state_drift(policy, log_state, p),
        RK4(),
        0.0,
        T,
        initial_log_state,
        dt_0=T / n_steps,
        controller=ConstantStepSize(),
        max_steps=n_steps,
        save_at=SaveAt(steps=True),
    )


graphdef, theta_0 = nnx.split(InvestmentPolicy(rngs=nnx.Rngs(SEED)), nnx.Param)


# Euler residuals on the closed-loop trajectory the current policy generates
# from (z_0, k_0); the initial state comes from p, so the solution stays
# differentiable in the initial condition.
def residual_fn(theta, args, p):
    policy = nnx.merge(args["graphdef"], theta)
    initial_log_state = jnp.log(jnp.array([p["z_0"], p["k_0"]]))
    solution = simulate(policy, initial_log_state, TRAIN_T, TRAIN_STEPS, p)
    euler = jax.vmap(lambda s: euler_residual(policy, s, p))(solution.xs)
    # A trajectory that never reached TRAIN_T is a loud optimizer failure
    # rather than a fit to its finite prefix.
    return jnp.where(solution.ok, euler / jnp.sqrt(euler.size), jnp.inf)


solver = LevenbergMarquardt(
    residual_fn,
    init_damping=INIT_DAMPING,
    damping_decrease=DAMPING_DECREASE,
    damping_increase=DAMPING_INCREASE,
    geodesic_acceptance_ratio=GEODESIC_ACCEPTANCE_RATIO,
)
start = time.perf_counter()
result = solver.solve(
    theta_0,
    {"graphdef": graphdef},
    p=p,
    max_steps=MAX_STEPS,
    atol=math.sqrt(LOSS_THRESHOLD),
    max_steps_is_success=False,
)
solve_time = time.perf_counter() - start
policy = nnx.merge(graphdef, result.x)

# Ex-post accuracy reference, never seen by training: the saddle path in the
# detrended coordinates kappa = k / z^{1/(1-a)}, c_tilde = c / z^{1/(1-a)},
# kappa' = kappa^a - c_tilde - (delta + g) kappa,
# c_tilde' = (c_tilde / sigma)(a kappa^{a-1} - rho_hat - delta - sigma g),
# with kappa(0) = kappa_0, the terminal pin kappa(T) = kappa_ss, and a
# saddle-shaped guess at the linearized stable eigenvalue.
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

initial_log_state = jnp.log(jnp.array([z_0, k_0]))
test_solution = simulate(policy, initial_log_state, TEST_T, TEST_STEPS, p)
euler_test = jax.vmap(lambda s: euler_residual(policy, s, p))(test_solution.xs)
z_test, k_test = jnp.exp(test_solution.xs).T
c_test = jax.vmap(lambda s: consumption(policy, s, p))(test_solution.xs)

# Benchmark re-trended to levels: k*(t) = z_0^{1/(1-a)} e^{g t} kappa(t).
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
tvc_solution = simulate(policy, initial_log_state, TVC_T, TVC_STEPS, p)
terminal_log_state = tvc_solution.xs[-1]
c_T = consumption(policy, terminal_log_state, p)
k_T = jnp.exp(terminal_log_state[1])
c_init = consumption(policy, initial_log_state, p)
log_tvc = (-rho_hat * TVC_T - sigma_crra * jnp.log(c_T) + jnp.log(k_T)) - (
    -sigma_crra * jnp.log(c_init) + jnp.log(k_0)
)

print(f"solve: {int(result.steps)} steps in {solve_time:.2f}s (incl. jit)")
print(f"final loss: {float(result.info.loss):.3e}")
print(f"mean |Euler residual| test roll: {float(jnp.abs(euler_test).mean()):.3e}")
print(
    f"mean relative error vs benchmark: k {float(k_error.mean()):.3e}, "
    f"c {float(c_error.mean()):.3e}"
)
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")
