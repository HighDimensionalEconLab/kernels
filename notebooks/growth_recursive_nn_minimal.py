import math
import time

import jax
import jax.numpy as jnp
from flax import nnx
from nlls_gram import LevenbergMarquardt
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_ode

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

WIDTH = 32
SEED = 0
TRAIN_T = 100.0
TRAIN_STEPS = 64  # RK4 steps; residuals sit on the 65 integrator states
TEST_T = 40.0
TEST_STEPS = 40
TVC_T = 200.0
TVC_STEPS = 256
MAX_STEPS = 250
LOSS_THRESHOLD = 1e-14  # sum of squared residuals; atol is its square root


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale.
def unit_softplus(raw):
    return jax.nn.softplus(2.0 * jnp.log(2.0) * raw) / jnp.log(2.0)


# Homothetic investment policy on the detrended state: a gelu MLP of
# log kappa = log k - log z / (1 - a), anchored at delta k so the near-zero
# head starts at replacement investment i = delta k.
class InvestmentPolicy(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(1, WIDTH, param_dtype=jnp.float64, rngs=rngs)
        self.l2 = nnx.Linear(WIDTH, WIDTH, param_dtype=jnp.float64, rngs=rngs)
        self.l3 = nnx.Linear(WIDTH, WIDTH, param_dtype=jnp.float64, rngs=rngs)
        self.head = nnx.Linear(
            WIDTH,
            1,
            kernel_init=nnx.initializers.normal(1e-3),
            param_dtype=jnp.float64,
            rngs=rngs,
        )

    def __call__(self, log_state, p):
        log_z, log_k = log_state
        h = jnp.atleast_1d(log_k - log_z / (1.0 - p["a"]))
        h = jax.nn.gelu(self.l1(h))
        h = jax.nn.gelu(self.l2(h))
        h = jax.nn.gelu(self.l3(h))
        return p["delta"] * jnp.exp(log_k) * unit_softplus(self.head(h)[0])


def consumption(policy, log_state, p):
    z, k = jnp.exp(log_state)
    return z * k ** p["a"] - policy(log_state, p)


def log_state_drift(policy, log_state, p):
    log_k = log_state[1]
    return jnp.stack(
        [(1.0 - p["a"]) * p["g"], policy(log_state, p) / jnp.exp(log_k) - p["delta"]]
    )


# d log c / dt along the state drift via jax.jvp, against the Euler rate
# (f_k - delta - rho_hat) / sigma.
def euler_residual(policy, log_state, p):
    drift = log_state_drift(policy, log_state, p)
    _, dlog_c_dt = jax.jvp(
        lambda state: jnp.log(consumption(policy, state, p)), (log_state,), (drift,)
    )
    z, k = jnp.exp(log_state)
    return (
        dlog_c_dt
        - (p["a"] * z * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
        / p["sigma_crra"]
    )


def drift_field(log_state, _, args, p):
    policy, structural = p
    return log_state_drift(policy, log_state, structural)


# Fixed-step RK4 closed-loop simulation, saving the integrator's own states.
def simulate(policy, initial_log_state, T, n_steps, p):
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
        p=(policy, p),
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
    return euler / jnp.sqrt(euler.size)


solver = LevenbergMarquardt(residual_fn)
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

initial_log_state = jnp.log(jnp.array([z_0, k_0]))
test_solution = simulate(policy, initial_log_state, TEST_T, TEST_STEPS, p)
euler_test = jax.vmap(lambda s: euler_residual(policy, s, p))(test_solution.xs)

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
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")
