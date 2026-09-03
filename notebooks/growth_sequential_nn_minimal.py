import math
import time

import jax
import jax.numpy as jnp
from flax import nnx
from nlls_gram import LevenbergMarquardt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

a = 1.0 / 3.0  # capital share in f(k) = k^a
delta = 0.1  # depreciation rate
rho_hat = -4.0 * math.log(0.9896)  # continuous discount rate (quarterly beta anchor)
sigma_crra = 2.0  # CRRA curvature
k_0 = 1.0  # initial capital
p = {"a": a, "delta": delta, "rho_hat": rho_hat, "sigma_crra": sigma_crra, "k_0": k_0}

WIDTH = 32
SEED = 0
TRAIN_T = 100.0
N_TRAIN = 101
TEST_T = 110.0
N_TEST = 101
N_VALIDATION = 801
TVC_T = 200.0
MAX_STEPS = 250
LOSS_THRESHOLD = 1e-15  # sum of squared residuals; atol is its square root
INIT_DAMPING = 1e-5
DAMPING_INCREASE = 2.0
GEODESIC_ACCEPTANCE_RATIO = 0.8
TRAIN_RESIDUAL = 1e-5
VALIDATION_RESIDUAL = 1e-4

t_train = jnp.linspace(0.0, TRAIN_T, N_TRAIN)
t_test = jnp.linspace(0.0, TEST_T, N_TEST)
t_validation = jnp.linspace(0.0, TRAIN_T, N_VALIDATION)

C_FLOW = k_0**a - delta * k_0  # steady-flow consumption scale


# U(0) = U'(0) = 1, so a near-zero head starts at its declared output scale.
def unit_softplus(raw):
    return jax.nn.softplus(2.0 * jnp.log(2.0) * raw) / jnp.log(2.0)


# t -> (k, c): tanh MLP on t / TRAIN_T with a small head, so the initial
# paths sit near the flat (k_0, C_FLOW) and stay positive through U.
class PathNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(1, WIDTH, param_dtype=jnp.float64, rngs=rngs)
        self.l2 = nnx.Linear(WIDTH, WIDTH, param_dtype=jnp.float64, rngs=rngs)
        self.l3 = nnx.Linear(WIDTH, WIDTH, param_dtype=jnp.float64, rngs=rngs)
        self.head = nnx.Linear(
            WIDTH,
            2,
            kernel_init=nnx.initializers.normal(1e-3),
            param_dtype=jnp.float64,
            rngs=rngs,
        )

    def __call__(self, t):
        h = jnp.atleast_1d(t / TRAIN_T)
        h = jnp.tanh(self.l1(h))
        h = jnp.tanh(self.l2(h))
        h = jnp.tanh(self.l3(h))
        # Fixed output scales anchor the parameterization; p["k_0"] imposes
        # the initial condition.
        return jnp.array([k_0, C_FLOW]) * unit_softplus(self.head(h))


graphdef, theta_0 = nnx.split(PathNN(rngs=nnx.Rngs(SEED)), nnx.Param)


# Resource and consumption-Euler residuals with time derivatives via jax.jvp,
# plus the initial-condition row k(0) - k_0.
def residual_fn(theta, args, p):
    model = nnx.merge(args["graphdef"], theta)
    values, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
        args["t"]
    )
    k, c = values[:, 0], values[:, 1]
    dk_dt, dc_dt = derivatives[:, 0], derivatives[:, 1]
    resource = dk_dt - (k ** p["a"] - p["delta"] * k - c)
    euler = (
        dc_dt
        - c
        * (p["a"] * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
        / p["sigma_crra"]
    )
    ic = model(jnp.asarray(0.0))[0] - p["k_0"]
    residual = jnp.concatenate([resource, euler, jnp.atleast_1d(ic)])
    return residual / jnp.sqrt(residual.size)


solver = LevenbergMarquardt(
    residual_fn,
    init_damping=INIT_DAMPING,
    damping_increase=DAMPING_INCREASE,
    geodesic_acceptance_ratio=GEODESIC_ACCEPTANCE_RATIO,
)
start = time.perf_counter()
result = solver.solve(
    theta_0,
    {"graphdef": graphdef, "t": t_train},
    p=p,
    max_steps=MAX_STEPS,
    atol=math.sqrt(LOSS_THRESHOLD),
    max_steps_is_success=False,
)
solve_time = time.perf_counter() - start
model = nnx.merge(graphdef, result.x)

values, derivatives = jax.vmap(lambda s: jax.jvp(model, (s,), (jnp.ones_like(s),)))(
    t_test
)
k_test, c_test = values[:, 0], values[:, 1]
dc_test = derivatives[:, 1]
euler_test = dc_test - c_test * (a * k_test ** (a - 1.0) - delta - rho_hat) / sigma_crra

# Log transversality certificate at T: -rho_hat T - sigma log c(T) + log k(T).
k_T, c_T = model(jnp.asarray(TVC_T))
log_tvc = -rho_hat * TVC_T - sigma_crra * jnp.log(c_T) + jnp.log(k_T)

print(f"solve: {int(result.steps)} steps in {solve_time:.2f}s (incl. jit)")
print(f"final loss: {float(result.info.loss):.3e}")
print(f"mean |Euler residual| test: {float(jnp.abs(euler_test).mean()):.3e}")
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")

# A single seed reaches the loss threshold here; a production run wraps the
# same solve in a multi-start over several draws and applies this same
# ex-post gate afterward, evaluating the residual on a finer validation grid
# over the training horizon rather than the loss the optimizer stopped on.
# residual_fn scales its rows by 1/sqrt(count); the gate compares the
# unscaled rows.
train_residual = residual_fn(result.x, {"graphdef": graphdef, "t": t_train}, p)
max_train_residual = float(
    jnp.max(jnp.abs(train_residual)) * jnp.sqrt(train_residual.size)
)
validation_residual = residual_fn(
    result.x, {"graphdef": graphdef, "t": t_validation}, p
)
max_validation_residual = float(
    jnp.max(jnp.abs(validation_residual)) * jnp.sqrt(validation_residual.size)
)
train_ok = max_train_residual <= TRAIN_RESIDUAL
validation_ok = max_validation_residual <= VALIDATION_RESIDUAL
print(
    f"max train residual {max_train_residual:.3e} <= {TRAIN_RESIDUAL:.0e}: {train_ok}"
)
print(
    f"max validation residual {max_validation_residual:.3e} <= "
    f"{VALIDATION_RESIDUAL:.0e}: {validation_ok}"
)
print(f"accepted: {train_ok and validation_ok}")
