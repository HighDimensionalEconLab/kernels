import time

import jax
import jax.numpy as jnp
from nlls_gram import RepeatedFactorMetric, RidgeLevenbergMarquardt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

r = 0.1  # interest rate
c = 0.02  # dividend flow constant
g = -0.2  # dividend growth rate
x_0 = 0.01  # initial dividend
p = {"r": r, "c": c, "g": g, "x_0": x_0}

ELL = 10.0  # Matern-1/2 length scale
SIGMA = 1.0  # kernel amplitude
TRAIN_T = 40.0
N_TRAIN = 41
TEST_T = 50.0
N_TEST = 100
RIDGE = 1e-10  # disappearing penalty weight
ATOL = 1e-9  # unscaled residual norm
GTOL = 1e-13
INIT_DAMPING = 1e-8
DAMPING_DECREASE = 0.5
DAMPING_INCREASE = 4.0
MAX_STEPS = 100

t_train = jnp.linspace(0.0, TRAIN_T, N_TRAIN)
t_test = jnp.linspace(0.0, TEST_T, N_TEST)


# Matern-1/2 kernel k(t, t') = sigma^2 e^{-|t - t'|/ell} with antiderivative
# h(d) = int_0^d k = sigma^2 ell (1 - e^{-d/ell}). The integrated kernel
# K_tilde(t_i, t_j) = int_0^{t_i} k(s, t_j) ds splits at s = t_j:
# h(t_j) + sign(t_i - t_j) h(|t_i - t_j|), so K_tilde(0, .) = 0 pins x(0).
def antiderivative(d):
    return SIGMA**2 * ELL * (1.0 - jnp.exp(-d / ELL))


def kernel_matrices(t_i, t_j):
    s = t_i[:, None] - t_j[None, :]
    K = SIGMA**2 * jnp.exp(-jnp.abs(s) / ELL)
    K_tilde = antiderivative(t_j)[None, :] + jnp.sign(s) * antiderivative(jnp.abs(s))
    return K, K_tilde


def dividends(t, p):
    return (p["x_0"] + p["c"] / p["g"]) * jnp.exp(p["g"] * t) - p["c"] / p["g"]


# Price representation p(t) = p_0 + K_tilde alpha with exact derivative
# p'(t) = K alpha; the no-arbitrage residual is p'(t) - (r p(t) - x(t)).
def residual_fn(theta, args, p):
    alpha, p_0 = theta["alpha"], theta["p_0"]
    price = p_0 + args["K_tilde"] @ alpha
    return args["K"] @ alpha - (p["r"] * price - dividends(args["t"], p))


K, K_tilde = kernel_matrices(t_train, t_train)
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=1)
args = {"K": K, "K_tilde": K_tilde, "t": t_train}
# Flattened by sorted key: "alpha" (penalized by the metric) precedes the free "p_0".
theta_0 = {"alpha": jnp.zeros(N_TRAIN), "p_0": jnp.zeros(())}

solver = RidgeLevenbergMarquardt(
    residual_fn,
    metric=metric,
    ridge=RIDGE,
    init_damping=INIT_DAMPING,
    damping_decrease=DAMPING_DECREASE,
    damping_increase=DAMPING_INCREASE,
)
start = time.perf_counter()
result = solver.solve(
    theta_0,
    args,
    p=p,
    max_steps=MAX_STEPS,
    atol=ATOL,
    gtol=GTOL,
    max_steps_is_success=False,
)
solve_time = time.perf_counter() - start
alpha, p_0 = result.x["alpha"], result.x["p_0"]

K_test, K_tilde_test = kernel_matrices(t_test, t_train)
price_test = p_0 + K_tilde_test @ alpha
test_residual = K_test @ alpha - (r * price_test - dividends(t_test, p))

# Closed-form fundamental (no-bubble) price.
price_fundamental = (x_0 + c / g) * jnp.exp(g * t_test) / (r - g) - c / (g * r)
rel_error = jnp.abs(price_test - price_fundamental) / price_fundamental

print(f"solve: {int(result.steps)} steps in {solve_time:.2f}s (incl. jit)")
print(f"p_0 = {float(p_0):.6f}")
print(f"max |test residual|: {float(jnp.max(jnp.abs(test_residual))):.3e}")
print(f"max rel error vs fundamental: {float(jnp.max(rel_error)):.3e}")
