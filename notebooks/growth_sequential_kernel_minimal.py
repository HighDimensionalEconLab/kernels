import math
import time

import jax
import jax.numpy as jnp
from nlls_gram import RepeatedFactorMetric, RidgeLevenbergMarquardt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

a = 1.0 / 3.0  # capital share in f(k) = k^a
delta = 0.1  # depreciation rate
rho_hat = -4.0 * math.log(0.9896)  # continuous discount rate (quarterly beta anchor)
sigma_crra = 2.0  # CRRA curvature
k_0 = 1.0  # initial capital
p = {"a": a, "delta": delta, "rho_hat": rho_hat, "sigma_crra": sigma_crra, "k_0": k_0}

ELL = 10.0  # Matern-1/2 length scale
SIGMA = 1.0  # kernel amplitude
TRAIN_T = 100.0
N_TRAIN = 101
TEST_T = 110.0
N_TEST = 100
TVC_T = 200.0
RIDGE = 1e-10  # disappearing penalty weight
ATOL = 2e-8  # on the 1/sqrt(3n)-scaled residual norm
GTOL = 1e-9
INIT_DAMPING = 1e-8
MAX_STEPS = 50

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


# Each path x(t) = x(0) + K_tilde alpha with exact derivative x'(t) = K alpha;
# k(0) = k_0 is pinned while c(0), mu(0) are free scalars in theta. CRRA
# first-order condition u'(c) = mu; the euler and resource residuals are
# utility-free.
def residual_fn(theta, args, p):
    mu = theta["mu_0"] + args["K_tilde"] @ theta["alpha_mu"]
    dmu_dt = args["K"] @ theta["alpha_mu"]
    c = theta["c_0"] + args["K_tilde"] @ theta["alpha_c"]
    k = p["k_0"] + args["K_tilde"] @ theta["alpha_k"]
    dk_dt = args["K"] @ theta["alpha_k"]
    resource = dk_dt - (k ** p["a"] - p["delta"] * k - c)
    euler = dmu_dt + mu * (p["a"] * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
    shadow_price = mu * c ** p["sigma_crra"] - 1.0
    residual = jnp.concatenate([resource, euler, shadow_price])
    return residual / jnp.sqrt(residual.size)


K, K_tilde = kernel_matrices(t_train, t_train)
metric = RepeatedFactorMetric(jnp.linalg.cholesky(K, upper=True), repeats=3)
args = {"K": K, "K_tilde": K_tilde}

c_flow = k_0**a - delta * k_0  # flat steady-flow warm start
# Flattened by sorted key: the "alpha_*" blocks (penalized by the metric)
# precede the free scalars ("c_0", "mu_0").
theta_0 = {
    "alpha_mu": jnp.zeros(N_TRAIN),
    "alpha_c": jnp.zeros(N_TRAIN),
    "alpha_k": jnp.zeros(N_TRAIN),
    "c_0": jnp.asarray(c_flow),
    "mu_0": jnp.asarray(c_flow**-sigma_crra),
}

solver = RidgeLevenbergMarquardt(
    residual_fn, metric=metric, ridge=RIDGE, init_damping=INIT_DAMPING
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
theta = result.x

K_test, K_tilde_test = kernel_matrices(t_test, t_train)
mu_test = theta["mu_0"] + K_tilde_test @ theta["alpha_mu"]
k_test = k_0 + K_tilde_test @ theta["alpha_k"]
euler_test = K_test @ theta["alpha_mu"] + mu_test * (
    a * k_test ** (a - 1.0) - delta - rho_hat
)

# Log transversality certificate at T: -rho_hat T - sigma log c(T) + log k(T).
_, K_tilde_tvc = kernel_matrices(jnp.array([TVC_T]), t_train)
c_T = theta["c_0"] + K_tilde_tvc @ theta["alpha_c"]
k_T = k_0 + K_tilde_tvc @ theta["alpha_k"]
log_tvc = -rho_hat * TVC_T - sigma_crra * jnp.log(c_T[0]) + jnp.log(k_T[0])

print(f"solve: {int(result.steps)} steps in {solve_time:.2f}s (incl. jit)")
print(f"c_0 = {float(theta['c_0']):.6f}, mu_0 = {float(theta['mu_0']):.6f}")
print(f"mean |Euler residual| test: {float(jnp.abs(euler_test).mean()):.3e}")
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")
