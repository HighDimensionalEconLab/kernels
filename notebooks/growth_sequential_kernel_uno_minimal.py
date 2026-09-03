import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import unopy
from jax.flatten_util import ravel_pytree

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
N_VALIDATION = 250  # shares no interior point with the training nodes
TVC_T = 200.0
TRAIN_RESIDUAL = 1e-5
VALIDATION_RESIDUAL = 2e-4

t_train = jnp.linspace(0.0, TRAIN_T, N_TRAIN)
t_test = jnp.linspace(0.0, TEST_T, N_TEST)
t_validation = jnp.linspace(0.0, TRAIN_T, N_VALIDATION)


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
# curvature enters only through the first-order condition u'(c) = mu.
def residual_fn(theta, args, p):
    K, K_tilde = args["K"], args["K_tilde"]
    mu = theta["mu_0"] + K_tilde @ theta["alpha_mu"]
    dmu_dt = K @ theta["alpha_mu"]
    c = theta["c_0"] + K_tilde @ theta["alpha_c"]
    k = p["k_0"] + K_tilde @ theta["alpha_k"]
    dk_dt = K @ theta["alpha_k"]
    resource = dk_dt - (k ** p["a"] - p["delta"] * k - c)
    euler = dmu_dt + mu * (p["a"] * k ** (p["a"] - 1.0) - p["delta"] - p["rho_hat"])
    shadow_price = mu * c ** p["sigma_crra"] - 1.0
    residual = jnp.concatenate([resource, euler, shadow_price])
    return residual / jnp.sqrt(residual.size)


K, K_tilde = kernel_matrices(t_train, t_train)
args = {"K": K, "K_tilde": K_tilde}

c_flow = k_0**a - delta * k_0  # flat steady-flow warm start
theta_0 = {
    "alpha_mu": jnp.zeros(N_TRAIN),
    "alpha_c": jnp.zeros(N_TRAIN),
    "alpha_k": jnp.zeros(N_TRAIN),
    "c_0": jnp.asarray(c_flow),
    "mu_0": jnp.asarray(c_flow**-sigma_crra),
}
# ravel_pytree flattens the ridgeless theta dict (sorted keys) into Uno's
# vector and returns the inverse unravel.
x_initial, unravel = ravel_pytree(theta_0)
N_VARIABLES = x_initial.size
N_CONSTRAINTS = 3 * N_TRAIN


# min sum_j alpha_j' K alpha_j over the three paths, subject to the NLLS
# residual_fn vanishing at every training date; its constant 1/sqrt(3n) row
# scale is immaterial to the constraint set.
@jax.jit
def objective(x):
    theta = unravel(x)
    return (
        theta["alpha_mu"] @ K @ theta["alpha_mu"]
        + theta["alpha_c"] @ K @ theta["alpha_c"]
        + theta["alpha_k"] @ K @ theta["alpha_k"]
    )


@jax.jit
def constraints(x):
    return residual_fn(unravel(x), args, p)


def lagrangian(x, objective_multiplier, multipliers):
    return objective_multiplier * objective(x) + multipliers @ constraints(x)


objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

# Interior-point NLP: dense Jacobian and lower-triangle Lagrangian Hessian
# callbacks; every variable is free and every constraint row is an equality.
variable_lower_bounds = np.full(N_VARIABLES, -np.inf)
variable_upper_bounds = np.full(N_VARIABLES, np.inf)
model = unopy.Model(
    unopy.PROBLEM_NONLINEAR,
    N_VARIABLES,
    variable_lower_bounds,
    variable_upper_bounds,
    unopy.ZERO_BASED_INDEXING,
)


def objective_callback(x):
    return float(objective(jnp.asarray(x)))


def objective_gradient_callback(x, gradient):
    gradient[:] = np.asarray(objective_gradient(jnp.asarray(x)))


model.set_objective(unopy.MINIMIZE, objective_callback, objective_gradient_callback)

jacobian_rows = np.repeat(np.arange(N_CONSTRAINTS, dtype=np.int32), N_VARIABLES)
jacobian_columns = np.tile(np.arange(N_VARIABLES, dtype=np.int32), N_CONSTRAINTS)


def constraints_callback(x, constraint_output):
    constraint_output[:] = np.asarray(constraints(jnp.asarray(x)))


def jacobian_callback(x, jacobian_output):
    jacobian_output[:] = np.asarray(constraint_jacobian(jnp.asarray(x))).reshape(-1)


model.set_constraints(
    N_CONSTRAINTS,
    constraints_callback,
    np.zeros(N_CONSTRAINTS),
    np.zeros(N_CONSTRAINTS),
    len(jacobian_rows),
    jacobian_rows,
    jacobian_columns,
    jacobian_callback,
)

hessian_rows, hessian_columns = (
    idx.astype(np.int32) for idx in np.tril_indices(N_VARIABLES)
)


def hessian_callback(x, objective_multiplier, multipliers, hessian_output):
    hessian = np.asarray(
        lagrangian_hessian(
            jnp.asarray(x), float(objective_multiplier), jnp.asarray(multipliers)
        )
    )
    hessian_output[:] = hessian[hessian_rows, hessian_columns]


model.set_lagrangian_hessian(
    len(hessian_rows),
    unopy.LOWER_TRIANGLE,
    hessian_rows,
    hessian_columns,
    hessian_callback,
)
model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
model.set_initial_primal_iterate(np.asarray(x_initial))

solver = unopy.UnoSolver()
solver.set_preset("ipopt")
solver.set_option("logger", "SILENT")
solver.set_option("print_solution", False)

start = time.perf_counter()
result = solver.optimize(model)
solve_time = time.perf_counter() - start

# primal_solution appends solver-internal auxiliaries; only the leading
# 3 N_TRAIN + 2 block is ours.
theta = unravel(jnp.asarray(np.array(result.primal_solution))[:N_VARIABLES])

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

print(f"solve: {result.solution_status.name} in {solve_time:.2f}s (incl. jit)")
print(f"c_0 = {float(theta['c_0']):.6f}, mu_0 = {float(theta['mu_0']):.6f}")
print(f"mean |Euler residual| test: {float(jnp.abs(euler_test).mean()):.3e}")
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")

# The solve status above is reported for diagnostics; acceptance is judged
# against the max train residual bound and, with residual_fn reused on a
# finer validation grid over the training horizon, the mean between-node
# residual. residual_fn scales its rows by 1/sqrt(count); the gate compares
# the unscaled rows.
train_residual = residual_fn(theta, args, p)
max_train_residual = float(
    jnp.max(jnp.abs(train_residual)) * jnp.sqrt(train_residual.size)
)
K_validation, K_tilde_validation = kernel_matrices(t_validation, t_train)
validation_residual = residual_fn(
    theta, {"K": K_validation, "K_tilde": K_tilde_validation}, p
)
max_validation_residual = float(
    jnp.max(jnp.abs(validation_residual)) * jnp.sqrt(validation_residual.size)
)
mean_validation_residual = float(
    jnp.mean(jnp.abs(validation_residual)) * jnp.sqrt(validation_residual.size)
)
train_ok = max_train_residual <= TRAIN_RESIDUAL
validation_ok = mean_validation_residual <= VALIDATION_RESIDUAL
print(
    f"max train residual {max_train_residual:.3e} <= {TRAIN_RESIDUAL:.0e}: {train_ok}"
)
print(
    f"mean validation residual {mean_validation_residual:.3e} <= "
    f"{VALIDATION_RESIDUAL:.0e}: {validation_ok}"
)
print(f"max validation residual {max_validation_residual:.3e}")
print(f"accepted: {train_ok and validation_ok}")
