import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import unopy

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

a = 1.0 / 3.0  # capital share in f(k) = k^a
delta = 0.1  # depreciation rate
rho_hat = -4.0 * math.log(0.9896)  # continuous discount rate (quarterly beta anchor)
sigma_crra = 2.0  # CRRA curvature
k_0 = 1.0  # initial capital

ELL = 10.0  # Matern-1/2 length scale
SIGMA = 1.0  # kernel amplitude
TRAIN_T = 100.0
N_TRAIN = 101
TEST_T = 110.0
N_TEST = 100
TVC_T = 200.0
DOMAIN_EPS = 1e-6  # domain floor for the interior point's infeasible iterates

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


K, K_tilde = kernel_matrices(t_train, t_train)

N_VARIABLES = 3 * N_TRAIN + 2
N_CONSTRAINTS = 6 * N_TRAIN


def unpack(x):
    alpha_mu = x[:N_TRAIN]
    alpha_c = x[N_TRAIN : 2 * N_TRAIN]
    alpha_k = x[2 * N_TRAIN : 3 * N_TRAIN]
    c_0 = x[3 * N_TRAIN]
    mu_0 = x[3 * N_TRAIN + 1]
    return alpha_mu, alpha_c, alpha_k, c_0, mu_0


# min sum_j alpha_j' K alpha_j over the three paths, subject to the Pontryagin
# residuals vanishing at every training date and (k, c, mu) staying positive.
@jax.jit
def objective(x):
    alpha_mu, alpha_c, alpha_k, _, _ = unpack(x)
    return alpha_mu @ K @ alpha_mu + alpha_c @ K @ alpha_c + alpha_k @ K @ alpha_k


# CRRA first-order condition u'(c) = mu; the costate and resource residuals
# are utility-free. IPOPT's interior point evaluates infeasible trial
# iterates, so k clamps at DOMAIN_EPS inside the model functions; the last
# 3n rows keep (k, c, mu) >= DOMAIN_EPS as inequality constraints.
@jax.jit
def constraints(x):
    alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)
    mu = mu_0 + K_tilde @ alpha_mu
    dmu_dt = K @ alpha_mu
    c = c_0 + K_tilde @ alpha_c
    k = k_0 + K_tilde @ alpha_k
    dk_dt = K @ alpha_k
    k_positive = jnp.maximum(k, DOMAIN_EPS)
    resource = dk_dt - (k_positive**a - delta * k - c)
    costate = dmu_dt + mu * (a * k_positive ** (a - 1.0) - delta - rho_hat)
    foc = mu * c**sigma_crra - 1.0
    return jnp.concatenate([resource, costate, foc, k, c, mu])


def lagrangian(x, objective_multiplier, multipliers):
    return objective_multiplier * objective(x) + multipliers @ constraints(x)


objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

# Interior-point NLP: dense Jacobian and lower-triangle Lagrangian Hessian
# callbacks; the free scalars (c_0, mu_0) are bounded below at DOMAIN_EPS.
variable_lower_bounds = np.full(N_VARIABLES, -np.inf)
variable_upper_bounds = np.full(N_VARIABLES, np.inf)
variable_lower_bounds[3 * N_TRAIN :] = DOMAIN_EPS
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


constraint_lower_bounds = np.concatenate(
    [np.zeros(3 * N_TRAIN), np.full(3 * N_TRAIN, DOMAIN_EPS)]
)
constraint_upper_bounds = np.concatenate(
    [np.zeros(3 * N_TRAIN), np.full(3 * N_TRAIN, np.inf)]
)
model.set_constraints(
    N_CONSTRAINTS,
    constraints_callback,
    constraint_lower_bounds,
    constraint_upper_bounds,
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

# Flat steady-flow warm start for the free scalars.
c_flow = k_0**a - delta * k_0
x_initial = np.zeros(N_VARIABLES)
x_initial[3 * N_TRAIN] = c_flow
x_initial[3 * N_TRAIN + 1] = c_flow**-sigma_crra
model.set_initial_primal_iterate(x_initial)

solver = unopy.UnoSolver()
solver.set_preset("ipopt")
solver.set_option("logger", "SILENT")
solver.set_option("print_solution", False)

start = time.perf_counter()
result = solver.optimize(model)
solve_time = time.perf_counter() - start

# primal_solution appends solver-internal auxiliaries; only the leading
# 3 N_TRAIN + 2 block is ours.
x = jnp.asarray(np.array(result.primal_solution))[:N_VARIABLES]
alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)

K_test, K_tilde_test = kernel_matrices(t_test, t_train)
mu_test = mu_0 + K_tilde_test @ alpha_mu
dmu_test = K_test @ alpha_mu
k_test = k_0 + K_tilde_test @ alpha_k
costate_test = dmu_test + mu_test * (a * k_test ** (a - 1.0) - delta - rho_hat)

# Log transversality certificate at T: -rho_hat T - sigma log c(T) + log k(T).
_, K_tilde_tvc = kernel_matrices(jnp.array([TVC_T]), t_train)
c_T = c_0 + K_tilde_tvc @ alpha_c
k_T = k_0 + K_tilde_tvc @ alpha_k
log_tvc = -rho_hat * TVC_T - sigma_crra * jnp.log(c_T[0]) + jnp.log(k_T[0])

print(f"solve: {result.solution_status.name} in {solve_time:.2f}s (incl. jit)")
print(f"c_0 = {float(c_0):.6f}, mu_0 = {float(mu_0):.6f}")
print(f"mean |Euler residual| test: {float(jnp.abs(costate_test).mean()):.3e}")
print(f"log-TVC at t={TVC_T:.0f}: {float(log_tvc):.1f}   (should be << 0)")
