import time

import jax
import jax.numpy as jnp
import numpy as np
import unopy
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
# Run twice for jit-free execution time.
jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

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
N_VALIDATION = 250  # shares no interior point with the training nodes
TRAIN_RESIDUAL = 1e-8
VALIDATION_RESIDUAL = 1e-4

t_train = jnp.linspace(0.0, TRAIN_T, N_TRAIN)
t_test = jnp.linspace(0.0, TEST_T, N_TEST)
t_validation = jnp.linspace(0.0, TRAIN_T, N_VALIDATION)


# Matern-1/2 kernel k(t, t') = sigma^2 e^{-|t - t'|/ell} with antiderivative
# h(d) = int_0^d k = sigma^2 ell (1 - e^{-d/ell}). The integrated kernel
# K_tilde(t_i, t_j) = int_0^{t_i} k(s, t_j) ds splits at s = t_j:
# h(t_j) + sign(t_i - t_j) h(|t_i - t_j|), so K_tilde(0, .) = 0 makes p_0
# exactly the initial price.
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
args = {"K": K, "K_tilde": K_tilde, "t": t_train}

theta_0 = {"alpha": jnp.zeros(N_TRAIN), "p_0": jnp.zeros(())}
# ravel_pytree flattens the ridgeless theta dict (sorted keys) into Uno's
# vector and returns the inverse unravel.
x_initial, unravel = ravel_pytree(theta_0)
N_VARIABLES = x_initial.size
N_CONSTRAINTS = N_TRAIN


# min alpha' K alpha subject to the NLLS residual_fn vanishing at every
# training date.
@jax.jit
def objective(x):
    alpha = unravel(x)["alpha"]
    return alpha @ K @ alpha


@jax.jit
def constraints(x):
    return residual_fn(unravel(x), args, p)


def lagrangian(x, objective_multiplier, multipliers):
    return objective_multiplier * objective(x) + multipliers @ constraints(x)


objective_gradient = jax.jit(jax.grad(objective))
constraint_jacobian = jax.jit(jax.jacfwd(constraints))
lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

# Interior-point NLP: dense Jacobian and lower-triangle Lagrangian Hessian
# callbacks.
variable_lower_bounds = np.full(N_VARIABLES, -np.inf)
variable_upper_bounds = np.full(N_VARIABLES, np.inf)
model = unopy.Model(
    unopy.PROBLEM_QUADRATIC,
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
# N_TRAIN + 1 block is ours.
theta = unravel(jnp.asarray(np.array(result.primal_solution)[:N_VARIABLES]))
alpha, p_0 = theta["alpha"], theta["p_0"]

K_test, K_tilde_test = kernel_matrices(t_test, t_train)
price_test = p_0 + K_tilde_test @ alpha
test_residual = K_test @ alpha - (r * price_test - dividends(t_test, p))

# Closed-form fundamental (no-bubble) price.
price_fundamental = (x_0 + c / g) * jnp.exp(g * t_test) / (r - g) - c / (g * r)
rel_error = jnp.abs(price_test - price_fundamental) / price_fundamental

print(f"solve: {result.solution_status.name} in {solve_time:.2f}s (incl. jit)")
print(f"p_0 = {float(p_0):.6f}")
print(f"max |test residual|: {float(jnp.max(jnp.abs(test_residual))):.3e}")
print(f"max rel error vs fundamental: {float(jnp.max(rel_error)):.3e}")

# The solve status above is reported for diagnostics; acceptance is judged
# against the max train residual bound and, with residual_fn reused on a
# finer validation grid over the training horizon, the between-node residual.
max_train_residual = float(jnp.max(jnp.abs(residual_fn(theta, args, p))))
K_validation, K_tilde_validation = kernel_matrices(t_validation, t_train)
args_validation = {"K": K_validation, "K_tilde": K_tilde_validation, "t": t_validation}
validation_residual = residual_fn(theta, args_validation, p)
max_validation_residual = float(jnp.max(jnp.abs(validation_residual)))
mean_validation_residual = float(jnp.mean(jnp.abs(validation_residual)))
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
