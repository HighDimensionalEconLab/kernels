# Where does nlls-gram 1.7's dual_solve_dtype=float64 matter for a float32
# model? Production model entry points now allow float32 residuals while
# keeping the RKHS metric and dense dual pipeline in float64. The unified
# shifted metric M = blockdiag(K, 0) + eps I
# with eps = 1e-7 puts the dense dual at condition ~1/eps, which is exactly
# the flag's validated target. This experiment rebuilds the asset-pricing
# dense collocation solve with float32 data (matrices computed in float64,
# then cast, so every solver sees identical float32-representable values)
# and compares three configurations against the float64 truth:
#   f32        -- plain float32 cholesky dual
#   f32+wide   -- float32 model, dual_solve_dtype=jnp.float64
#   f64        -- full float64 reference
# Forward accuracy (p_0, train residual) and the implicit-AD path
# (d p_0 / d r through the solve, the release's worst case) are reported,
# plus per-solve wall times. Backend-agnostic: run on GPU to measure the
# float32-model + wide-dual configuration where f64 throughput is scarce.
# Run from the repo root: uv run python -m experiments.dual_solve_dtype_float32
# Optional argv: train point count (default 41), e.g.
#   uv run python -m experiments.dual_solve_dtype_float32 2001
import sys
import time

import jax
import jax.numpy as jnp
from nlls_gram import (
    UnderdeterminedLevenbergMarquardt,
    blockdiag_metric,
    metric_from_cholesky,
    metric_from_diagonal,
)

from kernels import integrated_kernel_matrices

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

R, C, G, X_0 = 0.1, 0.02, -0.2, 1.0
EPS = 1e-7
KERNEL, ELL, SIGMA = "matern12", 10.0, 1.0
TRAIN_T = 40.0
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 41
MAX_STEPS, ATOL = 100, 1e-11


def build(dtype, dual_solve_dtype=None):
    # Ground-truth matrices in float64, then cast: identical
    # float32-representable inputs across configurations.
    t64 = jnp.linspace(0.0, TRAIN_T, N_TRAIN, dtype=jnp.float64)
    K64, K_tilde64 = integrated_kernel_matrices(t64, t64, KERNEL, SIGMA, ELL)
    K64 = (K64 + K64.T) / 2
    K = K64.astype(dtype)
    K_tilde = K_tilde64.astype(dtype)
    t = t64.astype(dtype)
    x_train = ((X_0 + C / G) * jnp.exp(G * t) - C / G).astype(dtype)
    eps = jnp.asarray(EPS, dtype)

    # r enters as the differentiable parameter p so jax.grad through the
    # solve exercises the implicit rule (the release's worst case: float32
    # VJPs were ~40% wrong on this conditioning).
    def residual_fn(theta, args, r):
        alpha, p_0 = theta
        p = p_0 + K_tilde @ alpha
        dp_dt = K @ alpha
        return dp_dt - (r * p - x_train)

    chol = jnp.linalg.cholesky(K + eps * jnp.eye(N_TRAIN, dtype=dtype))
    metric = blockdiag_metric(
        [
            (metric_from_cholesky(chol), N_TRAIN),
            (metric_from_diagonal(jnp.full(1, eps, dtype)), 1),
        ]
    )
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver="cholesky",
        init_damping=1e-8,
        dual_solve_dtype=dual_solve_dtype,
        metric=metric,
    )
    theta_0 = (jnp.zeros(N_TRAIN, dtype), jnp.zeros((), dtype))

    def p_0_of_r(r):
        return solver.solve(
            theta_0, max_steps=MAX_STEPS, atol=ATOL, p=jnp.asarray(r, dtype)
        ).x[1]

    return solver, theta_0, residual_fn, p_0_of_r


def solve_case(name, dtype, dual_solve_dtype=None):
    solver, theta_0, residual_fn, p_0_of_r = build(dtype, dual_solve_dtype)
    r = jnp.asarray(R, dtype)
    jax.block_until_ready(
        solver.solve(theta_0, max_steps=1, atol=ATOL, p=r)
    )  # warm the jit cache
    start = time.perf_counter()
    result = solver.solve(theta_0, max_steps=MAX_STEPS, atol=ATOL, p=r)
    jax.block_until_ready(result)
    solve_s = time.perf_counter() - start

    p_0 = float(result.x[1])
    max_res = float(jnp.max(jnp.abs(residual_fn(result.x, None, r))))
    grad_start = time.perf_counter()
    dp0_dr = float(jax.block_until_ready(jax.grad(p_0_of_r)(R)))
    grad_s = time.perf_counter() - grad_start
    return {
        "name": name,
        "p_0": p_0,
        "max_res": max_res,
        "dp0_dr": dp0_dr,
        "solve_s": solve_s,
        "grad_s": grad_s,
        "steps": int(result.steps),
    }


if __name__ == "__main__":
    print(f"backend: {jax.default_backend()}")
    reference = solve_case("f64", jnp.float64)
    cases = [
        solve_case("f32", jnp.float32),
        solve_case("f32+wide", jnp.float32, dual_solve_dtype=jnp.float64),
        reference,
    ]
    print(
        f"{'case':10} {'p_0':>12} {'rel_err(p_0)':>12} {'max_res':>10} "
        f"{'dp0/dr':>12} {'rel_err(grad)':>13} {'steps':>5} "
        f"{'solve_s':>8} {'grad_s':>7}"
    )
    for case in cases:
        p_err = abs(case["p_0"] - reference["p_0"]) / abs(reference["p_0"])
        g_err = abs(case["dp0_dr"] - reference["dp0_dr"]) / abs(reference["dp0_dr"])
        print(
            f"{case['name']:10} {case['p_0']:>12.6f} {p_err:>12.2e} "
            f"{case['max_res']:>10.2e} {case['dp0_dr']:>12.6f} {g_err:>13.2e} "
            f"{case['steps']:>5} {case['solve_s']:>8.4f} {case['grad_s']:>7.2f}"
        )
