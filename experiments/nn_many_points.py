# Can the sequential growth NN absorb far more collocation points than it
# needs (the kernel models' 1001-node stress test), and does the nlls-gram
# 1.5 nystrom_preconditioner make the large-m dual solve cheap? Reuses the
# exact residual, net, and warm start of neoclassical_growth_nn (seed 0
# throughout, so every solver sees the same initialization) but constructs
# its own solvers: dense cholesky vs cg with the randomized Nystrom
# preconditioner built from the NTK dual J J' at the warm start (frozen
# there; staleness across LM steps is safe). Findings in README.md.
# Run from the repo root: uv run python -m experiments.nn_many_points
import math
import time

import jax
import jax.numpy as jnp
from flax import nnx
from jax import config
from jax.flatten_util import ravel_pytree
from nlls_gram import UnderdeterminedLevenbergMarquardt, nystrom_preconditioner

from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from neoclassical_growth_nn import (
    DataSettings,
    GrowthPolicyNN,
    GrowthPolicySettings,
    NLLSGramSettings,
    StructuralParameters,
    make_train_t_grid,
    policy_values,
    training_residual,
)

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")

MAX_STEPS = 400
TRAIN_LOSS_THRESHOLD = 1e-15  # the model's certification bar
NYSTROM_RANKS = (64, 256)
POINT_COUNTS = (41, 251, 1001)
CG_MAXITER = 50


def run(n_points, solver_name, rank=None):
    dtype = jnp.float64
    p = StructuralParameters.create(
        dtype,
        a=1 / 3,
        delta=0.1,
        rho_hat=0.11,
        sigma_crra=1.0,
        k_0=1.0,
        A=1.0,
        b_1=1.0,
        b_2=0.0,
    )
    data_set = DataSettings(train_points=n_points)
    t_train = make_train_t_grid(data_set, dtype)
    train_data = {"t": t_train}

    # The model's warm-start construction (scales from the initial condition).
    c_flow_guess = max(1.0 ** (1 / 3) - 0.1 * 1.0, 1e-8)
    default_head_level = jax.nn.softplus(0.0)
    nn_set = GrowthPolicySettings(input_scale=max(float(data_set.train_T), 1.0))
    k_scale = max(1.0, float(1.0 / default_head_level))
    c_scale = max(1.0, float(c_flow_guess / default_head_level))
    model = GrowthPolicyNN(
        nn_set, k_scale=k_scale, c_scale=c_scale, dtype=dtype, rngs=nnx.Rngs(0)
    )
    graphdef, theta_0 = nnx.split(model, nnx.Param)

    def residual_fn(theta, data):
        return training_residual(nnx.merge(graphdef, theta), data, p)

    flat_0, unravel = ravel_pytree(theta_0)
    m = int(residual_fn(theta_0, train_data).shape[0])

    dual_preconditioner = None
    implicit_preconditioner = None
    if solver_name == "cg":
        # NTK dual matvec J J' at the warm start via linearize + transpose;
        # the (m, k) block contract comes from vmapping columns.
        _, jvp = jax.linearize(lambda f: residual_fn(unravel(f), train_data), flat_0)
        vjp = jax.linear_transpose(jvp, flat_0)

        def dual_matvec(V):
            apply_one = lambda v: jvp(vjp(v)[0])  # noqa: E731
            if V.ndim == 1:
                return apply_one(V)
            return jax.vmap(apply_one, in_axes=1, out_axes=1)(V)

        pre = nystrom_preconditioner(
            dual_matvec, m, min(rank, m), jax.random.PRNGKey(0)
        )
        dual_preconditioner = pre
        implicit_preconditioner = pre  # single-arg call applies the undamped inverse

    lm = NLLSGramSettings()
    solver = UnderdeterminedLevenbergMarquardt(
        residual_fn,
        linear_solver=solver_name,
        iterative_maxiter=CG_MAXITER,
        init_damping=lm.init_damping,
        damping_decrease=lm.damping_decrease,
        damping_increase=lm.damping_increase,
        geodesic_acceleration=lm.geodesic_acceleration,
        geodesic_acceptance_ratio=lm.geodesic_acceptance_ratio,
        dual_preconditioner=dual_preconditioner,
        implicit_preconditioner=implicit_preconditioner,
    )
    atol = math.sqrt(TRAIN_LOSS_THRESHOLD)

    compile_start = time.perf_counter()
    jax.block_until_ready(solver.solve(theta_0, train_data, max_steps=1, atol=atol))
    compile_time = time.perf_counter() - compile_start

    start = time.perf_counter()
    result = solver.solve(theta_0, train_data, max_steps=MAX_STEPS, atol=atol)
    jax.block_until_ready(result)
    train_time = time.perf_counter() - start

    trained = nnx.merge(graphdef, result.x)
    test_t = jnp.linspace(0.0, data_set.test_T, data_set.test_points, dtype=dtype)
    values = policy_values(trained, test_t, p)
    benchmark_grid = jnp.linspace(
        0.0, data_set.benchmark_T, data_set.benchmark_points, dtype=dtype
    )
    k_bar, c_bar = neoclassical_growth_benchmark(
        1 / 3, 0.1, 0.11, 1.0, 1.0, benchmark_grid
    )(test_t)
    k_rel = float(jnp.mean(jnp.abs(values["k"] - k_bar) / k_bar))
    c_rel = float(jnp.mean(jnp.abs(values["c"] - c_bar) / c_bar))
    tvc = policy_values(trained, jnp.atleast_1d(jnp.asarray(200.0, dtype)), p)
    log_tvc = float(tvc["log_tvc"][0])
    train_loss = float(result.info.loss)
    certified = train_loss < TRAIN_LOSS_THRESHOLD and log_tvc < 0.0

    label = solver_name if rank is None else f"{solver_name}+nystrom(r={rank})"
    print(
        f"n={n_points:>5} m={m:>5} p={flat_0.size:>5} {label:20} "
        f"certified={str(certified):5} steps={int(result.steps):>4} "
        f"loss={train_loss:.2e} k_err={k_rel:.2e} c_err={c_rel:.2e} "
        f"log_tvc={log_tvc:6.1f} compile={compile_time:5.1f}s "
        f"train={train_time:7.2f}s"
    )


if __name__ == "__main__":
    for n_points in POINT_COUNTS:
        run(n_points, "cholesky")
        for rank in NYSTROM_RANKS:
            run(n_points, "cg", rank=rank)
