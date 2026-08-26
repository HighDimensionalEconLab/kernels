from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from tinydiffeq import solve_bvp

from models.open_economy_growth_shared import (
    capital_drift,
    consumption_annuity_rate,
    consumption_growth_rate,
    current_account,
    investment_expenditure,
    production,
    q_drift,
    stable_tail_terms,
    steady_state,
)

# One joint BVP in the state {k, q, a} with the consumption level c_0 as the
# solved unknown parameter: c(t) = c_0 e^{g_c t} on the permanent-income path,
# the asset equation a' = r a + k^alpha - k s(q) - c closes the system, and
# pinning a(0) = a_0 together with the terminal annuity condition
# a(T) = c(T)/annuity - tail(h(T)) imposes the intertemporal budget exactly.
# The solve relies on the drivers' always-on x64 setup -- under float32 the
# collocation tolerance clamps far above tol.


# max_nodes is the static padded-mesh budget and must exceed the refined mesh
# (~1100 nodes at tol=1e-9 across the calibrated variants).
@struct.dataclass
class BenchmarkParameters:
    T: float = 200.0
    tol: float = 1e-9
    initial_points: int = struct.field(pytree_node=False, default=241)
    max_nodes: int = struct.field(pytree_node=False, default=1536)


def state_field(t, y, z, args):
    p, _ = args
    k, q, a = y["k"], y["q"], y["a"]
    c = z[0] * jnp.exp(consumption_growth_rate(p) * t)
    return {
        "k": capital_drift(k, q, p),
        "q": q_drift(k, q, 1.0, p),
        "a": current_account(k, a, q, c, 1.0, p),
    }


def state_bc(ya, yb, z, args):
    p, T = args
    annuity_rate = consumption_annuity_rate(p)
    steady = steady_state(p)
    tail = stable_tail_terms(p)
    h_terminal = jnp.log(yb["k"] / steady["k"])
    v_terminal = jnp.log(yb["q"] / steady["q"])
    tail_resources = steady["g"] / p.r_star + tail["resource_loading"] * h_terminal / (
        p.r_star - tail["stable_root"]
    )
    c_terminal = z[0] * jnp.exp(consumption_growth_rate(p) * T)
    return jnp.array(
        [
            ya["k"] - p.k_0,
            ya["a"] - p.a_0,
            v_terminal - tail["stable_slope"] * h_terminal,
            yb["a"] - (c_terminal / annuity_rate - tail_resources),
        ]
    )


@jax.jit
def solve_transition_bvp(p, bench_set):
    steady = steady_state(p)
    tail = stable_tail_terms(p)
    mesh = jnp.linspace(0.0, bench_set.T, bench_set.initial_points)
    # Saddle-shaped (k, q) guess on the linearized stable manifold. Given
    # (k, q), the collocation system is linear in (a, c_0), so a flat asset
    # guess and a steady-flow consumption guess suffice.
    h_guess = jnp.log(p.k_0 / steady["k"]) * jnp.exp(tail["stable_root"] * mesh)
    y_guess = {
        "k": steady["k"] * jnp.exp(h_guess),
        "q": steady["q"] * jnp.exp(tail["stable_slope"] * h_guess),
        "a": jnp.full(bench_set.initial_points, p.a_0),
    }
    z_guess = jnp.atleast_1d(
        consumption_annuity_rate(p) * (p.a_0 + steady["g"] / p.r_star)
    )
    return solve_bvp(
        state_field,
        state_bc,
        mesh,
        y_guess,
        z_guess,
        args=(p, bench_set.T),
        tol=bench_set.tol,
        max_nodes=bench_set.max_nodes,
    )


def open_economy_growth_benchmark(p, bench_set=BenchmarkParameters()):
    consumption_growth = consumption_growth_rate(p)
    annuity_rate = consumption_annuity_rate(p)
    solution = solve_transition_bvp(p, bench_set)
    if not solution.ok:
        raise RuntimeError(f"solve_bvp failed: status {int(solution.status)}")
    c_0 = float(solution.z[0])
    if c_0 <= 0.0:
        raise ValueError(
            "initial assets and transition resources imply nonpositive c_0"
        )

    # evaluate is pure jnp (jit/grad/vmap-safe); the Hermite spline clamps
    # flat outside the mesh span, so callers stay inside the horizon.
    def evaluate(points):
        points = jnp.asarray(points)
        state = solution(points)
        k, q, a = state["k"], state["q"], state["a"]
        c = c_0 * jnp.exp(consumption_growth * points)
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": q,
            "c": c,
            "x": investment_expenditure(k, q, p),
            "output": production(k, 1.0, p),
            "current_account": current_account(k, a, q, c, 1.0, p),
        }

    steady = steady_state(p)
    tail = stable_tail_terms(p)
    terminal = evaluate(np.asarray([bench_set.T]))
    h_terminal = float(np.log(terminal["k"][0] / steady["k"]))
    tail_resources = float(
        steady["g"] / p.r_star
        + tail["resource_loading"] * h_terminal / (p.r_star - tail["stable_root"])
    )
    a_terminal = float(terminal["a"][0])
    c_terminal = float(terminal["c"][0])
    diagnostics = {
        "c_0": c_0,
        "consumption_growth": consumption_growth,
        "a_terminal": a_terminal,
        "stable_residual": float(
            np.log(terminal["q"][0] / steady["q"]) - tail["stable_slope"] * h_terminal
        ),
        "present_value_residual": float(
            c_terminal - annuity_rate * (a_terminal + tail_resources)
        ),
        "bvp_nodes": int(solution.num_nodes),
    }
    return evaluate, diagnostics
