from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from tinydiffeq import solve_bvp

from models.open_economy_growth import (
    ModelParameters,
    consumption_annuity_rate,
    consumption_growth_rate,
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
    alpha, delta, eta, chi = args["alpha"], args["delta"], args["eta"], args["chi"]
    r_star = args["interest_rate"]
    k, q, a = y["k"], y["q"], y["a"]
    s = (chi * eta * q) ** (1.0 / (1.0 - eta))
    c = z[0] * jnp.exp((r_star - args["r"]) / args["gamma"] * t)
    return {
        "k": k * (chi * s**eta - delta),
        "q": (r_star + delta - (1.0 - eta) * chi * s**eta) * q
        - alpha * k ** (alpha - 1.0),
        "a": r_star * a + k**alpha - k * s - c,
    }


def state_bc(ya, yb, z, args):
    alpha, delta, eta = args["alpha"], args["delta"], args["eta"]
    r_star = args["interest_rate"]
    consumption_growth = (r_star - args["r"]) / args["gamma"]
    annuity_rate = r_star - consumption_growth
    q_ss = 1.0 / eta
    k_ss = (alpha / (delta + r_star / eta)) ** (1.0 / (1.0 - alpha))
    g_ss = k_ss**alpha - delta * k_ss
    capital_loading = delta * eta / (1.0 - eta)
    q_loading = (1.0 - alpha) * (r_star + eta * delta)
    stable_root = 0.5 * (
        r_star - jnp.sqrt(r_star**2 + 4.0 * capital_loading * q_loading)
    )
    stable_slope = stable_root / capital_loading
    resource_loading = (
        alpha * k_ss**alpha - delta * k_ss - delta * k_ss * stable_slope / (1.0 - eta)
    )
    h_terminal = jnp.log(yb["k"] / k_ss)
    v_terminal = jnp.log(yb["q"] / q_ss)
    tail_resources = g_ss / r_star + resource_loading * h_terminal / (
        r_star - stable_root
    )
    c_terminal = z[0] * jnp.exp(consumption_growth * args["T"])
    return jnp.array(
        [
            ya["k"] - args["k_0"],
            ya["a"] - args["a_0"],
            v_terminal - stable_slope * h_terminal,
            yb["a"] - (c_terminal / annuity_rate - tail_resources),
        ]
    )


@jax.jit
def solve_transition_bvp(args, bench_set):
    alpha, delta, eta = args["alpha"], args["delta"], args["eta"]
    r_star = args["interest_rate"]
    annuity_rate = r_star - (r_star - args["r"]) / args["gamma"]
    q_ss = 1.0 / eta
    k_ss = (alpha / (delta + r_star / eta)) ** (1.0 / (1.0 - alpha))
    g_ss = k_ss**alpha - delta * k_ss
    capital_loading = delta * eta / (1.0 - eta)
    q_loading = (1.0 - alpha) * (r_star + eta * delta)
    stable_root = 0.5 * (
        r_star - jnp.sqrt(r_star**2 + 4.0 * capital_loading * q_loading)
    )
    stable_slope = stable_root / capital_loading
    mesh = jnp.linspace(0.0, bench_set.T, bench_set.initial_points)
    # Saddle-shaped (k, q) guess on the linearized stable manifold. Given
    # (k, q), the collocation system is linear in (a, c_0), so a flat asset
    # guess and a steady-flow consumption guess suffice.
    h_guess = jnp.log(args["k_0"] / k_ss) * jnp.exp(stable_root * mesh)
    y_guess = {
        "k": k_ss * jnp.exp(h_guess),
        "q": q_ss * jnp.exp(stable_slope * h_guess),
        "a": jnp.full(bench_set.initial_points, args["a_0"]),
    }
    z_guess = jnp.atleast_1d(annuity_rate * (args["a_0"] + g_ss / r_star))
    return solve_bvp(
        state_field,
        state_bc,
        mesh,
        y_guess,
        z_guess,
        args=args,
        tol=bench_set.tol,
        max_nodes=bench_set.max_nodes,
    )


def open_economy_growth_benchmark(
    k_0,
    a_0,
    parameters=ModelParameters(),
    bench_set=BenchmarkParameters(),
):
    alpha = parameters.alpha
    delta = parameters.delta
    interest_rate = parameters.interest_rate
    eta = parameters.eta
    chi = parameters.chi
    consumption_growth = consumption_growth_rate(parameters)
    annuity_rate = consumption_annuity_rate(parameters)

    args = {
        "alpha": alpha,
        "delta": delta,
        "interest_rate": interest_rate,
        "eta": eta,
        "chi": chi,
        "r": parameters.r,
        "gamma": parameters.gamma,
        "k_0": k_0,
        "a_0": a_0,
        "T": bench_set.T,
    }
    solution = solve_transition_bvp(args, bench_set)
    if not solution.ok:
        raise RuntimeError(f"solve_bvp failed: status {int(solution.status)}")
    c_0 = float(solution.z[0])
    if c_0 <= 0.0:
        raise ValueError(
            "initial assets and transition resources imply nonpositive c_0"
        )

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        # The Hermite spline clamps silently outside the mesh span.
        if np.any(points < 0.0) or np.any(points > bench_set.T):
            raise ValueError("evaluation points must lie inside the benchmark horizon")
        state = solution(jnp.asarray(points))
        k = np.asarray(state["k"])
        q = np.asarray(state["q"])
        a = np.asarray(state["a"])
        s = (chi * eta * q) ** (1.0 / (1.0 - eta))
        x = k * s
        output = k**alpha
        c = c_0 * np.exp(consumption_growth * points)
        current_account = interest_rate * a + output - c - x
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": q,
            "c": c,
            "x": x,
            "output": output,
            "current_account": current_account,
        }

    q_ss = 1.0 / eta
    k_ss = (alpha / (delta + interest_rate / eta)) ** (1.0 / (1.0 - alpha))
    g_ss = k_ss**alpha - delta * k_ss
    capital_loading = delta * eta / (1.0 - eta)
    q_loading = (1.0 - alpha) * (interest_rate + eta * delta)
    stable_root = 0.5 * (
        interest_rate - np.sqrt(interest_rate**2 + 4.0 * capital_loading * q_loading)
    )
    stable_slope = stable_root / capital_loading
    resource_loading = (
        alpha * k_ss**alpha - delta * k_ss - delta * k_ss * stable_slope / (1.0 - eta)
    )
    terminal = evaluate(np.asarray([bench_set.T]))
    h_terminal = float(np.log(terminal["k"][0] / k_ss))
    tail_resources = g_ss / interest_rate + resource_loading * h_terminal / (
        interest_rate - stable_root
    )
    a_terminal = float(terminal["a"][0])
    c_terminal = float(terminal["c"][0])
    diagnostics = {
        "c_0": c_0,
        "consumption_growth": consumption_growth,
        "a_terminal": a_terminal,
        "stable_residual": float(
            np.log(terminal["q"][0] / q_ss) - stable_slope * h_terminal
        ),
        "present_value_residual": float(
            c_terminal - annuity_rate * (a_terminal + tail_resources)
        ),
        "bvp_nodes": int(solution.num_nodes),
    }
    return evaluate, diagnostics
