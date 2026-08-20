from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from tinydiffeq import hermite_interpolate, solve_bvp

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
# The collocation solve runs in float64 (float32 clamps the tolerance far
# above tol) and fails loudly on a downgraded dtype.

MAX_NODES = 2048


def state_field(t, y, z, args):
    alpha, delta, eta, chi = args["alpha"], args["delta"], args["eta"], args["chi"]
    r = args["interest_rate"]
    k, q, a = y["k"], y["q"], y["a"]
    s = (chi * eta * q) ** (1.0 / (1.0 - eta))
    c = z[0] * jnp.exp(args["consumption_growth"] * t)
    return {
        "k": k * (chi * s**eta - delta),
        "q": (r + delta - (1.0 - eta) * chi * s**eta) * q - alpha * k ** (alpha - 1.0),
        "a": r * a + k**alpha - k * s - c,
    }


def state_bc(ya, yb, z, args):
    r = args["interest_rate"]
    h_terminal = jnp.log(yb["k"] / args["k_ss"])
    v_terminal = jnp.log(yb["q"] / args["q_ss"])
    tail_resources = args["g_ss"] / r + args["resource_loading"] * h_terminal / (
        r - args["stable_root"]
    )
    c_terminal = z[0] * jnp.exp(args["consumption_growth"] * args["terminal_time"])
    return jnp.array(
        [
            ya["k"] - args["k_0"],
            ya["a"] - args["a_0"],
            v_terminal - args["stable_slope"] * h_terminal,
            yb["a"] - (c_terminal / args["annuity_rate"] - tail_resources),
        ]
    )


@jax.jit
def solve_transition_bvp(mesh, y_guess, z_guess, args, tol):
    return solve_bvp(
        state_field,
        state_bc,
        mesh,
        y_guess,
        z_guess,
        args=args,
        tol=tol,
        max_nodes=MAX_NODES,
    )


def open_economy_growth_benchmark(
    k_0,
    a_0,
    t_grid,
    parameters=ModelParameters(),
    tol=1e-9,
):
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid[0] != 0.0 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must start at zero and be strictly increasing")

    alpha = parameters.alpha
    delta = parameters.delta
    interest_rate = parameters.interest_rate
    eta = parameters.eta
    chi = parameters.chi
    consumption_growth = consumption_growth_rate(parameters)
    annuity_rate = consumption_annuity_rate(parameters)
    if annuity_rate <= 0.0:
        raise ValueError("r_star - consumption growth must be positive")

    q_ss = 1.0 / eta
    marginal_product_ss = delta + interest_rate / eta
    k_ss = (alpha / marginal_product_ss) ** (1.0 / (1.0 - alpha))
    output_ss = k_ss**alpha
    g_ss = output_ss - delta * k_ss
    capital_loading = delta * eta / (1.0 - eta)
    q_loading = (1.0 - alpha) * (interest_rate + eta * delta)
    stable_root = 0.5 * (
        interest_rate - np.sqrt(interest_rate**2 + 4.0 * capital_loading * q_loading)
    )
    stable_slope = stable_root / capital_loading
    resource_loading = (
        alpha * output_ss - delta * k_ss - delta * k_ss * stable_slope / (1.0 - eta)
    )
    terminal_time = float(t_grid[-1])

    # Saddle-shaped (k, q) guess on the linearized stable manifold. Given
    # (k, q), the collocation system is linear in (a, c_0), so a flat asset
    # guess and a steady-flow consumption guess suffice.
    h_guess = np.log(k_0 / k_ss) * np.exp(stable_root * t_grid)
    y_guess = {
        "k": jnp.asarray(k_ss * np.exp(h_guess)),
        "q": jnp.asarray(q_ss * np.exp(stable_slope * h_guess)),
        "a": jnp.asarray(np.full(t_grid.size, float(a_0))),
    }
    z_guess = jnp.asarray([annuity_rate * (a_0 + g_ss / interest_rate)])
    args = {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in dict(
            alpha=alpha,
            delta=delta,
            interest_rate=interest_rate,
            eta=eta,
            chi=chi,
            consumption_growth=consumption_growth,
            annuity_rate=annuity_rate,
            k_0=k_0,
            a_0=a_0,
            k_ss=k_ss,
            q_ss=q_ss,
            g_ss=g_ss,
            stable_root=stable_root,
            stable_slope=stable_slope,
            resource_loading=resource_loading,
            terminal_time=terminal_time,
        ).items()
    }
    solution = solve_transition_bvp(
        jnp.asarray(t_grid), y_guess, z_guess, args, jnp.asarray(tol, jnp.float64)
    )
    c_0 = float(solution.z[0])
    if not bool(solution.ok) or np.asarray(solution.z).dtype != np.float64:
        raise RuntimeError(
            f"solve_bvp failed: status {int(solution.status)}, "
            f"dtype {np.asarray(solution.z).dtype}"
        )
    if c_0 <= 0.0:
        raise ValueError(
            "initial assets and transition resources imply nonpositive c_0"
        )

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        if np.any(points < 0.0) or np.any(points > terminal_time):
            raise ValueError("evaluation points must lie inside the benchmark horizon")
        state = hermite_interpolate(
            jnp.asarray(points), solution.t, solution.y, solution.yp
        )
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

    terminal = evaluate(np.asarray([terminal_time]))
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
