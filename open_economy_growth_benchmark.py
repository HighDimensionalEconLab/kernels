from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_bvp, solve_ivp

from open_economy_growth import ModelParameters


def open_economy_growth_benchmark(
    k_0,
    a_0,
    t_grid,
    parameters=ModelParameters(),
    productivity=None,
    tol=1e-9,
    max_nodes=100000,
):
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1 or len(t_grid) < 2:
        raise ValueError(
            "t_grid must be a one-dimensional array with at least 2 points"
        )
    if t_grid[0] != 0.0 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must start at zero and be strictly increasing")
    if k_0 <= 0.0:
        raise ValueError("k_0 must be positive")

    productivity = (
        (lambda t: np.ones_like(t, dtype=float))
        if productivity is None
        else productivity
    )
    alpha = parameters.alpha
    delta = parameters.delta
    interest_rate = parameters.interest_rate
    installation_elasticity = parameters.installation_elasticity
    installation_efficiency = parameters.installation_efficiency

    q_ss = 1.0 / installation_elasticity
    marginal_product_ss = delta + interest_rate / installation_elasticity
    k_ss = (alpha / marginal_product_ss) ** (1.0 / (1.0 - alpha))
    output_ss = k_ss**alpha
    g_ss = output_ss - delta * k_ss
    capital_loading = delta * installation_elasticity / (1.0 - installation_elasticity)
    q_loading = (1.0 - alpha) * (interest_rate + installation_elasticity * delta)
    stable_root = 0.5 * (
        interest_rate - np.sqrt(interest_rate**2 + 4.0 * capital_loading * q_loading)
    )
    stable_slope = stable_root / capital_loading
    resource_loading = (
        alpha * output_ss
        - delta * k_ss
        - delta * k_ss * stable_slope / (1.0 - installation_elasticity)
    )

    def investment_rate(tobin_q):
        return (installation_efficiency * installation_elasticity * tobin_q) ** (
            1.0 / (1.0 - installation_elasticity)
        )

    def ode(t, values):
        k, tobin_q = values
        z = np.asarray(productivity(t), dtype=float)
        s = investment_rate(tobin_q)
        dk_dt = k * (installation_efficiency * s**installation_elasticity - delta)
        dq_dt = (
            interest_rate
            + delta
            - (1.0 - installation_elasticity)
            * installation_efficiency
            * s**installation_elasticity
        ) * tobin_q - alpha * z * k ** (alpha - 1.0)
        return np.vstack((dk_dt, dq_dt))

    def boundary_condition(initial, terminal):
        h_terminal = np.log(terminal[0] / k_ss)
        v_terminal = np.log(terminal[1] / q_ss)
        return np.array([initial[0] - k_0, v_terminal - stable_slope * h_terminal])

    h_guess = np.log(k_0 / k_ss) * np.exp(stable_root * t_grid)
    v_guess = stable_slope * h_guess
    initial_guess = np.vstack((k_ss * np.exp(h_guess), q_ss * np.exp(v_guess)))
    solution = solve_bvp(
        ode,
        boundary_condition,
        t_grid,
        initial_guess,
        tol=tol,
        max_nodes=max_nodes,
    )
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed: {solution.message}")

    terminal_time = float(t_grid[-1])

    def domestic_resources(t):
        k, tobin_q = solution.sol(t)
        z = np.asarray(productivity(t), dtype=float)
        return z * k**alpha - k * investment_rate(tobin_q)

    discounted_resources = quad(
        lambda t: np.exp(-interest_rate * t) * float(domestic_resources(t)),
        0.0,
        terminal_time,
        epsabs=tol,
        epsrel=tol,
        limit=500,
    )[0]
    k_terminal, q_terminal = solution.sol(terminal_time)
    h_terminal = np.log(k_terminal / k_ss)
    tail_resources = g_ss / interest_rate + resource_loading * h_terminal / (
        interest_rate - stable_root
    )
    c = interest_rate * (
        a_0
        + discounted_resources
        + np.exp(-interest_rate * terminal_time) * tail_resources
    )
    if c <= 0.0:
        raise ValueError("initial assets and transition resources imply nonpositive c")

    a_terminal = (c - g_ss) / interest_rate - resource_loading * h_terminal / (
        interest_rate - stable_root
    )

    def asset_ode(t, asset):
        return interest_rate * asset + float(domestic_resources(t)) - c

    asset_solution = solve_ivp(
        asset_ode,
        (terminal_time, 0.0),
        np.array([a_terminal]),
        rtol=tol,
        atol=tol,
        dense_output=True,
        max_step=max(terminal_time / 500.0, 1e-3),
    )
    if not asset_solution.success:
        raise RuntimeError(f"asset solve failed: {asset_solution.message}")

    a_0_gap = float(asset_solution.sol(0.0)[0] - a_0)

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        if np.any(points < 0.0) or np.any(points > terminal_time):
            raise ValueError("evaluation points must lie inside the benchmark horizon")
        k, tobin_q = solution.sol(points)
        a = asset_solution.sol(points)[0]
        z = np.asarray(productivity(points), dtype=float)
        s = investment_rate(tobin_q)
        x = k * s
        output = z * k**alpha
        current_account = interest_rate * a + output - c - x
        return {
            "t": points,
            "k": k,
            "a": a,
            "q": tobin_q,
            "c": np.full_like(points, c),
            "x": x,
            "output": output,
            "current_account": current_account,
        }

    diagnostics = {
        "c": c,
        "a_terminal": a_terminal,
        "a_0_gap": a_0_gap,
        "stable_residual": float(np.log(q_terminal / q_ss) - stable_slope * h_terminal),
        "present_value_residual": float(
            c
            - interest_rate * a_terminal
            - g_ss
            - interest_rate
            * resource_loading
            * h_terminal
            / (interest_rate - stable_root)
        ),
        "bvp_nodes": int(solution.x.size),
    }
    return evaluate, diagnostics
