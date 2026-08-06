from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_bvp, solve_ivp

from models.open_economy_growth import (
    ModelParameters,
    consumption_annuity_rate,
    consumption_growth_rate,
)


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

    def investment_rate(q):
        return (chi * eta * q) ** (1.0 / (1.0 - eta))

    def ode(t, values):
        k, q = values
        z = np.asarray(productivity(t), dtype=float)
        s = investment_rate(q)
        dk_dt = k * (chi * s**eta - delta)
        dq_dt = (
            interest_rate + delta - (1.0 - eta) * chi * s**eta
        ) * q - alpha * z * k ** (alpha - 1.0)
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
        k, q = solution.sol(t)
        z = np.asarray(productivity(t), dtype=float)
        return z * k**alpha - k * investment_rate(q)

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
    c_0 = annuity_rate * (
        a_0
        + discounted_resources
        + np.exp(-interest_rate * terminal_time) * tail_resources
    )
    if c_0 <= 0.0:
        raise ValueError(
            "initial assets and transition resources imply nonpositive c_0"
        )

    c_terminal = c_0 * np.exp(consumption_growth * terminal_time)
    a_terminal = c_terminal / annuity_rate - tail_resources

    def asset_ode(t, asset):
        c = c_0 * np.exp(consumption_growth * t)
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
        k, q = solution.sol(points)
        a = asset_solution.sol(points)[0]
        z = np.asarray(productivity(points), dtype=float)
        s = investment_rate(q)
        x = k * s
        output = z * k**alpha
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

    diagnostics = {
        "c_0": c_0,
        "consumption_growth": consumption_growth,
        "a_terminal": a_terminal,
        "a_0_gap": a_0_gap,
        "stable_residual": float(np.log(q_terminal / q_ss) - stable_slope * h_terminal),
        "present_value_residual": float(
            c_terminal - annuity_rate * (a_terminal + tail_resources)
        ),
        "bvp_nodes": int(solution.x.size),
    }
    return evaluate, diagnostics
