from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.integrate import solve_bvp

from multicountry_growth import MulticountryParameters


def multicountry_growth_benchmark(
    k_0,
    a_0,
    t_grid,
    parameters=MulticountryParameters(),
    tol=1e-8,
    max_nodes=100000,
):
    if parameters.num_countries != 2:
        raise ValueError("the benchmark currently requires two countries")
    k_0 = np.asarray(k_0, dtype=float)
    a_0 = np.atleast_1d(np.asarray(a_0, dtype=float))
    t_grid = np.asarray(t_grid, dtype=float)
    if k_0.shape != (2,) or np.any(k_0 <= 0.0):
        raise ValueError("k_0 must contain two positive capital stocks")
    if a_0.shape != (1,):
        raise ValueError("a_0 must contain the first country's NFA position")
    if t_grid.ndim != 1 or t_grid.size < 2:
        raise ValueError("t_grid must be one-dimensional with at least two points")
    if t_grid[0] != 0.0 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must start at zero and be strictly increasing")

    alpha = parameters.alpha
    gamma = parameters.gamma
    delta = parameters.delta
    r = parameters.r
    elasticity = parameters.eta
    efficiency = parameters.chi
    populations = np.asarray(parameters.populations, dtype=float)
    productivities = np.asarray(parameters.productivities, dtype=float)
    q_ss = np.full(2, 1.0 / elasticity)
    mpk_ss = delta + r / elasticity
    k_ss = (alpha * productivities / mpk_ss) ** (1.0 / (1.0 - alpha))
    output_ss = productivities * k_ss**alpha
    scale = float(np.sum(populations * output_ss) / np.sum(populations))

    def steady(log_relative_weight):
        consumption_weights = np.array([1.0, np.exp(log_relative_weight / gamma)])
        investment_ss = delta * k_ss
        resources_ss = output_ss - investment_ss
        aggregate_resources = float(np.sum(populations * resources_ss))
        weight_sum = float(np.sum(populations * consumption_weights))
        consumption_ss = consumption_weights * aggregate_resources / weight_sum
        assets_ss = (consumption_ss - resources_ss) / r
        mu_ss = (weight_sum / aggregate_resources) ** gamma
        return {
            "k": k_ss,
            "q": q_ss,
            "a": assets_ss,
            "c": consumption_ss,
            "x": investment_ss,
            "output": output_ss,
            "resources": resources_ss,
            "marginal_wealth": mu_ss,
        }

    def allocations(k, q, log_relative_weight):
        consumption_weights = np.array([1.0, np.exp(log_relative_weight / gamma)])[
            :, None
        ]
        investment_rate = (efficiency * elasticity * q) ** (1.0 / (1.0 - elasticity))
        investment = k * investment_rate
        output = productivities[:, None] * k**alpha
        mpk = alpha * productivities[:, None] * k ** (alpha - 1.0)
        resources = output - investment
        aggregate_resources = np.sum(populations[:, None] * resources, axis=0)
        weight_sum = float(np.sum(populations * consumption_weights[:, 0]))
        consumption = consumption_weights * aggregate_resources[None, :] / weight_sum
        mu = (weight_sum / aggregate_resources) ** gamma
        capital_drift = k * (efficiency * investment_rate**elasticity - delta)
        q_intercept = (
            delta
            - (1.0 - elasticity) * efficiency * investment_rate**elasticity
            - mpk / q
        )
        rate_denominator = aggregate_resources / gamma + np.sum(
            populations[:, None] * investment / (1.0 - elasticity), axis=0
        )
        rate_numerator = (
            np.sum(
                populations[:, None] * (mpk - investment_rate) * capital_drift,
                axis=0,
            )
            + r * aggregate_resources / gamma
            - np.sum(
                populations[:, None] * investment * q_intercept / (1.0 - elasticity),
                axis=0,
            )
        )
        interest_rate = rate_numerator / rate_denominator
        q_drift = q * (interest_rate[None, :] + q_intercept)
        resource_drift = np.sum(
            populations[:, None]
            * (
                (mpk - investment_rate) * capital_drift
                - investment
                * (interest_rate[None, :] + q_intercept)
                / (1.0 - elasticity)
            ),
            axis=0,
        )
        return {
            "investment_rate": investment_rate,
            "investment": investment,
            "output": output,
            "mpk": mpk,
            "resources": resources,
            "aggregate_resources": aggregate_resources,
            "consumption": consumption,
            "marginal_wealth": mu,
            "capital_drift": capital_drift,
            "q_intercept": q_intercept,
            "interest_rate": interest_rate,
            "tobin_q_drift": q_drift,
            "resource_drift": resource_drift,
            "rate_denominator": rate_denominator,
        }

    def transformed_values(state, log_relative_weight):
        k = k_ss[:, None] * np.exp(state[:2])
        q = q_ss[:, None] * np.exp(state[2:4])
        a_1 = scale * state[4]
        values = allocations(k, q, log_relative_weight)
        budget_1 = (
            values["interest_rate"] * a_1
            + values["resources"][0]
            - values["consumption"][0]
        )
        drift = np.vstack(
            (
                values["capital_drift"] / k,
                values["tobin_q_drift"] / q,
                budget_1[None, :] / scale,
            )
        )
        return drift, values

    def transformed_ode(_, state, relative_weight_parameter):
        return transformed_values(state, float(relative_weight_parameter[0]))[0]

    def steady_transformed(log_relative_weight):
        values = steady(log_relative_weight)
        return np.concatenate((np.zeros(4), values["a"][:1] / scale))

    def transformed_jacobian(log_relative_weight):
        state_ss = steady_transformed(log_relative_weight)
        jacobian = np.empty((5, 5))
        for column in range(5):
            step = 1e-6 * max(1.0, abs(state_ss[column]))
            direction = np.zeros(5)
            direction[column] = step
            plus = transformed_values(
                (state_ss + direction)[:, None], log_relative_weight
            )[0][:, 0]
            minus = transformed_values(
                (state_ss - direction)[:, None], log_relative_weight
            )[0][:, 0]
            jacobian[:, column] = (plus - minus) / (2.0 * step)
        return jacobian

    def terminal_projection(log_relative_weight):
        jacobian = transformed_jacobian(log_relative_weight)
        _, vectors, stable_count = scipy.linalg.schur(
            jacobian,
            output="real",
            sort=lambda real, imaginary: real < -1e-8,
        )
        if stable_count != 2:
            raise RuntimeError(f"expected two stable roots, found {stable_count}")
        return vectors[:, stable_count:]

    h_0 = np.log(k_0 / k_ss)
    decay = np.exp(-0.08 * t_grid)
    h_guess = h_0[:, None] * decay[None, :]
    v_guess = -0.5 * h_guess
    if np.allclose(productivities, productivities[0]) and np.allclose(
        populations, populations[0]
    ):
        resources_ss = float(output_ss[0] - delta * k_ss[0])
        c_1_guess = resources_ss + r * float(a_0[0])
        ratio_guess = 2.0 * resources_ss / c_1_guess - 1.0
        log_weight_guess = gamma * np.log(ratio_guess) if ratio_guess > 0.0 else 0.0
    else:
        log_weight_guess = 0.0
    a_ss_guess = steady(log_weight_guess)["a"][0]
    a_guess = a_ss_guess + (a_0[0] - a_ss_guess) * decay
    initial_guess = np.vstack((h_guess, v_guess, a_guess[None, :] / scale))

    def boundary_condition(initial, terminal, relative_weight_parameter):
        log_relative_weight = float(relative_weight_parameter[0])
        terminal_deviation = terminal - steady_transformed(log_relative_weight)
        projection = terminal_projection(log_relative_weight)
        return np.concatenate(
            (
                initial[:2] - h_0,
                initial[4:5] - a_0 / scale,
                projection.T @ terminal_deviation,
            )
        )

    solution = solve_bvp(
        transformed_ode,
        boundary_condition,
        t_grid,
        initial_guess,
        p=np.array([log_weight_guess]),
        tol=tol,
        max_nodes=max_nodes,
        verbose=0,
    )
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed: {solution.message}")

    log_relative_weight = float(solution.p[0])
    terminal_basis = terminal_projection(log_relative_weight)
    terminal_state = solution.sol(t_grid[-1])
    terminal_residual = terminal_basis.T @ (
        terminal_state - steady_transformed(log_relative_weight)
    )
    fixed_jacobian = transformed_jacobian(log_relative_weight)
    state_ss = steady_transformed(log_relative_weight)
    weight_step = 1e-6 * max(1.0, abs(log_relative_weight))
    weight_column = (
        transformed_values(state_ss[:, None], log_relative_weight + weight_step)[0][
            :, 0
        ]
        - transformed_values(state_ss[:, None], log_relative_weight - weight_step)[0][
            :, 0
        ]
    ) / (2.0 * weight_step)
    full_jacobian = np.block(
        [
            [fixed_jacobian, weight_column[:, None]],
            [np.zeros((1, 5)), np.zeros((1, 1))],
        ]
    )
    eigenvalues = np.linalg.eigvals(full_jacobian)

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        if np.any(points < 0.0) or np.any(points > t_grid[-1]):
            raise ValueError("evaluation points must lie inside the benchmark horizon")
        state = solution.sol(points)
        state_drift, values = transformed_values(state, log_relative_weight)
        k = k_ss[:, None] * np.exp(state[:2])
        q = q_ss[:, None] * np.exp(state[2:4])
        a_1 = scale * state[4]
        assets = np.vstack((a_1, -populations[0] * a_1 / populations[1]))
        asset_drifts = np.vstack(
            (
                scale * state_drift[4],
                -populations[0] * scale * state_drift[4] / populations[1],
            )
        )
        budget_drifts = (
            values["interest_rate"][None, :] * assets
            + values["resources"]
            - values["consumption"]
        )
        q_rates = (
            state_drift[2:4]
            - delta
            + (1.0 - elasticity) * efficiency * values["investment_rate"] ** elasticity
            + values["mpk"] / q
        )
        return {
            "t": points,
            "k": k.T,
            "a": assets.T,
            "q": q.T,
            "c": values["consumption"].T,
            "x": values["investment"].T,
            "output": values["output"].T,
            "resources": values["resources"].T,
            "current_account": asset_drifts.T,
            "interest_rate": values["interest_rate"],
            "marginal_wealth": values["marginal_wealth"],
            "omitted_budget_gap": (asset_drifts[1] - budget_drifts[1]) / scale,
            "resource_gap": np.sum(
                populations[:, None] * (values["resources"] - values["consumption"]),
                axis=0,
            )
            / scale,
            "asset_clearing_gap": np.sum(populations[:, None] * assets, axis=0) / scale,
            "interest_rate_index_gap": values["interest_rate"]
            - (r + gamma * values["resource_drift"] / values["aggregate_resources"]),
            "interest_rate_q_gap": q_rates - values["interest_rate"][None, :],
            "rate_denominator": values["rate_denominator"],
        }

    diagnostics = {
        "log_relative_weight": log_relative_weight,
        "relative_weight": float(np.exp(log_relative_weight)),
        "boundary_residual_max": float(
            np.max(
                np.abs(
                    boundary_condition(solution.y[:, 0], solution.y[:, -1], solution.p)
                )
            )
        ),
        "terminal_projection_max": float(np.max(np.abs(terminal_residual))),
        "stable_roots": int(np.sum(eigenvalues.real < -1e-8)),
        "unstable_roots": int(np.sum(eigenvalues.real > 1e-8)),
        "center_roots": int(np.sum(np.abs(eigenvalues.real) <= 1e-8)),
        "minimum_rate_denominator": float(np.min(evaluate(t_grid)["rate_denominator"])),
        "bvp_nodes": int(solution.x.size),
    }
    return evaluate, diagnostics
