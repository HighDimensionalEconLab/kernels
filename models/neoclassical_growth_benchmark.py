import math

import numpy as np
from scipy.integrate import solve_bvp


# Neoclassical Growth Benchmark solution
def neoclassical_growth_benchmark(
    a,
    delta,
    r,
    sigma_crra,
    k_0,
    t_grid,
    tol=1e-10,
    max_nodes=100000,
    *,
    z_0=1.0,
    g=0.0,
):
    # a: capital share
    # delta: depreciation
    # r: discount rate
    # sigma_crra: the constant relative risk aversion coefficient
    # k_0, z_0: initial conditions for capital and Hicks-neutral TFP (Y = z k^a)
    # g: per-capita balanced growth rate; TFP grows at (1-a) g so the
    #   labor-augmenting scale z^{1/(1-a)} grows at g. The static model is
    #   z_0=1, g=0.
    # t_grid: The grid for the normalized boundary-value solve. Truncation pins
    #   kappa(T) = kappa_ss; the necessary buffer past a diagnostic window is
    #   calibration dependent through the stable and unstable eigenvalues.
    # tol/max_nodes: solve_bvp accuracy -- scipy's default tol=1e-3 leaves ~1e-4
    # errors in the "benchmark" itself, polluting every reported solver error

    if g >= r:
        raise ValueError("g must be less than r")
    if r + (sigma_crra - 1.0) * g <= 0.0:
        raise ValueError("r + (sigma_crra - 1) * g must be positive")
    if delta + g <= 0.0:
        raise ValueError("delta + g must be positive")
    marginal_product_ss = delta + r + sigma_crra * g
    if marginal_product_ss <= 0.0:
        raise ValueError("delta + r + sigma_crra * g must be positive")

    # Production is Y = z k^a with the labor-augmenting scale s = z^{1/(1-a)},
    # so Y = s (k/s)^a. In detrended coordinates kappa=k/s and c_tilde=c/s, the
    # balanced-growth boundary is stationary.
    scale_0 = z_0 ** (1.0 / (1.0 - a))
    kappa_0 = k_0 / scale_0
    kappa_ss = (marginal_product_ss / a) ** (1 / (a - 1))
    kappa_T = kappa_ss

    # Pure numpy: a jax ODE here would inherit the caller's x64 flag, and under
    # float32 solve_bvp can never verify tol=1e-10 and exhausts max_nodes.
    def ODE(t, y):
        kappa = y[0]
        c_tilde = y[1]
        return np.vstack(
            (
                kappa**a - c_tilde - (delta + g) * kappa,
                (c_tilde / sigma_crra)
                * (a * kappa ** (a - 1) - r - delta - sigma_crra * g),
            )
        )

    def bc(ya, yb):
        return np.array(
            [ya[0] - kappa_0, yb[0] - kappa_T]
        )  # boundary condition, kappa(0) = kappa_0, kappa(T) = kappa_ss

    # Saddle-shaped initial guess: exponential approach to the steady state at
    # the linearized stable eigenvalue, with steady-flow consumption. A naive
    # flat guess sends Newton to a spurious delayed-takeoff branch on long
    # horizons.
    f_pp = a * (a - 1) * kappa_ss ** (a - 2)
    c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
    if c_tilde_ss <= 0.0:
        raise ValueError("balanced-growth consumption per unit of z must be positive")
    trace = r + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4 * c_tilde_ss * f_pp / sigma_crra
    if discriminant <= 0.0:
        raise ValueError("balanced-growth saddle discriminant must be positive")
    lambda_stable = 0.5 * (trace - np.sqrt(discriminant))
    t_np = np.asarray(t_grid, dtype=float)
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * np.exp(lambda_stable * t_np)
    c_tilde_guess = kappa_guess**a - (delta + g) * kappa_guess
    iv = np.vstack((kappa_guess, c_tilde_guess))
    solution = solve_bvp(ODE, bc, t_np, iv, tol=tol, max_nodes=max_nodes)
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed: {solution.message}")

    # the "solution" is an interpolator already, can just unpack
    T_max = t_grid[-1]

    def interpolate_solution(t_grid):
        if t_grid[-1] > T_max:
            raise ValueError("Extrapolation not supported")
        val = solution.sol(t_grid)
        scale = scale_0 * np.exp(g * np.asarray(t_grid))
        return scale * val[0], scale * val[1]

    return interpolate_solution


# Closed-form steady state of the static model (z = 1, g = 0) and its exact
# derivatives, the AD-free target the solver sensitivities are checked
# against: f'(k_ss) = delta + rho_hat with f(k) = k^a gives
# k_ss = (a / (delta + rho_hat))^(1/(1 - a)) and c_ss = k_ss^a - delta k_ss.
# sigma_crra and k_0 do not enter, so those derivatives are exactly zero.
def neoclassical_steady_state_sensitivities(a, delta, rho_hat):
    mpk_ss = delta + rho_hat
    k_ss = (a / mpk_ss) ** (1.0 / (1.0 - a))
    c_ss = k_ss**a - delta * k_ss
    d_k_ss_d_a = k_ss * (1.0 / (a * (1.0 - a)) + math.log(a / mpk_ss) / (1.0 - a) ** 2)
    d_k_ss_d_delta = -k_ss / ((1.0 - a) * mpk_ss)
    # a k_ss^(a-1) - delta = rho_hat collapses the chain rule for c_ss.
    return {
        "k_ss": k_ss,
        "c_ss": c_ss,
        "d_k_ss_d_a": d_k_ss_d_a,
        "d_k_ss_d_delta": d_k_ss_d_delta,
        "d_k_ss_d_rho_hat": d_k_ss_d_delta,
        "d_k_ss_d_sigma_crra": 0.0,
        "d_k_ss_d_k_0": 0.0,
        "d_c_ss_d_a": k_ss**a * math.log(k_ss) + rho_hat * d_k_ss_d_a,
        "d_c_ss_d_delta": rho_hat * d_k_ss_d_delta - k_ss,
        "d_c_ss_d_rho_hat": rho_hat * d_k_ss_d_delta,
        "d_c_ss_d_sigma_crra": 0.0,
        "d_c_ss_d_k_0": 0.0,
    }
