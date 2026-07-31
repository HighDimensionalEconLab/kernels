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
    # k_0, z_0: initial conditions for capital and the growing scale
    # g: exogenous growth, dz/dt = g z. The legacy model is z_0=1, g=0.
    # t_grid: The grid for the normalized boundary-value solve. Truncation pins
    #   kappa(T) = kappa_ss; the necessary buffer past a diagnostic window is
    #   calibration dependent through the stable and unstable eigenvalues.
    # tol/max_nodes: solve_bvp accuracy -- scipy's default tol=1e-3 leaves ~1e-4
    # errors in the "benchmark" itself, polluting every reported solver error

    if not 0.0 < a < 1.0:
        raise ValueError("a must lie strictly between zero and one")
    if k_0 <= 0.0 or z_0 <= 0.0:
        raise ValueError("k_0 and z_0 must be positive")
    if sigma_crra <= 0.0:
        raise ValueError("sigma_crra must be positive")
    if g >= r:
        raise ValueError("g must be less than r")
    if r + (sigma_crra - 1.0) * g <= 0.0:
        raise ValueError("r + (sigma_crra - 1) * g must be positive")
    if delta + g <= 0.0:
        raise ValueError("delta + g must be positive")
    marginal_product_ss = delta + r + sigma_crra * g
    if marginal_product_ss <= 0.0:
        raise ValueError("delta + r + sigma_crra * g must be positive")

    # Homogeneous production is Y = z (k/z)^a. In detrended coordinates
    # kappa=k/z and c_tilde=c/z, the balanced-growth boundary is stationary.
    kappa_0 = k_0 / z_0
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
        z = z_0 * np.exp(g * np.asarray(t_grid))
        return z * val[0], z * val[1]

    return interpolate_solution
