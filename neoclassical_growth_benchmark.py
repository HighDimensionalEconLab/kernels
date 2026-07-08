import numpy as np
from scipy.integrate import solve_bvp


# Neoclassical Growth Benchmark solution
def neoclassical_growth_benchmark(
    a, delta, r, sigma_crra, k_0, t_grid, tol=1e-10, max_nodes=100000
):
    # a: capital share
    # delta: depreciation
    # r: discount rate
    # sigma_crra: the constant relative risk aversion coefficient
    # k_0: initial condition for capital
    # t_grid: The grid for time where we want to solve the problem over (0,t_1,...,T)
    #   Truncation pins k(T) = k_ss even though the true path is still ~
    #   (k_ss - k_0) exp(lambda_stable T) below it; that mismatch decays
    #   backward from T at the unstable rate, so keep T ~30 time units past
    #   any window where errors are measured (T=80 for a [0,50] test window
    #   leaves the benchmark accurate to ~1e-11 there).
    # tol/max_nodes: solve_bvp accuracy -- scipy's default tol=1e-3 leaves ~1e-4
    # errors in the "benchmark" itself, polluting every reported solver error

    k_ss = ((delta + r) / a) ** (1 / (a - 1))
    k_T = k_ss

    # Pure numpy: a jax ODE here would inherit the caller's x64 flag, and under
    # float32 solve_bvp can never verify tol=1e-10 and exhausts max_nodes.
    def ODE(t, y):
        k = y[0]
        c = y[1]
        return np.vstack(
            (
                k**a - c - delta * k,
                (c / sigma_crra) * (a * k ** (a - 1) - r - delta),
            )
        )

    def bc(ya, yb):
        return np.array(
            [ya[0] - k_0, yb[0] - k_T]
        )  # boundary condition, k(0) = k_0, k(T) = k_T

    # Saddle-shaped initial guess: exponential approach to the steady state at
    # the linearized stable eigenvalue, with steady-flow consumption. A naive
    # flat guess sends Newton to a spurious delayed-takeoff branch on long
    # horizons.
    f_pp = a * (a - 1) * k_ss ** (a - 2)
    c_ss = k_ss**a - delta * k_ss
    lambda_stable = 0.5 * (r - np.sqrt(r**2 - 4 * c_ss * f_pp / sigma_crra))
    t_np = np.asarray(t_grid, dtype=float)
    k_guess = k_ss + (k_0 - k_ss) * np.exp(lambda_stable * t_np)
    c_guess = k_guess**a - delta * k_guess
    iv = np.vstack((k_guess, c_guess))
    solution = solve_bvp(ODE, bc, t_np, iv, tol=tol, max_nodes=max_nodes)
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed: {solution.message}")

    # the "solution" is an interpolator already, can just unpack
    T_max = t_grid[-1]

    def interpolate_solution(t_grid):
        if t_grid[-1] > T_max:
            raise ValueError("Extrapolation not supported")
        val = solution.sol(t_grid)
        return val[0], val[1]

    return interpolate_solution
