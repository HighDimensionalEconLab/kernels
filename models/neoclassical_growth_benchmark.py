import numpy as np
from scipy.integrate import solve_bvp


# Reference saddle path by boundary-value solve, truncated by pinning
# kappa(T) = kappa_ss at the end of t_grid. TFP grows at (1 - a) g so the
# labor-augmenting scale z^{1/(1-a)} grows at g; the static model is z_0 = 1,
# g = 0. scipy's default tol=1e-3 leaves ~1e-4 errors in the "benchmark"
# itself, polluting every reported solver error.
def neoclassical_growth_benchmark(
    a,
    delta,
    rho_hat,
    sigma_crra,
    k_0,
    t_grid,
    tol=1e-10,
    max_nodes=100000,
    *,
    z_0=1.0,
    g=0.0,
):
    marginal_product_ss = delta + rho_hat + sigma_crra * g

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
                * (a * kappa ** (a - 1) - rho_hat - delta - sigma_crra * g),
            )
        )

    def bc(ya, yb):
        return np.array([ya[0] - kappa_0, yb[0] - kappa_T])

    # Saddle-shaped initial guess: exponential approach to the steady state at
    # the linearized stable eigenvalue, with steady-flow consumption. A naive
    # flat guess sends Newton to a spurious delayed-takeoff branch on long
    # horizons.
    f_pp = a * (a - 1) * kappa_ss ** (a - 2)
    c_tilde_ss = kappa_ss**a - (delta + g) * kappa_ss
    trace = rho_hat + (sigma_crra - 1.0) * g
    discriminant = trace**2 - 4 * c_tilde_ss * f_pp / sigma_crra
    lambda_stable = 0.5 * (trace - np.sqrt(discriminant))
    t_np = np.asarray(t_grid, dtype=float)
    kappa_guess = kappa_ss + (kappa_0 - kappa_ss) * np.exp(lambda_stable * t_np)
    c_tilde_guess = kappa_guess**a - (delta + g) * kappa_guess
    iv = np.vstack((kappa_guess, c_tilde_guess))
    solution = solve_bvp(ODE, bc, t_np, iv, tol=tol, max_nodes=max_nodes)
    if solution.status != 0:
        raise RuntimeError(f"solve_bvp failed: {solution.message}")

    T_max = t_grid[-1]

    def interpolate_solution(t_grid):
        if t_grid[-1] > T_max:
            raise ValueError("Extrapolation not supported")
        val = solution.sol(t_grid)
        scale = scale_0 * np.exp(g * np.asarray(t_grid))
        return scale * val[0], scale * val[1]

    return interpolate_solution
