import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve


# Stochastic growth benchmark: implicit upwind finite differences for the HJBE
#   rho V = u(c*) + V_x eta (log_z_bar - x) + V_y ((f - c*)/k - delta)
#           + (sigma_z^2 / 2) V_xx
# on the (x, y) = (log z, log k) grid, with c* = (V_y / k)^(-1/sigma_crra)
# from the consumption first-order condition and reflecting boundaries.
def stochastic_growth_benchmark(
    a,
    delta,
    rho_hat,
    sigma_crra,
    eta,
    sigma_z,
    *,
    log_z_bar=0.0,
    num_z_points=41,
    num_k_points=301,
    z_sd_span=5.0,
    k_lo=0.4,
    k_hi=None,
    implicit_time_step=100.0,
    tol=1e-9,
    max_iterations=1000,
):
    # a: capital share; delta: depreciation; rho_hat: discount rate
    # sigma_crra: CRRA coefficient (1.0 selects log utility)
    # eta, sigma_z, log_z_bar: OU log-TFP d x = eta (log_z_bar - x) dt + sigma_z dW
    # z_sd_span: the log z grid covers log_z_bar +- span * ergodic sd
    # k_lo / k_hi: capital grid bounds in levels; k_hi defaults to 1.5 k_ss
    # implicit_time_step: pseudo-timestep of the implicit scheme (large is fine,
    #   the scheme is unconditionally stable)
    # Pure numpy/scipy: independent of the caller's jax x64 flag, like the
    # deterministic solve_bvp benchmark.
    z_bar = np.exp(log_z_bar)
    kappa_ss = ((delta + rho_hat) / a) ** (1.0 / (a - 1.0))
    k_ss = z_bar ** (1.0 / (1.0 - a)) * kappa_ss
    if k_hi is None:
        k_hi = 1.5 * k_ss
    if not k_lo < k_ss < k_hi:
        raise ValueError("the capital grid must bracket the deterministic k_ss")

    ergodic_sd = sigma_z / np.sqrt(2.0 * eta)
    # A degenerate z grid breaks interpolation, so the deterministic limit
    # keeps a small span; rows decouple in z there apart from mean reversion.
    half_width = z_sd_span * ergodic_sd if ergodic_sd > 0.0 else 0.02
    x_grid = np.linspace(log_z_bar - half_width, log_z_bar + half_width, num_z_points)
    y_grid = np.linspace(np.log(k_lo), np.log(k_hi), num_k_points)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    z = np.exp(x_grid)[:, None]
    k = np.exp(y_grid)[None, :]
    f = z * k**a
    c_zero_drift = f - delta * k
    if np.min(c_zero_drift) <= 0.0:
        raise ValueError("f - delta k must be positive on the whole grid")

    def utility(c):
        if sigma_crra == 1.0:
            return np.log(c)
        return (c ** (1.0 - sigma_crra) - 1.0) / (1.0 - sigma_crra)

    def foc_consumption(marginal_value):
        return (marginal_value / k) ** (-1.0 / sigma_crra)

    # x-direction upwind + central diffusion coefficients are policy
    # independent; zeroing the outward boundary legs folds the reflecting
    # ghost node onto the diagonal, keeping generator row sums at zero.
    mu_x = eta * (log_z_bar - x_grid)
    s_x = 0.5 * sigma_z**2 / dx**2
    up_x = np.maximum(mu_x, 0.0) / dx + s_x
    lo_x = np.maximum(-mu_x, 0.0) / dx + s_x
    up_x[-1] = 0.0
    lo_x[0] = 0.0
    up_x_grid = np.broadcast_to(up_x[:, None], f.shape).copy()
    lo_x_grid = np.broadcast_to(lo_x[:, None], f.shape).copy()

    marginal_value_boundary = c_zero_drift ** (-sigma_crra) * k
    v = utility(c_zero_drift) / rho_hat
    n = num_z_points * num_k_points
    identity = sp.identity(n, format="csc")

    # Safeguarded FOC: a pseudo-time transient can dent V's monotonicity at
    # the state-constraint corners, and a nonpositive difference would then
    # strand the cell in a spurious self-consistent zero-drift branch.
    # Flooring the derivative candidate at a tiny positive per-cell scale
    # prescribes aggressive dissaving there, which lifts the dented value and
    # heals the dip; the converged solution is verified strictly monotone.
    derivative_floor = 1e-6

    def upwind_consumption(v, *, require_monotone):
        v_y_forward = np.empty_like(v)
        v_y_forward[:, :-1] = (v[:, 1:] - v[:, :-1]) / dy
        v_y_forward[:, -1] = marginal_value_boundary[:, -1]
        v_y_backward = np.empty_like(v)
        v_y_backward[:, 1:] = (v[:, 1:] - v[:, :-1]) / dy
        v_y_backward[:, 0] = marginal_value_boundary[:, 0]
        floor = derivative_floor * marginal_value_boundary
        # Above the floor, not merely positive: the converged policy must not
        # depend on the safeguard.
        if require_monotone and not (
            (v_y_forward > floor).all() and (v_y_backward > floor).all()
        ):
            raise RuntimeError(
                "converged V is not increasing in k above the safeguard floor"
            )
        c_forward = foc_consumption(np.maximum(v_y_forward, floor))
        c_backward = foc_consumption(np.maximum(v_y_backward, floor))
        drift_forward = (f - c_forward) / k - delta
        drift_backward = (f - c_backward) / k - delta
        # The padded boundary candidates are zero drift in exact arithmetic;
        # imposing that exactly stops float cancellation noise from letting
        # the forward branch outrank backward at the state constraint.
        c_forward[:, -1] = c_zero_drift[:, -1]
        drift_forward[:, -1] = 0.0
        c_backward[:, 0] = c_zero_drift[:, 0]
        drift_backward[:, 0] = 0.0
        use_forward = drift_forward > 0.0
        use_backward = (drift_backward < 0.0) & ~use_forward
        c = np.where(
            use_forward, c_forward, np.where(use_backward, c_backward, c_zero_drift)
        )
        drift = np.where(
            use_forward, drift_forward, np.where(use_backward, drift_backward, 0.0)
        )
        return c, drift

    converged = False
    for iteration in range(max_iterations):
        c, drift = upwind_consumption(v, require_monotone=False)
        up_y = np.maximum(drift, 0.0) / dy
        lo_y = np.maximum(-drift, 0.0) / dy
        diagonal = -(up_x_grid + lo_x_grid + up_y + lo_y)
        generator = sp.diags(
            [
                diagonal.reshape(-1),
                up_y.reshape(-1)[:-1],
                lo_y.reshape(-1)[1:],
                up_x_grid.reshape(-1)[:-num_k_points],
                lo_x_grid.reshape(-1)[num_k_points:],
            ],
            [0, 1, -1, num_k_points, -num_k_points],
            format="csc",
        )
        # Warm up the pseudo-time step: a large step under the lagged policy
        # can jump to a spurious zero-drift fixed point at the state
        # constraint corners, while small early steps track the monotone HJB
        # flow until the policy settles.
        time_step = min(implicit_time_step, 1.5**iteration)
        system = (1.0 / time_step + rho_hat) * identity - generator
        rhs = utility(c).reshape(-1) + v.reshape(-1) / time_step
        v_new = spsolve(system, rhs).reshape(v.shape)
        change = np.max(np.abs(v_new - v))
        v = v_new
        # High curvature blows up the value scale, so the tolerance is
        # relative to it.
        if change < tol * max(1.0, float(np.max(np.abs(v)))):
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"HJBE iteration did not converge: last change {change:.3e} after "
            f"{max_iterations} iterations"
        )
    c, drift = upwind_consumption(v, require_monotone=True)

    interpolator = RegularGridInterpolator(
        (x_grid, y_grid), c, bounds_error=False, fill_value=None
    )

    def interpolate_consumption(log_z, log_k):
        return interpolator(np.stack(np.broadcast_arrays(log_z, log_k), axis=-1))

    return {
        "log_z_grid": x_grid,
        "log_k_grid": y_grid,
        "V": v,
        "c": c,
        "log_k_drift": drift,
        "k_ss": k_ss,
        "kappa_ss": kappa_ss,
        "z_ergodic_sd": ergodic_sd,
        "iterations": iteration + 1,
        "interpolate_consumption": interpolate_consumption,
    }
