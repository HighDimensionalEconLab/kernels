from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import config

# Differentiable ODE integrators for policy rollouts: fixed-grid RK4 and a
# free-stepping adaptive Tsit5. Both operate on array states (scalar or
# vector -- every operation is elementwise and the adaptive error norm
# reduces over components; pytree states would need tree_map plumbing). Both
# are differentiable in both forward and reverse mode through the states,
# which is what an optimizer differentiating through the rollout needs
# (nlls_gram's vjp Jacobian plus the forward-mode directional derivative of
# its geodesic acceleration). `project` (e.g. a positivity clamp) is applied
# to every accepted state.

config.update("jax_enable_x64", True)

# Tsitouras (2011) 5(4) coefficients (FSAL: k7 = f(y5) is the next step's k1).
A21 = 0.161
A31, A32 = -0.008480655492356989, 0.335480655492357
A41, A42, A43 = 2.8971530571054935, -6.359448489975075, 4.3622954328695815
A51, A52, A53, A54 = (
    5.325864828439257,
    -11.748883564062828,
    7.4955393428898365,
    -0.09249506636175525,
)
A61, A62, A63, A64, A65 = (
    5.86145544294642,
    -12.92096931784711,
    8.159367898576159,
    -0.071584973281401,
    -0.028269050394068383,
)
B1, B2, B3, B4, B5, B6 = (
    0.09646076681806523,
    0.01,
    0.4798896504144996,
    1.379008574103742,
    -3.290069515436081,
    2.324710524099774,
)
# embedded 4th-order error coefficients (b - bhat)
E1, E2, E3, E4, E5, E6, E7 = (
    -0.00178001105222577714,
    -0.0008164344596567469,
    0.007880878010261995,
    -0.1447110071732629,
    0.5823571654525552,
    -0.45808210592918697,
    1.0 / 66.0,
)

SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 5.0


def identity_project(y):
    return y


# Classic fixed-step RK4 on the dt grid; returns all n_steps + 1 states.
def rk4_grid(f, y0, n_steps, dt, project=identity_project):
    def step(y, _):
        s1 = f(y)
        s2 = f(project(y + 0.5 * dt * s1))
        s3 = f(project(y + 0.5 * dt * s2))
        s4 = f(project(y + dt * s3))
        y_next = project(y + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4))
        return y_next, y

    y_final, states = jax.lax.scan(step, y0, None, length=n_steps)
    return jnp.concatenate(
        [states, y_final[None] if jnp.ndim(y0) else jnp.atleast_1d(y_final)]
    )


def tsit5_stages(f, y, h, k1=None):
    if k1 is None:
        k1 = f(y)
    k2 = f(y + h * (A21 * k1))
    k3 = f(y + h * (A31 * k1 + A32 * k2))
    k4 = f(y + h * (A41 * k1 + A42 * k2 + A43 * k3))
    k5 = f(y + h * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4))
    k6 = f(y + h * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5))
    y5 = y + h * (B1 * k1 + B2 * k2 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6)
    k7 = f(y5)
    err = h * (E1 * k1 + E2 * k2 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)
    return y5, k7, err


# Free-stepping adaptive Tsit5 over [0, T]: steps adapt freely with no save
# grid, clipped only so no step lands past the horizon (the final accepted
# step hits T exactly). The horizon clip doubles as the guard on step growth:
# without it, a near-flat vector field lets steps quintuple into
# quarter-horizon leaps whose Gauss-Newton linearization stalls a
# trust-region optimizer differentiating through the rollout. Runs a fixed
# budget of n_iters attempts (static shapes for jit); every iteration emits
# its (t, y) -- an accepted step advances (FSAL: its k7 seeds the next
# attempt), a rejection or post-horizon freeze repeats the previous state,
# which downstream least-squares residuals tolerate as a harmless duplicate
# row. If the budget is exhausted before reaching T the outputs are poisoned
# to inf (the caller's residual then rejects rather than reading a
# truncated path). Returns (ts, ys) of length n_iters + 1 including the initial
# state.
#
# The step-size CONTROLLER is wrapped in stop_gradient:
# accept/reject is a non-differentiable branch either way, the gradient of
# ratio**(-1/5) blows up at the exact-zero error of a flat-start policy, and
# the dh/dtheta term only slides sample points along the visited trajectory --
# irrelevant to a residual that must vanish at every state. The states
# themselves remain fully differentiable through the RK stages.
def tsit5_free(
    f, y0, T, n_iters, rtol=1e-6, atol=1e-6, dt0=1.0, project=identity_project
):
    def step(carry, _):
        t, y, h, k1, done = carry
        remaining = jnp.maximum(T - t, 0.0)
        h_eff = jnp.minimum(h, jnp.maximum(remaining, 1e-12))
        y5, k7, err = tsit5_stages(f, y, h_eff, k1)
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y5))
        ratio = jax.lax.stop_gradient(jnp.max(jnp.abs(err) / scale))
        accept = (ratio <= 1.0) | (h_eff <= 1e-10)
        factor = jnp.clip(
            SAFETY * jnp.maximum(ratio, 1e-12) ** (-0.2), MIN_FACTOR, MAX_FACTOR
        )
        h_next = jnp.where(done, h, jax.lax.stop_gradient(h_eff) * factor)

        advance = accept & ~done
        y_new = jnp.where(advance, project(y5), y)
        t_new = jnp.where(advance, t + h_eff, t)
        k1_new = jnp.where(advance, k7, k1)
        done_new = done | (t_new >= T - 1e-9)
        return (t_new, y_new, h_next, k1_new, done_new), (t_new, y_new)

    y0 = jnp.asarray(y0)
    t0 = jnp.asarray(0.0, dtype=jnp.result_type(y0, float))
    carry0 = (t0, y0, jnp.asarray(dt0, t0.dtype), f(y0), jnp.asarray(False))
    (_, _, _, _, done), (ts, ys) = jax.lax.scan(step, carry0, None, length=n_iters)
    ts = jnp.concatenate([t0[None], ts])
    ys = jnp.concatenate([y0[None] if jnp.ndim(y0) else jnp.atleast_1d(y0), ys])
    # budget exhausted before T -> poison (caller's residual rejects)
    return jnp.where(done, ts, jnp.inf), jnp.where(done, ys, jnp.inf)
