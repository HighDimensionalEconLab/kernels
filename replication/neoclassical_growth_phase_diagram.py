from __future__ import annotations

import pathlib

import jax.numpy as jnp
import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
from tinydiffeq import RK4, ConstantStepSize, SaveAt, solve_ode

from models.neoclassical_growth_benchmark import neoclassical_growth_benchmark
from models.neoclassical_growth_shared import StructuralParameters
from replication.utilities import configure_process, style


# Log-utility phase portrait of the appendix system in (x, mu):
#   dot x = x^a - 1/mu - delta x,  dot mu = mu (rho_hat + delta - a x^{a-1}),
# with the dot-x = 0 isocline mu = 1/(x^a - delta x), the dot-mu = 0 vertical
# at the steady state, the stable manifold mu*(x) = 1/c*(x) traced
# parametrically from two benchmark saddle paths, and divergent trajectories
# integrated off the manifold. The defaults are the smooth-branch values of
# the kinked calibration, whose sigma_crra = 1 is the log utility the
# isocline formula requires (k_ss = 2).
def field(state, _, args, p):
    x, mu = state
    a, delta, rho_hat = p["a"], p["delta"], p["rho_hat"]
    return jnp.array(
        [
            x**a - 1.0 / mu - delta * x,
            mu * (rho_hat + delta - a * x ** (a - 1.0)),
        ]
    )


def run(
    output_dir: str = ".results",
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    fontsize: int = 14,
    ticksize: int = 13,
):
    configure_process()
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_ss = (a / (delta + rho_hat)) ** (1.0 / (1.0 - a))
    x_lo, x_hi = 0.2 * x_ss, 2.5 * x_ss

    def mu_nc(x):
        return 1.0 / (x**a - delta * x)

    # Stable manifold from two saddle paths (one from each side of x_ss),
    # merged and sorted in x; mu = 1/c under log utility.
    times = np.linspace(0.0, 120.0, 400)
    p_bench = StructuralParameters.create(
        jnp.float64,
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=1.0,
        z_0=1.0,
        k_0=1.0,
        g=0.0,
        A=1.0,
        b_1=1.0,
        b_2=0.0,
        k_thr=float("inf"),
    )
    branches = [
        neoclassical_growth_benchmark(p_bench.replace(k_0=x_0))(times)
        for x_0 in (x_lo, x_hi)
    ]
    branch_k = np.concatenate([k for k, _ in branches])
    branch_mu = 1.0 / np.concatenate([c for _, c in branches])
    order = np.argsort(branch_k)
    manifold_x = branch_k[order]
    manifold_mu = branch_mu[order]

    def mu_star(x):
        return np.interp(x, manifold_x, manifold_mu)

    x_grid = np.linspace(x_lo, x_hi, 400)
    curve_values = np.concatenate([mu_nc(x_grid), manifold_mu])
    mu_lo, mu_hi = 0.8 * curve_values.min(), 1.15 * curve_values.max()

    p = {"a": a, "delta": delta, "rho_hat": rho_hat}

    # Divergent trajectories: fixed-step forward paths from small mu
    # perturbations off the manifold, truncated at the first saved state
    # outside the plot box (one exterior point is kept so the clipped curve
    # reaches the box edge; NaN/inf blow-up tails fail the box test too).
    def divergent_path(x_0, mu_ratio, T=80.0, dt=0.05):
        solution = solve_ode(
            field,
            RK4(),
            0.0,
            T,
            jnp.asarray([x_0, mu_ratio * mu_star(x_0)]),
            dt_0=dt,
            controller=ConstantStepSize(),
            max_steps=int(round(T / dt)),
            save_at=SaveAt(steps=True),
            p=p,
        )
        path = np.asarray(solution.xs).T
        inside = (
            (path[0] >= x_lo)
            & (path[0] <= x_hi)
            & (path[1] >= mu_lo)
            & (path[1] <= mu_hi)
        )
        exits = np.flatnonzero(~inside)
        return path[:, : exits[0] + 1] if exits.size else path

    style(figsize=(10, 6), fontsize=fontsize, ticksize=ticksize)
    fig, ax = plt.subplots()

    ax.plot(
        x_grid, mu_nc(x_grid), color="green", lw=1.5, label=r"$\mu^{NC}(x): \dot{x}=0$"
    )
    ax.axvline(x_ss, color="blue", lw=1.5, label=r"$\dot{\mu}=0$")
    ax.plot(
        manifold_x,
        manifold_mu,
        color="black",
        ls="--",
        lw=1.5,
        label=r"Stable Manifold: $\mu^{*}(x)$",
    )

    # Flow arrowheads on the manifold, pointing toward the steady state.
    for x_arrow in (0.55 * x_ss, 1.6 * x_ss):
        direction = 1.0 if x_arrow < x_ss else -1.0
        x_tip = x_arrow + direction * 0.02 * (x_hi - x_lo)
        ax.annotate(
            "",
            xy=(x_tip, mu_star(x_tip)),
            xytext=(x_arrow, mu_star(x_arrow)),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        )

    for x_0, mu_ratio in [
        (0.5 * x_ss, 1.06),
        (0.5 * x_ss, 0.94),
        (1.8 * x_ss, 1.06),
        (1.8 * x_ss, 0.94),
        (2.3 * x_ss, 1.04),
    ]:
        path = divergent_path(x_0, mu_ratio)
        ax.plot(path[0], path[1], color="red", ls=":", lw=1.2)
        tip = path.shape[1] * 3 // 5
        ax.annotate(
            "",
            xy=(path[0, tip + 1], path[1, tip + 1]),
            xytext=(path[0, tip], path[1, tip]),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.2),
        )

    # Region labels with (dot x, dot mu) sign glyphs: dot x > 0 above the
    # green isocline, dot mu > 0 right of the blue line. R_1..R_3 sit right
    # of the steady state (top to bottom), R_4..R_6 left (bottom to top),
    # matching the appendix's ordering.
    def region_label(name, x, mu):
        dx = 1.0 if mu > mu_nc(x) else -1.0
        dmu = 1.0 if x > x_ss else -1.0
        ax.annotate(name, (x, mu), fontsize=fontsize)
        span_x, span_mu = 0.035 * (x_hi - x_lo), 0.045 * (mu_hi - mu_lo)
        anchor = (x - 0.2 * span_x, mu - 0.3 * span_mu)
        ax.annotate(
            "",
            xy=(anchor[0] + dx * span_x, anchor[1]),
            xytext=anchor,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )
        ax.annotate(
            "",
            xy=(anchor[0], anchor[1] + dmu * span_mu),
            xytext=anchor,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )

    x_left, x_right = 0.45 * x_ss, 1.7 * x_ss
    green_left, man_left = mu_nc(x_left), mu_star(x_left)
    green_right, man_right = mu_nc(x_right), mu_star(x_right)
    region_label(r"$R_1$", x_right, 1.3 * green_right)
    region_label(r"$R_2$", x_right, np.sqrt(green_right * man_right))
    region_label(r"$R_3$", x_right, 0.8 * man_right)
    region_label(r"$R_4$", x_left, 0.75 * green_left)
    region_label(r"$R_5$", x_left, np.sqrt(green_left * man_left))
    region_label(r"$R_6$", x_left, 1.25 * man_left)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(mu_lo, mu_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\mu$")
    ax.legend(loc="upper right")

    path = out / "phase_diagram_mu_k.pdf"
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    print(f"wrote {path}")


if __name__ == "__main__":
    jsonargparse.CLI(run)
