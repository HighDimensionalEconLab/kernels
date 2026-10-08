from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import erf

from models.kernels import integrated_kernel_matrices
from replication.utilities import configure_process, style

PREFIX = "kernel_interpolation_example"

# The ridge is used only where the ridgeless Gram solve is unusable in
# float64: the penalty is ridge * alpha' (K + jitter I) alpha.
RIDGE = 1e-6
JITTER = 1e-6

# (kernel, ell, ridged, label); the same length scales in both rows.
PANELS = [
    ("matern12", 0.25, False, r"Matérn-1/2, $\ell = 0.25$"),
    ("matern12", 5.0, False, r"Matérn-1/2, $\ell = 5$"),
    ("matern12", 100.0, False, r"Matérn-1/2, $\ell = 100$"),
    ("rbf", 0.25, False, r"Gaussian, $\ell = 0.25$"),
    ("rbf", 5.0, False, r"Gaussian, $\ell = 5$"),
    ("rbf", 100.0, True, r"Gaussian, $\ell = 100$, ridge"),
]
COLORS = {"matern12": "tab:blue", "rbf": "tab:red"}
NAMES = {"matern12": "Matérn-1/2", "rbf": "Gaussian"}


# K_tilde(inf, t_j) = int_0^inf k(s, t_j) ds in closed form.
def integrated_kernel_at_infinity(centers, kernel, ell):
    if kernel == "matern12":
        return ell * (2.0 - jnp.exp(-centers / ell))
    if kernel == "rbf":
        return (
            ell * jnp.sqrt(jnp.pi / 2.0) * (1.0 + erf(centers / (jnp.sqrt(2.0) * ell)))
        )
    raise ValueError(kernel)


# x(t) = x_0 + K_tilde(t, centers) alpha fitted to x(t_i) - x_0 at the dates
# t_i > 0 (the t = 0 row is identically zero). Ridgeless: the minimum
# alpha' K alpha interpolant by the KKT system. Ridged: the augmented
# least-squares problem |A alpha - y|^2 + ridge alpha' (K + jitter I) alpha,
# which no longer interpolates. Returns the coefficients, the Gram matrix K,
# and K_tilde on the centers.
def fit(centers, y, kernel, ell, ridged):
    K, K_tilde = integrated_kernel_matrices(centers, centers, kernel, 1.0, ell)
    rows = K_tilde[1:]
    n, m = rows.shape
    if ridged:
        R = jnp.linalg.cholesky(K + JITTER * jnp.eye(m), upper=True)
        augmented = jnp.vstack([rows, jnp.sqrt(RIDGE) * R])
        alpha = jnp.linalg.lstsq(augmented, jnp.concatenate([y, jnp.zeros(m)]))[0]
    else:
        kkt = jnp.block([[K, rows.T], [rows, jnp.zeros((n, n))]])
        alpha = jnp.linalg.solve(kkt, jnp.concatenate([jnp.zeros(m), y]))[:m]
    return alpha, K, K_tilde


# Logistic S-curve from x_0 to x_ss with rate r; the derivative is a bell of
# width about 1/r centered at the inflection log(x_ss / x_0 - 1) / r.
def run(
    output_dir: str = ".results",
    x_0: float = 0.1,
    x_ss: float = 1.0,
    r: float = 0.15,
    train_T: float = 40.0,
    train_points: int = 11,
    test_T: float = 80.0,
    test_points: int = 1601,
    kernel_center: float = 20.0,
):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def x(t):
        return x_ss / (1.0 + (x_ss / x_0 - 1.0) * jnp.exp(-r * t))

    train = jnp.linspace(0.0, train_T, train_points)
    test = jnp.linspace(0.0, test_T, test_points)
    x_train = x(train)
    x_test = x(test)
    in_sample = test <= train_T

    style(figsize=(18, 10), fontsize=15, ticksize=13)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    results = {
        "x_0": x_0,
        "x_ss": x_ss,
        "r": r,
        "inflection": float(jnp.log(x_ss / x_0 - 1.0) / r),
        "transition_time_scale": 1.0 / r,
        "train_T": train_T,
        "train_points": train_points,
        "train_spacing": float(train[1] - train[0]),
        "x_train_T": float(x(jnp.asarray(train_T))),
        "test_T": test_T,
        "test_points": test_points,
        "ridge": RIDGE,
        "jitter": JITTER,
        "kernel_center": kernel_center,
        "panels": {},
    }
    for ax, (kernel, ell, ridged, label) in zip(axes.flat, PANELS):
        alpha, K, K_tilde_train = fit(train, x_train[1:] - x_0, kernel, ell, ridged)
        gram_condition = float(jnp.linalg.cond(K))
        K_test, K_tilde_test = integrated_kernel_matrices(test, train, kernel, 1.0, ell)
        x_hat = x_0 + K_tilde_test @ alpha
        x_T = x(jnp.asarray(train_T))
        asymptote = float(
            x_0 + integrated_kernel_at_infinity(train, kernel, ell) @ alpha
        )
        abs_error = jnp.abs(x_hat - x_test)
        panel = {
            "kernel": kernel,
            "ell": ell,
            "ridged": ridged,
            "gram_condition": gram_condition,
            "max_abs_alpha": float(jnp.max(jnp.abs(alpha))),
            "train_max_abs_error": float(
                jnp.max(jnp.abs(x_0 + K_tilde_train @ alpha - x_train))
            ),
            "train_window_mean_abs_error": float(abs_error[in_sample].mean()),
            "extrapolation_mean_abs_error": float(abs_error[~in_sample].mean()),
            "terminal_slope": float(K_test[in_sample][-1] @ alpha),
            "true_terminal_slope": float(r * x_T * (1.0 - x_T / x_ss)),
            "asymptote": asymptote,
        }
        results["panels"][f"{kernel}_ell{ell:g}".replace(".", "p")] = panel
        print(f"{label}: {panel}")

        color = COLORS[kernel]
        ax.plot(np.asarray(test), np.asarray(x_test), color="black", label="$x(t)$")
        ax.plot(np.asarray(test), np.asarray(x_hat), color=color, label=r"$\hat{x}(t)$")
        ax.plot(
            np.asarray(train),
            np.asarray(x_train),
            "o",
            color="black",
            markersize=4,
            label="training dates",
        )
        ax.axhline(y=x_ss, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(
            y=asymptote,
            color=color,
            linestyle="--",
            linewidth=0.8,
            label=r"learned $\lim_{t \to \infty} \hat{x}(t)$",
        )
        ax.axvline(x=train_T, color="grey", linestyle=":")
        ax.set_title(label)
        ax.text(
            0.97,
            0.05,
            "\n".join(
                [
                    f"error $[0, {train_T:g}]$: {panel['train_window_mean_abs_error']:.1e}",
                    f"error $({train_T:g}, {test_T:g}]$: {panel['extrapolation_mean_abs_error']:.1e}",
                    f"asymptote: {asymptote:.3f} ($x_{{ss}} = {x_ss:g}$)",
                    f"cond $K$: {gram_condition:.1e}",
                ]
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
    for ax in axes[1]:
        ax.set_xlabel("Time")
    for ax in axes[:, 0]:
        ax.set_ylabel("$x(t)$")
    axes[0, 0].legend(loc="upper left", fontsize=12)
    axes[0, 0].set_ylim(0.0, 1.5)
    fig.tight_layout()
    pdf = out / f"{PREFIX}.pdf"
    fig.savefig(pdf, format="pdf")
    plt.close(fig)
    print(f"wrote {pdf}")

    # The kernels themselves, centered at one training date, and their
    # integrals from zero, at the three length scales of the main figure.
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    center = jnp.array([kernel_center])
    for column, ell in enumerate(ell for _, ell, _, _ in PANELS[:3]):
        for kernel in ("matern12", "rbf"):
            K, K_tilde = integrated_kernel_matrices(test, center, kernel, 1.0, ell)
            axes[0, column].plot(
                np.asarray(test),
                np.asarray(K[:, 0]),
                color=COLORS[kernel],
                label=NAMES[kernel],
            )
            axes[1, column].plot(
                np.asarray(test),
                np.asarray(K_tilde[:, 0]),
                color=COLORS[kernel],
                label=NAMES[kernel],
            )
        for ax in axes[:, column]:
            ax.plot(
                np.asarray(train),
                np.zeros(train_points),
                "|",
                color="black",
                markersize=10,
                label="training dates",
            )
            ax.axvline(x=train_T, color="grey", linestyle=":")
        axes[0, column].set_title(rf"$\ell = {ell:g}$")
        axes[1, column].set_xlabel("Time")
    axes[0, 0].set_ylabel(rf"$k(t, {kernel_center:g})$")
    axes[1, 0].set_ylabel(
        rf"$\tilde{{K}}(t, {kernel_center:g}) = \int_0^t k(s, {kernel_center:g})\,ds$"
    )
    axes[0, 0].legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    pdf_kernels = out / f"{PREFIX}_kernels.pdf"
    fig.savefig(pdf_kernels, format="pdf")
    plt.close(fig)
    print(f"wrote {pdf_kernels}")

    (out / f"{PREFIX}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    configure_process()
    jsonargparse.CLI(run)
