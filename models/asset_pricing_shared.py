from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from models.utilities import ParameterPyTree


class StructuralParameters(ParameterPyTree):
    r: jax.Array
    c: jax.Array
    g: jax.Array
    x_0: jax.Array


def dividends(t, p):
    return (p.x_0 + p.c / p.g) * jnp.exp(p.g * t) - p.c / p.g


# No-bubble fundamental price for the linear asset-pricing ODE
#   mu'(t) = r mu(t) - x(t),  x(t) = (x_0 + c/g) e^{g t} - c/g,
# with r > g. The general solution is mu(t) = mu_f(t) + zeta e^{r t}; the
# transversality (no-bubble) condition selects zeta = 0.
def mu_f_array(t, p):
    return (p.x_0 + p.c / p.g) * jnp.exp(p.g * t) / (p.r - p.g) - p.c / (p.g * p.r)


# Analytic finite-horizon p_0 bias of the dense-grid min-norm selection
# (Matern-1/2 only): the continuum criterion ||p'||^2_H over [0, T] selects
# p*(t) + zeta_T e^{rt}; returns zeta_T, nan for other kernels.
def kernel_horizon_bias_p0(p, kernel_set, T):
    if kernel_set.kernel != "matern12":
        return float("nan")
    r, c, g, x_0 = float(p.r), float(p.c), float(p.g), float(p.x_0)
    ell = kernel_set.ell
    A = g * (x_0 + c / g) / (r - g)  # p*'(t) = A e^{g t}
    E, F = np.exp((g + r) * T), np.exp(2 * r * T)
    I_1, I_2 = (E - 1) / (g + r), (F - 1) / (2 * r)
    return float(
        -(A / r)
        * ((ell * g * r + 1 / ell) * I_1 + 1 + E)
        / ((ell * r * r + 1 / ell) * I_2 + 1 + F)
    )


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100


# Acceptance gates, CLI-reachable so a lower-precision caller can pass its
# own bars; keep train_residual above ~8x machine epsilon for the run dtype.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-8
