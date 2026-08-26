from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

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


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100


# Acceptance gates, CLI-reachable so a lower-precision caller can pass its
# own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-8
