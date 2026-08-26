from __future__ import annotations

from dataclasses import dataclass

import jax

from models.utilities import ParameterPyTree


class StructuralParameters(ParameterPyTree):
    rho_hat: jax.Array
    c: jax.Array
    beta: jax.Array
    kappa: jax.Array
    x_0: jax.Array


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100
    validation_points: int = 300


# Acceptance gates, CLI-reachable so a lower-precision caller can pass its
# own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-8
