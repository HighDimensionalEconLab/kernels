from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from models.kernels import IntegratedPath, integrated_kernel_paths


# Alpha first (the metric penalizes the leading flat coordinates).
class Theta(NamedTuple):
    alpha: jax.Array
    p_0: jax.Array


def dividends(t, p):
    return (p["x_0"] + p["c"] / p["g"]) * jnp.exp(p["g"] * t) - p["c"] / p["g"]


# No-arbitrage residual dp/dt = r p - x along the dividend path.
def residual_fn(theta, args, p):
    paths = integrated_kernel_paths(
        {"price": IntegratedPath(theta.alpha, theta.p_0)}, args["K"], args["K_tilde"]
    )
    price, dprice_dt = paths["price"]
    return dprice_dt - (p["r"] * price - dividends(args["t"], p))


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 40.0
    train_points: int = 41
    train_points_list: list[float] | None = None
    test_T: float = 50.0
    test_points: int = 100


# Acceptance gates, CLI-reachable so a float32 caller can pass its own bar
# (the float32 replication driver does).
@dataclass(frozen=True)
class ToleranceSettings:
    train_residual: float = 1e-8
