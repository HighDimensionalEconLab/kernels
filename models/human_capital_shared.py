from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from nlls_gram import LevenbergMarquardt

from models.utilities import ParameterPyTree


class StructuralParameters(ParameterPyTree):
    a_k: jax.Array
    a_h: jax.Array
    delta_k: jax.Array
    delta_h: jax.Array
    rho_hat: jax.Array
    k_0: jax.Array
    h_0: jax.Array


def production_terms(k, h, p):
    a_k, a_h = p.a_k, p.a_h
    output = (k**a_k) * (h**a_h)
    f_k = a_k * k ** (a_k - 1.0) * h**a_h
    f_h = a_h * k**a_k * h ** (a_h - 1.0)
    return output, f_k, f_h


def human_capital_initial_residual(params, batch):
    a_k, a_h, delta_k, delta_h, k_0 = batch
    h_0 = jnp.exp(params["log_h"])
    c_0 = jnp.exp(params["log_c"])
    output = (k_0**a_k) * (h_0**a_h)
    marginal_k = a_k * (k_0 ** (a_k - 1.0)) * (h_0**a_h)
    marginal_h = a_h * (k_0**a_k) * (h_0 ** (a_h - 1.0))
    return jnp.array(
        [
            marginal_h - delta_h - (marginal_k - delta_k),
            c_0 + delta_h * h_0 + delta_k * k_0 - output,
        ],
        dtype=h_0.dtype,
    )


def human_capital_initial_conditions(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    k_0: float = 1.5,
    iterations: int = 20,
    dtype=jnp.float64,
):
    batch = tuple(
        jnp.asarray(value, dtype=dtype) for value in (a_k, a_h, delta_k, delta_h, k_0)
    )
    k_0_jax = batch[-1]
    c_guess = (
        (k_0_jax ** batch[0]) * (k_0_jax ** batch[1])
        - batch[3] * k_0_jax
        - batch[2] * k_0_jax
    )
    params = {
        "log_h": jnp.log(k_0_jax),
        "log_c": jnp.log(jnp.maximum(c_guess, jnp.asarray(1e-12, dtype=dtype))),
    }
    solver = LevenbergMarquardt(human_capital_initial_residual, init_damping=1e-4)
    result = solver.solve(
        params,
        batch,
        max_steps=iterations,
        max_steps_is_success=False,
    )
    h_0 = jnp.exp(result.x["log_h"])
    c_0 = jnp.exp(result.x["log_c"])
    residual = human_capital_initial_residual(result.x, batch)
    return h_0, c_0, residual, result


@dataclass(frozen=True)
class DataSettings:
    train_T: float = 80.0
    train_points: int = 81
    train_points_list: list[float] | None = None
    test_T: float = 100.0
    test_points: int = 100
    # 249 intervals share no interior point with the training nodes.
    validation_points: int = 250


# Acceptance gates, CLI-reachable so a lower-precision caller can pass its
# own bars.
@dataclass(frozen=True)
class ToleranceSettings:
    gate_train: bool = True
    train_residual: float = 1e-5
    gate_validation: bool = True
    validation_residual: float = 1e-4
