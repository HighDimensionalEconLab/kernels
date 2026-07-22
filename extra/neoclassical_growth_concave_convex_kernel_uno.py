from __future__ import annotations

import jax
import jsonargparse

from extra.neoclassical_growth_kernel_uno import (
    DataSettings,
    KernelSettings,
    SolverSettings,
    neoclassical_growth_kernel_uno,
)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


def neoclassical_growth_concave_convex_kernel_uno(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    A: float = 0.5,
    b_1: float = 3.0,
    b_2: float = 2.5,
    k_0: float = 1.0,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(test_points=41),
    solver_set: SolverSettings | None = None,
    c_0_guess: float | None = None,
    verbose: bool = False,
):
    return neoclassical_growth_kernel_uno(
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        kernel_set=kernel_set,
        data_set=data_set,
        solver_set=solver_set,
        c_0_guess=c_0_guess,
        verbose=verbose,
    )


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_kernel_uno)
