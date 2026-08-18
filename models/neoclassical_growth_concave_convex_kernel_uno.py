from __future__ import annotations

import jsonargparse

from models.neoclassical_growth import KINKED_CALIBRATION as KINKED
from models.neoclassical_growth_kernel_uno import (
    DataSettings,
    KernelSettings,
    SolverSettings,
    ToleranceSettings,
    neoclassical_growth_kernel_uno,
)


def neoclassical_growth_concave_convex_kernel_uno(
    a: float = KINKED.a,
    delta: float = KINKED.delta,
    rho_hat: float = KINKED.rho_hat,
    A: float = KINKED.A,
    b_1: float = KINKED.b_1,
    b_2: float = KINKED.b_2,
    k_0: float = 1.0,
    sigma_crra: float = KINKED.sigma_crra,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(test_points=41),
    solver_set: SolverSettings | None = None,
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    c_0_guess: float | None = None,
    verbose: bool = False,
):
    return neoclassical_growth_kernel_uno(
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        kernel_set=kernel_set,
        data_set=data_set,
        solver_set=solver_set,
        tolerance_set=tolerance_set,
        c_0_guess=c_0_guess,
        verbose=verbose,
    )


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_kernel_uno)
