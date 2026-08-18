from __future__ import annotations

import jsonargparse

from models.neoclassical_growth import KINKED_CALIBRATION as KINKED
from models.neoclassical_growth_kernel import (
    DataSettings,
    KernelSettings,
    SolverSettings,
    ToleranceSettings,
    neoclassical_growth_kernel,
)


def neoclassical_growth_concave_convex_kernel(
    a: float = KINKED.a,
    delta: float = KINKED.delta,
    rho_hat: float = KINKED.rho_hat,
    A: float = KINKED.A,
    b_1: float = KINKED.b_1,
    b_2: float = KINKED.b_2,
    k_0: float = 1.0,
    sigma_crra: float = KINKED.sigma_crra,
    use_float64: bool = True,
    kernel_set: KernelSettings = KernelSettings(),
    data_set: DataSettings = DataSettings(test_points=41),
    lm_set: SolverSettings = SolverSettings(),
    tolerance_set: ToleranceSettings = ToleranceSettings(),
    c_0_guess: float | None = None,
):
    return neoclassical_growth_kernel(
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        sigma_crra=sigma_crra,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        use_float64=use_float64,
        kernel_set=kernel_set,
        data_set=data_set,
        lm_set=lm_set,
        tolerance_set=tolerance_set,
        c_0_guess=c_0_guess,
    )


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_kernel)
