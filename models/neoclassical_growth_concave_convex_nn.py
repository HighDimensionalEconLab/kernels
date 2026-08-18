from __future__ import annotations

import jsonargparse

from models.neoclassical_growth import KINKED_CALIBRATION as KINKED
from models.neoclassical_growth_nn import (
    DataSettings,
    GrowthPolicySettings,
    StoppingCriteria,
    neoclassical_growth_nn,
)
from models.utilities import OptimizerSettings


def neoclassical_growth_concave_convex_nn(
    a: float = KINKED.a,
    delta: float = KINKED.delta,
    rho_hat: float = KINKED.rho_hat,
    A: float = KINKED.A,
    b_1: float = KINKED.b_1,
    b_2: float = KINKED.b_2,
    k_0: float = 1.0,
    sigma_crra: float = KINKED.sigma_crra,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(test_points=41),
    nn_set: GrowthPolicySettings = GrowthPolicySettings(),
    lm_set: OptimizerSettings = OptimizerSettings(
        init_damping=1e-5,
        damping_decrease=0.5,
        damping_increase=2.0,
    ),
    stop_set: StoppingCriteria = StoppingCriteria(),
    verbose: bool = True,
):
    return neoclassical_growth_nn(
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        sigma_crra=sigma_crra,
        seed=seed,
        use_float64=use_float64,
        data_set=data_set,
        nn_set=nn_set,
        lm_set=lm_set,
        stop_set=stop_set,
        verbose=verbose,
    )


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_nn)
