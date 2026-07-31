from __future__ import annotations

import jax
import jsonargparse

from neoclassical_growth_nn import (
    DataSettings,
    GrowthPolicySettings,
    SolverSettings,
    StoppingCriteria,
    neoclassical_growth_nn,
)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


def neoclassical_growth_concave_convex_nn(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    A: float = 0.5,
    b_1: float = 3.0,
    b_2: float = 2.5,
    k_0: float = 1.0,
    sigma_crra: float = 1.0,
    seed: int = 0,
    use_float64: bool = True,
    data_set: DataSettings = DataSettings(test_points=41),
    nn_set: GrowthPolicySettings = GrowthPolicySettings(),
    lm_set: SolverSettings = SolverSettings(),
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
