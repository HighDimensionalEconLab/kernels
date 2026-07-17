# Does shrinking metric_eps close the small error gap between the NLLS
# selection and UNO's exact min-seminorm KKT point at the baseline n = 41?
# Answer: no -- the matern12 gap is eps-independent (an LM-path selection
# difference on the interpolation manifold, inside the O(dt^2) grid bias),
# and smaller eps degrades the smoother kernels. Findings in README.md.
# Run from the repo root: uv run python -m experiments.eps_sensitivity
import jax

from neoclassical_growth_kernel_nlls import KernelSettings as GrKernel
from neoclassical_growth_kernel_nlls import NLLSGramSettings as GrSettings
from neoclassical_growth_kernel_nlls import neoclassical_growth_kernel_nlls

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# k_rel_error_mean of the UNO/IPOPT solver on the same configurations
# (measured 2026-07-09, see uno_vs_nlls.py).
UNO_REF = {"matern12": 3.39e-04, "matern32": 2.37e-04, "matern52": 5.14e-04}

for kernel in ("matern12", "matern32", "matern52"):
    for eps in (1e-7, 1e-8, 1e-9, 1e-10):
        r, _ = neoclassical_growth_kernel_nlls(
            kernel_set=GrKernel(kernel=kernel),
            lm_set=GrSettings(metric_eps=eps),
        )
        print(
            f"{kernel:9} eps={eps:.0e} ok={str(r['solution_accepted']):>5} "
            f"err={r['k_rel_error_mean']:.3e} (uno {UNO_REF[kernel]:.2e}) "
            f"res={r['max_train_residual']:.1e} steps={r['steps']}"
        )
