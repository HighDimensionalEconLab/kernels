# QR vs Cholesky on the growth model at n = 101 across both horizons, the
# regime where the two dense solvers separate. The stall once observed here
# (matern32/rbf at T = 50 under qr frozen at the warm start for all 50
# steps in one long-lived process) was traced to jaxlib heap corruption on
# macOS arm64 (jax 0.10.2) -- the silent-corruption face of a bug whose loud
# face is SIGSEGV/SIGTRAP inside libjax_common.dylib. Not an nlls-gram
# issue; see README.md for the root-cause analysis.
# Run from the repo root: uv run python -m experiments.qr_horizon_followup
import jax

from neoclassical_growth_kernel_nlls import DataSettings as GrData
from neoclassical_growth_kernel_nlls import KernelSettings as GrKernel
from neoclassical_growth_kernel_nlls import NLLSGramSettings as GrSettings
from neoclassical_growth_kernel_nlls import neoclassical_growth_kernel_nlls

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

print(
    f"{'kernel':9} {'T':>4} {'n':>4} {'solver':9} {'ok':>3} {'reason':14} "
    f"{'steps':>5} {'max_train_res':>13} {'k_err_mean':>10}"
)
for kernel, ell in (("matern32", 10.0), ("matern52", 10.0), ("rbf", 2.0)):
    for T in (40.0, 50.0):
        for solver in ("cholesky", "qr"):
            r, _ = neoclassical_growth_kernel_nlls(
                kernel_set=GrKernel(kernel=kernel, ell=ell),
                data_set=GrData(train_T=T, train_points=101),
                lm_set=GrSettings(linear_solver=solver),
            )
            print(
                f"{kernel:9} {T:>4} {101:>4} {solver:9} "
                f"{str(r['solution_accepted']):>3} {r['rejection_reason']:14} "
                f"{r['steps']:>5} {r['max_train_residual']:>13.3e} "
                f"{r['k_rel_error_mean']:>10.3e}"
            )
