# Cholesky vs QR on every dense-resolving NLLS configuration the experiments
# and tests exercise (both models x kernels x n). Findings in README.md; this
# sweep motivated the growth model's auto-qr boundary (DENSE_QR_MIN_POINTS).
# Run from the repo root: uv run python -m experiments.qr_sweep
import jax

from asset_pricing_kernel_nlls import DataSettings as APData
from asset_pricing_kernel_nlls import KernelSettings as APKernel
from asset_pricing_kernel_nlls import NLLSGramSettings as APSettings
from asset_pricing_kernel_nlls import asset_pricing_kernel_nlls
from neoclassical_growth_kernel_nlls import DataSettings as GrData
from neoclassical_growth_kernel_nlls import KernelSettings as GrKernel
from neoclassical_growth_kernel_nlls import NLLSGramSettings as GrSettings
from neoclassical_growth_kernel_nlls import neoclassical_growth_kernel_nlls

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

AP_CASES = [
    ("matern12", 10.0, 40.0, 41),
    ("matern32", 10.0, 40.0, 41),
    ("matern52", 10.0, 40.0, 41),
    ("rbf", 2.0, 40.0, 41),
    ("matern12", 10.0, 50.0, 101),
    ("matern32", 10.0, 50.0, 101),
    ("matern52", 10.0, 50.0, 101),
    ("rbf", 2.0, 50.0, 101),
    ("matern52", 10.0, 50.0, 501),
    ("rbf", 2.0, 50.0, 501),
    ("matern32", 10.0, 50.0, 1001),
]

GR_CASES = [
    ("matern12", 10.0, 40.0, 41),
    ("matern32", 10.0, 40.0, 41),
    ("matern52", 10.0, 40.0, 41),
    ("rbf", 2.0, 40.0, 41),
    ("matern12", 10.0, 50.0, 101),
    ("matern32", 10.0, 50.0, 101),
    ("matern52", 10.0, 50.0, 101),
    ("rbf", 2.0, 50.0, 101),
    ("matern32", 10.0, 50.0, 501),
    ("matern52", 10.0, 50.0, 501),
]

print(
    f"{'model':6} {'kernel':9} {'n':>5} {'solver':9} {'ok':>3} {'reason':14} "
    f"{'steps':>5} {'max_train_res':>13} {'err_mean':>10} {'solve_s':>8}"
)

for kernel, ell, T, n in AP_CASES:
    for solver in ("cholesky", "qr"):
        r, _ = asset_pricing_kernel_nlls(
            kernel_set=APKernel(kernel=kernel, ell=ell),
            data_set=APData(train_T=T, train_points=n),
            lm_set=APSettings(linear_solver=solver),
        )
        print(
            f"{'asset':6} {kernel:9} {n:>5} {solver:9} "
            f"{str(r['solution_accepted']):>3} {r['rejection_reason']:14} "
            f"{r['steps']:>5} {r['max_train_residual']:>13.3e} "
            f"{r['p_rel_error_mean']:>10.3e} {r['solve_time']:>8.4f}"
        )

for kernel, ell, T, n in GR_CASES:
    for solver in ("cholesky", "qr"):
        r, _ = neoclassical_growth_kernel_nlls(
            kernel_set=GrKernel(kernel=kernel, ell=ell),
            data_set=GrData(train_T=T, train_points=n),
            lm_set=GrSettings(linear_solver=solver),
        )
        print(
            f"{'growth':6} {kernel:9} {n:>5} {solver:9} "
            f"{str(r['solution_accepted']):>3} {r['rejection_reason']:14} "
            f"{r['steps']:>5} {r['max_train_residual']:>13.3e} "
            f"{r['k_rel_error_mean']:>10.3e} {r['solve_time']:>8.4f}"
        )
