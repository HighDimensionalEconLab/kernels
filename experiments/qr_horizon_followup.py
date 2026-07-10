# QR vs Cholesky on the growth model at n = 101 across both horizons, the
# regime where the two dense solvers separate. Also documents the observed
# knife-edge: matern32/rbf at T = 50 under qr once stalled at the warm start
# for all 50 steps inside a long-lived process (train residual 0.137, the
# kernel-independent warm-start residual) while converging reproducibly in
# fresh processes -- likely first-step accept/reject sensitivity at
# init_damping = 1e-8. Findings in README.md.
# Run from the repo root: uv run python -m experiments.qr_horizon_followup
from neoclassical_growth_kernel_nlls import DataSettings as GrData
from neoclassical_growth_kernel_nlls import KernelSettings as GrKernel
from neoclassical_growth_kernel_nlls import NLLSGramSettings as GrSettings
from neoclassical_growth_kernel_nlls import neoclassical_growth_kernel_nlls

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
