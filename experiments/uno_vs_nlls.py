# Run every replication experiment configuration that has an NLLS port
# (growth: baseline, sparse, far_steady_state, the nu/ell robustness rows,
# the rbf referee rows; asset pricing: baseline) through both the UNO/IPOPT
# solver and the nlls-gram solver, side by side. A results snapshot lives in
# uno_vs_nlls_results.json; the verdict is in README.md.
# Run from the repo root:
#   uv run python -m experiments.uno_vs_nlls [output.json]
import json
import math
import sys
import time

import asset_pricing_kernel as ap_uno
import asset_pricing_kernel_nlls as ap_nlls
import neoclassical_growth_kernel as gr_uno
import neoclassical_growth_kernel_nlls as gr_nlls

NU_TO_KERNEL = {0.5: "matern12", 1.5: "matern32", 2.5: "matern52", math.inf: "rbf"}

SPARSE_POINTS = [0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 38.0, 40.0]

# name, nu, ell, data kwargs -- mirrors replication/neoclassical_growth_kernel.py
GROWTH_EXPERIMENTS = [
    ("baseline", 0.5, 10.0, {}),
    ("sparse", 0.5, 10.0, {"train_points_list": SPARSE_POINTS}),
    (
        "far_steady_state",
        0.5,
        10.0,
        {"train_T": 10.0, "train_points": 11, "test_T": 15.0},
    ),
    ("nu15_ell10", 1.5, 10.0, {}),
    ("nu25_ell10", 2.5, 10.0, {}),
    ("nu05_ell2", 0.5, 2.0, {}),
    ("nu05_ell20", 0.5, 20.0, {}),
    ("rbf_ell10", math.inf, 10.0, {}),
    ("rbf_ell2", math.inf, 2.0, {}),
    ("rbf_ell20", math.inf, 20.0, {}),
]

rows = []

for name, nu, ell, data_kwargs in GROWTH_EXPERIMENTS:
    uno_start = time.perf_counter()
    try:
        uno, _ = gr_uno.neoclassical_growth_kernel(
            kernel_set=gr_uno.KernelSettings(nu=nu, ell=ell),
            data_set=gr_uno.DataSettings(**data_kwargs),
            verbose=False,
        )
    except Exception as e:
        uno = {"solution_accepted": False, "rejection_reason": f"exception: {e}"}
    uno_wall = time.perf_counter() - uno_start

    nlls_start = time.perf_counter()
    try:
        nlls, _ = gr_nlls.neoclassical_growth_kernel_nlls(
            kernel_set=gr_nlls.KernelSettings(kernel=NU_TO_KERNEL[nu], ell=ell),
            data_set=gr_nlls.DataSettings(**data_kwargs),
        )
    except Exception as e:
        nlls = {"solution_accepted": False, "rejection_reason": f"exception: {e}"}
    nlls_wall = time.perf_counter() - nlls_start

    rows.append(("growth", name, uno, nlls, uno_wall, nlls_wall))

uno_start = time.perf_counter()
uno, _ = ap_uno.asset_pricing_kernel()
uno_wall = time.perf_counter() - uno_start
nlls_start = time.perf_counter()
nlls, _ = ap_nlls.asset_pricing_kernel_nlls()
nlls_wall = time.perf_counter() - nlls_start
rows.append(("asset", "baseline", uno, nlls, uno_wall, nlls_wall))

KEYS = [
    "solution_accepted",
    "rejection_reason",
    "linear_solver_resolved",
    "residuals_method_resolved",
    "max_train_residual",
    "p_rel_error_mean",
    "p_rel_error_max",
    "k_rel_error_mean",
    "k_rel_error_max",
    "c_rel_error_mean",
    "c_rel_error_max",
    "solve_time",
    "steps",
]
if len(sys.argv) > 1:
    out = [
        {
            "model": model,
            "experiment": name,
            "uno": {k: uno.get(k) for k in KEYS if k in uno},
            "nlls": {k: nlls.get(k) for k in KEYS if k in nlls},
            "uno_wall": uno_wall,
            "nlls_wall": nlls_wall,
        }
        for model, name, uno, nlls, uno_wall, nlls_wall in rows
    ]
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=2)

err_key = {"growth": "k_rel_error_mean", "asset": "p_rel_error_mean"}
print(
    f"\n{'experiment':18} {'src':5} {'ok':>5} {'err_mean':>10} {'max_res':>10} "
    f"{'solve_s':>9} {'wall_s':>7}"
)
for model, name, uno, nlls, uno_wall, nlls_wall in rows:
    for src, r, wall in (("uno", uno, uno_wall), ("nlls", nlls, nlls_wall)):
        err = r.get(err_key[model])
        res = r.get("max_train_residual")
        st = r.get("solve_time")
        print(
            f"{name:18} {src:5} {str(r.get('solution_accepted')):>5} "
            f"{(f'{err:.2e}' if err is not None else '--'):>10} "
            f"{(f'{res:.2e}' if res is not None else '--'):>10} "
            f"{(f'{st:.4f}' if st is not None else '--'):>9} {wall:>7.2f}"
        )
