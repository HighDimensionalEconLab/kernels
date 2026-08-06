from __future__ import annotations

import pathlib
import shutil

import jsonargparse

# Every figure and generated table the paper consumes: (.results filename,
# producing replication module). The copy refuses to run with missing sources,
# and any file present in the paper's figures directory but absent here is
# reported as unmanaged -- the paper-side trace of what feeds what.
MANIFEST = {
    "neoclassical_growth_model_baseline.pdf": "neoclassical_growth_kernel_figures",
    "neoclassical_growth_model_sparse.pdf": "neoclassical_growth_kernel_figures",
    "neoclassical_growth_model_far_steady_state.pdf": (
        "neoclassical_growth_kernel_figures"
    ),
    "neoclassical_growth_model_nu_rho.tex": "collect_json",
    "neoclassical_growth_model_concave_convex_threshold.pdf": (
        "neoclassical_growth_concave_convex_kernel_figures"
    ),
    "asset_pricing_contiguous.pdf": "asset_pricing_kernel_figures",
    "open_economy_growth_transitions.pdf": "open_economy_growth_kernel_figures",
    "neoclassical_human_capital.pdf": "human_capital_kernel_figures",
    "optimal_advertising.pdf": "optimal_advertising_kernel_figures",
    "multicountry_growth_transitions.pdf": "multicountry_growth_kernel_figures",
    "sequential_solver_comparison.tex": "collect_json",
    "neoclassical_growth_solver_comparison.tex": "collect_json",
    "neoclassical_growth_sensitivity.tex": "collect_json",
    "multicountry_growth_sensitivity.tex": "collect_json",
    "stochastic_growth_paths.pdf": "stochastic_growth_kernel_recursive_figures",
    "stochastic_open_economy_paths.pdf": (
        "stochastic_open_economy_growth_kernel_recursive_figures"
    ),
}


def run(
    output_dir: str = ".results",
    paper_dir: str = "../kernels_dynamical_systems/QE/figures",
    dry_run: bool = False,
):
    out = pathlib.Path(output_dir)
    paper = pathlib.Path(paper_dir)
    if not paper.is_dir():
        raise FileNotFoundError(f"paper figures directory not found: {paper}")

    missing = [name for name in MANIFEST if not (out / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing generated assets in {out}/: {missing}; "
            "run generate_all_results.sh (or the listed producers) first"
        )

    width = max(len(name) for name in MANIFEST)
    for name, producer in sorted(MANIFEST.items()):
        print(f"{name:<{width}}  <- replication.{producer}")
        if not dry_run:
            shutil.copy2(out / name, paper / name)

    unmanaged = sorted(
        p.name for p in paper.iterdir() if p.is_file() and p.name not in MANIFEST
    )
    if unmanaged:
        print(f"unmanaged files in {paper}/ (no replication producer): {unmanaged}")
    if not dry_run:
        print(f"copied {len(MANIFEST)} assets -> {paper}/")


if __name__ == "__main__":
    jsonargparse.CLI(run)
