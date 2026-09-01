#!/usr/bin/env bash
# Generate the full replication output (data + figures + paper numbers) end to end.
set -euo pipefail
cd "$(dirname "$0")"

output_dir=.results

# Clusters have no uv; a prebuilt venv (VENV=.venv-gpu ...) carries the stack.
VENV="${VENV:-.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "error: no virtual environment at $VENV (run 'uv sync' or set VENV=...)" >&2
    exit 1
fi
source "$VENV/bin/activate"

start=$SECONDS
echo "==> start: $(date)"

echo "==> data -> $output_dir/"
python -m replication.neoclassical_growth_kernel
python -m replication.neoclassical_growth_kernel_robustness
python -m replication.neoclassical_growth_kernel_sensitivity
python -m replication.neoclassical_growth_concave_convex_kernel
python -m replication.neoclassical_growth_concave_convex_nn
python -m replication.neoclassical_growth_concave_convex_equilibrium_selection
python -m replication.asset_pricing_kernel
python -m replication.asset_pricing_nn
python -m replication.optimal_advertising_kernel
python -m replication.human_capital_kernel
python -m replication.neoclassical_growth_nn
python -m replication.neoclassical_growth_nn_sensitivity
python -m replication.neoclassical_growth_nn_recursive
python -m replication.neoclassical_growth_kernel_recursive
python -m replication.stochastic_growth_nn_recursive
python -m replication.stochastic_growth_kernel_recursive
python -m replication.open_economy_growth_kernel
python -m replication.open_economy_growth_nn
python -m replication.open_economy_growth_nn_recursive
python -m replication.stochastic_open_economy_growth_nn_recursive
python -m replication.stochastic_open_economy_growth_kernel_recursive
python -m replication.multicountry_growth_kernel
python -m replication.multicountry_growth_kernel_matern32
python -m replication.multicountry_growth_kernel_sensitivity
python -m replication.multicountry_growth_nn
python -m replication.multicountry_growth_nn_sensitivity

python -m replication.neoclassical_growth_kernel_uno --output_dir "$output_dir"
python -m replication.neoclassical_growth_concave_convex_kernel_uno --output_dir "$output_dir"
python -m replication.asset_pricing_kernel_uno --output_dir "$output_dir"
python -m replication.open_economy_growth_kernel_uno --output_dir "$output_dir"
python -m replication.multicountry_growth_scaling --output_dir "$output_dir"
python -m replication.multicountry_growth_gamma_common --output_dir "$output_dir"

echo "==> figures -> $output_dir/"
python -m replication.neoclassical_growth_kernel_figures --output_dir "$output_dir"
python -m replication.neoclassical_growth_concave_convex_kernel_figures --output_dir "$output_dir"
python -m replication.asset_pricing_kernel_figures --output_dir "$output_dir"
python -m replication.optimal_advertising_kernel_figures --output_dir "$output_dir"
python -m replication.human_capital_kernel_figures --output_dir "$output_dir"
python -m replication.open_economy_growth_kernel_figures --output_dir "$output_dir"
python -m replication.multicountry_growth_kernel_figures --output_dir "$output_dir"
python -m replication.multicountry_growth_scaling_figures --output_dir "$output_dir"
python -m replication.stochastic_growth_kernel_recursive_figures --output_dir "$output_dir"
python -m replication.stochastic_open_economy_growth_kernel_recursive_figures --output_dir "$output_dir"
python -m replication.neoclassical_growth_phase_diagram --output_dir "$output_dir"

echo "==> paper numbers -> $output_dir/all_results.json"
python -m replication.collect_json --output_dir "$output_dir"

elapsed=$((SECONDS - start))
echo "==> end: $(date)"
echo "==> done -> $output_dir/  (elapsed ${elapsed}s = $((elapsed / 60))m $((elapsed % 60))s)"
