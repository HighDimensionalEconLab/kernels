#!/usr/bin/env bash
# Generate the full replication output (data + figures + paper numbers) end to end.
#   ./generate_all_results.sh            # float64 -> .results/
#   ./generate_all_results.sh --float32  # float32 -> .results_float32/ (skips uno)
set -euo pipefail
cd "$(dirname "$0")"

use_float64=true
output_dir=.results
if [[ "${1:-}" == "--float32" ]]; then
    use_float64=false
    output_dir=.results_float32
fi

# Clusters have no uv; a prebuilt venv (VENV=.venv-gpu ...) carries the stack.
VENV="${VENV:-.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "error: no virtual environment at $VENV (run 'uv sync' or set VENV=...)" >&2
    exit 1
fi
source "$VENV/bin/activate"

start=$SECONDS
echo "==> start: $(date)"

echo "==> data -> $output_dir/ (use_float64=$use_float64)"
python -m replication.neoclassical_growth_kernel --use_float64 "$use_float64"
python -m replication.neoclassical_growth_kernel_robustness --use_float64 "$use_float64"
python -m replication.neoclassical_growth_kernel_sensitivity --use_float64 "$use_float64"
python -m replication.neoclassical_growth_concave_convex_kernel --use_float64 "$use_float64"
python -m replication.neoclassical_growth_concave_convex_nn --use_float64 "$use_float64"
python -m replication.asset_pricing_kernel --use_float64 "$use_float64"
python -m replication.asset_pricing_nn --use_float64 "$use_float64"
python -m replication.optimal_advertising_kernel --use_float64 "$use_float64"
python -m replication.human_capital_kernel --use_float64 "$use_float64"
python -m replication.neoclassical_growth_nn --use_float64 "$use_float64"
python -m replication.neoclassical_growth_nn_sensitivity --use_float64 "$use_float64"
python -m replication.neoclassical_growth_nn_recursive --use_float64 "$use_float64"
python -m replication.neoclassical_growth_kernel_recursive --use_float64 "$use_float64"
python -m replication.stochastic_growth_nn_recursive --use_float64 "$use_float64"
python -m replication.stochastic_growth_kernel_recursive --use_float64 "$use_float64"
python -m replication.open_economy_growth_kernel --use_float64 "$use_float64"
python -m replication.open_economy_growth_nn --use_float64 "$use_float64"
python -m replication.open_economy_growth_nn_recursive --use_float64 "$use_float64"
python -m replication.stochastic_open_economy_growth_nn_recursive --use_float64 "$use_float64"
python -m replication.stochastic_open_economy_growth_kernel_recursive --use_float64 "$use_float64"
python -m replication.multicountry_growth_kernel --use_float64 "$use_float64"
python -m replication.multicountry_growth_kernel_sensitivity --use_float64 "$use_float64"
python -m replication.multicountry_growth_nn --use_float64 "$use_float64"
python -m replication.multicountry_growth_nn_sensitivity --use_float64 "$use_float64"

# The uno/IPOPT baselines and the multicountry scaling ladder are float64-only.
if [[ "$use_float64" == "true" ]]; then
    python -m replication.neoclassical_growth_kernel_uno --output_dir "$output_dir"
    python -m replication.neoclassical_growth_concave_convex_kernel_uno --output_dir "$output_dir"
    python -m replication.asset_pricing_kernel_uno --output_dir "$output_dir"
    python -m replication.open_economy_growth_kernel_uno --output_dir "$output_dir"
    python -m replication.multicountry_growth_scaling --output_dir "$output_dir"
    python -m replication.multicountry_growth_gamma_common --output_dir "$output_dir"
else
    echo "==> skipping uno and the scaling ladder (float64-only)"
fi

echo "==> figures -> $output_dir/"
python -m replication.neoclassical_growth_kernel_figures --output_dir "$output_dir"
python -m replication.neoclassical_growth_concave_convex_kernel_figures --output_dir "$output_dir"
python -m replication.asset_pricing_kernel_figures --output_dir "$output_dir"
python -m replication.optimal_advertising_kernel_figures --output_dir "$output_dir"
python -m replication.human_capital_kernel_figures --output_dir "$output_dir"
python -m replication.open_economy_growth_kernel_figures --output_dir "$output_dir"
python -m replication.multicountry_growth_kernel_figures --output_dir "$output_dir"
if [[ "$use_float64" == "true" ]]; then
    python -m replication.multicountry_growth_scaling_figures --output_dir "$output_dir"
fi
python -m replication.stochastic_growth_kernel_recursive_figures --output_dir "$output_dir"
python -m replication.stochastic_open_economy_growth_kernel_recursive_figures --output_dir "$output_dir"

echo "==> paper numbers -> $output_dir/all_results.json"
python -m replication.collect_json --output_dir "$output_dir" --include_uno "$use_float64"

elapsed=$((SECONDS - start))
echo "==> end: $(date)"
echo "==> done -> $output_dir/  (elapsed ${elapsed}s = $((elapsed / 60))m $((elapsed % 60))s)"
