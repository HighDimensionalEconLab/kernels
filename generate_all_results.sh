#!/usr/bin/env bash
# Generate the full replication output (data + figures + paper numbers) end to end.
#   ./generate_all_results.sh            # -> .results/
set -euo pipefail
cd "$(dirname "$0")"

output_dir=.results

start=$SECONDS
echo "==> start: $(date)"

echo "==> data -> $output_dir/"
uv run python -m replication.neoclassical_growth_kernel --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_kernel_uno --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_concave_convex_kernel --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_concave_convex_kernel_uno --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_concave_convex_nn --output_dir "$output_dir"
uv run python -m replication.asset_pricing_kernel --output_dir "$output_dir"
uv run python -m replication.asset_pricing_kernel_uno --output_dir "$output_dir"
uv run python -m replication.asset_pricing_nn --output_dir "$output_dir"
uv run python -m replication.optimal_advertising_kernel --output_dir "$output_dir"
uv run python -m replication.human_capital_kernel --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_nn --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_nn_recursive --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_kernel_recursive --output_dir "$output_dir"
uv run python -m replication.stochastic_growth_nn_recursive --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_kernel --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_kernel_uno --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_nn --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_nn_recursive --output_dir "$output_dir"
uv run python -m replication.multicountry_growth_kernel --output_dir "$output_dir"
uv run python -m replication.multicountry_growth_nn --output_dir "$output_dir"

echo "==> figures -> $output_dir/"
uv run python -m replication.neoclassical_growth_kernel_figures --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_concave_convex_kernel_figures --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_concave_convex_nn_figures --output_dir "$output_dir"
uv run python -m replication.asset_pricing_kernel_figures --output_dir "$output_dir"
uv run python -m replication.asset_pricing_nn_figures --output_dir "$output_dir"
uv run python -m replication.optimal_advertising_kernel_figures --output_dir "$output_dir"
uv run python -m replication.human_capital_kernel_figures --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_nn_figures --output_dir "$output_dir"
uv run python -m replication.neoclassical_growth_nn_recursive_figures --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_kernel_figures --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_nn_figures --output_dir "$output_dir"
uv run python -m replication.open_economy_growth_nn_recursive_figures --output_dir "$output_dir"
uv run python -m replication.multicountry_growth_kernel_figures --output_dir "$output_dir"

echo "==> paper numbers -> $output_dir/all_results.json"
uv run python -m replication.collect_json --output_dir "$output_dir"

elapsed=$((SECONDS - start))
echo "==> end: $(date)"
echo "==> done -> $output_dir/  (elapsed ${elapsed}s = $((elapsed / 60))m $((elapsed % 60))s)"
