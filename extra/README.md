# Optional examples

This directory holds useful variations that are not part of the main paper
replication pipeline.

The asset-pricing and neoclassical-growth UNO implementations are main examples
with drivers under `replication/`. The human-capital and advertising UNO
implementations remain here as optional comparisons.

## Comparisons and sweeps

`extra/replication/neoclassical_growth_recursive_comparison.py` compares the
recursive (state-space) kernel and NN solvers on the neoclassical growth
model. It is intentionally excluded from `generate_all_results.sh`.

HPO material is under `extra/sweeps/`. Run optional modules from the
repository root, for example:

```bash
uv run python -m extra.replication.neoclassical_growth_recursive_comparison \
  --output_dir .results-extra
uv run python -m extra.replication.human_capital_kernel_uno \
  --output_dir .results-extra
uv run python -m extra.replication.optimal_advertising_kernel_uno \
  --output_dir .results-extra
```