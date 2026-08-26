# Optional examples

This directory holds useful variations that are not part of the main paper
replication pipeline.

All UNO implementations live under `models/`; the human-capital and
advertising UNO ensemble drivers remain here as optional comparisons.

## Comparisons and sweeps

`extra/replication/neoclassical_growth_recursive_comparison.py` compares the
recursive (state-space) kernel and NN solvers on the neoclassical growth
model. It is intentionally excluded from `generate_all_results.sh`.

`extra/replication/neoclassical_growth_trend_comparison.py` solves the
neoclassical growth model with a positive productivity trend, the sequential
kernel and NN learning the trend rate and the recursive solvers given it, and
writes a comparison figure.

HPO material is under `extra/sweeps/`. Run optional modules from the
repository root, for example:

```bash
uv run python -m extra.replication.neoclassical_growth_recursive_comparison \
  --output_dir .results-extra
uv run python -m extra.replication.neoclassical_growth_trend_comparison \
  --output_dir .results-extra
uv run python -m extra.replication.human_capital_kernel_uno \
  --output_dir .results-extra
uv run python -m extra.replication.optimal_advertising_kernel_uno \
  --output_dir .results-extra
```