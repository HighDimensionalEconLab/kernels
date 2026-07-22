# Replication package

Drivers and figure scripts that regenerate the core paper data and figures
(JAX/flax + gram-form Levenberg--Marquardt).

**Setup and full instructions: see the top-level [README](../README.md).**

## Files

- `<model>.py` -- cached ensemble or repeated-solve driver; writes data to the
  output directory.
- `<model>_figures.py` -- figures generated only from saved Parquet results.
- `collect_json.py` -- collects every number embedded in the paper into
  `all_results.json`; runs last.
- `utilities.py` -- shared ensemble, repeated timing, I/O, and plotting helpers.

Deterministic solvers run five times with JAX's persistent compilation cache
enabled. Their reported solve times are medians, and the individual timing runs
are saved in `<model>_<experiment>_timing_df.parquet`. Core stochastic NN
baselines use 100 seeds and report medians across accepted seeds. The two
open-economy asset scenarios used only in the transition figure retain one
displayed seed each.

## Generate everything

```bash
./generate_all_results.sh
```

The final step writes `.results/all_results.json`, the **authoritative source
for every number embedded in the paper**. Selection and rounding to copy/paste
form happen there and only there; never recompute or re-round a paper number
downstream. Add a value to `collect_json.py` when the paper needs it.

## Run one model

```bash
uv run python -m replication.neoclassical_growth_kernel --output_dir .results
uv run python -m replication.neoclassical_growth_kernel_figures --output_dir .results
```

The output directory and persistent JAX cache are git-ignored.
