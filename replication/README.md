# Replication package

Drivers and figure scripts that regenerate the paper data, figures, and
tables (JAX/flax + gram-form Levenberg--Marquardt).

**Setup and full instructions: see the top-level [README](../README.md).**

## Files

- `<model>.py` -- ensemble or repeated-solve driver; writes data to the
  output directory.
- `<model>_figures.py` -- figures generated only from saved Parquet results.
- `collect_json.py` -- writes the generated LaTeX tables and collects every
  number embedded in the paper into `all_results.json`; runs last.
- `export_paper_assets.py` -- copies the figures and tables into the paper
  repository.
- `verify_paper_tags.py` -- checks that every `\jsonpath` tag in the paper
  resolves to a leaf of `all_results.json` and matches the printed value.
- `utilities.py` -- shared ensemble, timing, I/O, and plotting helpers.

Every driver enables JAX's persistent compilation cache (`.jax_cache`), so a
second run of the pipeline is fully warm and its timings are compiled-cache
timings. Deterministic solvers run `DETERMINISTIC_TIMING_REPETITIONS` times
(set in `utilities.py`); the timing runs are saved in
`<model>_<experiment>_timing_df.parquet` and the reported solve time is their
median. NN ensembles report medians across accepted seeds: 7 seeds for the
sequence-space baselines, 5 for the recursive baselines, and one for the
recursive growth driver's variant experiments. The multicountry scaling ladder
uses a phase-tagged timing file instead: one first solve followed by two warmed
solves, with the warmed median used in its figure and both phases retained in
`all_results.json`. The stochastic NN drivers write their policies into the
output directory and the stochastic kernel drivers read them from there, so
the script runs each NN driver before its kernel counterpart.

## Generate everything

```bash
./generate_all_results.sh 2>&1 | tee .results/generate.log
```

The final step writes `.results/all_results.json`, the **authoritative source
for every number embedded in the paper**. Selection and rounding to copy/paste
form happen there and only there; never recompute or re-round a paper number
downstream. Add a value to `collect_json.py` when the paper needs it. Then

```bash
uv run python -m replication.export_paper_assets
uv run python -m replication.verify_paper_tags
```

copy the assets into the paper repository and check the paper's tags.

## Run one model

```bash
uv run python -m replication.neoclassical_growth_kernel --output_dir .results
uv run python -m replication.neoclassical_growth_kernel_figures --output_dir .results
```

The output directory and persistent JAX cache are git-ignored.
