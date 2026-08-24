# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

Replication code for [Solving Models of Economic Dynamics with Ridgeless Kernel
Regressions](https://arxiv.org/pdf/2406.01898) by Mahdi Ebrahimi Kahou, Jesse
Perla, and Geoff Pleiss.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
uv sync
```

## Replicating the paper

```bash
./generate_all_results.sh
```

runs the paper experiments, renders the figures, and writes
`.results/all_results.json` — the authoritative source for every number in the
paper. Single-model runs and driver details:
[`replication/README.md`](replication/README.md).

## Running one model

Every solver in `models/` is importable and has a
[jsonargparse](https://github.com/omni-us/jsonargparse) CLI; each
`replication/` driver regenerates one model's paper data and figures:

```bash
uv run python -m models.neoclassical_growth_kernel --kernel_set.ell 5.0
uv run python -m replication.neoclassical_growth_kernel --output_dir .results
uv run python -m replication.neoclassical_growth_kernel_figures --output_dir .results
```

## Layout

- [`models/`](models/) — one solver per model/method.
- [`replication/`](replication/README.md) — ensemble and figure drivers and the
  paper-number collector.
- [`notebooks/`](notebooks/README.md) — self-contained annotated notebooks for
  the canonical examples, with prose-free `*_minimal.py` companions.
- [`extra/`](extra/README.md) — nonessential variants; not part of the main
  replication.

## License

[MIT](LICENSE)
