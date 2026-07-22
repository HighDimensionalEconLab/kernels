# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

Replication code for [Solving Models of Economic Dynamics with Ridgeless Kernel
Regressions](https://arxiv.org/pdf/2406.01898) by Mahdi Ebrahimi Kahou, Jesse
Perla, and Geoff Pleiss.

The top level contains the canonical examples. Kernel solves use the released
`nlls-gram` Levenberg--Marquardt implementation; the old UNO implementations and
nonessential comparisons live in [`extra/`](extra/README.md). The specialized
multicountry trade-growth project lives in
[`trade_growth/`](trade_growth/README.md).

## Setup

```bash
uv sync
```

The numerical stack requires Python 3.13, `nlls-gram>=2.4.0`, and
`tinydiffeq>=2.2.0`. Small kernel and NN examples use `nlls-gram`'s automatic
linear-solver choice, as do the open-economy and one-good multicountry models.
The specialized trade-growth project retains explicit solver settings where
comparing dense and iterative paths is useful.

## Main replication

```bash
./generate_all_results.sh
```

This runs only the core paper experiments, renders their figures, and writes
`.results/all_results.json`. Generated JSON is ignored by default; add a
targeted `!path/to/file.json` rule when a JSON file is intentionally tracked.

The core pipeline covers:

- `neoclassical_growth_kernel.py`
- `neoclassical_growth_concave_convex_kernel.py`
- `asset_pricing_kernel.py`
- `asset_pricing_nn.py`
- `optimal_advertising_kernel.py`
- `human_capital_kernel.py`
- `neoclassical_growth_nn_recursive.py`
- `open_economy_growth_kernel.py`
- `open_economy_growth_nn_recursive.py`
- `multicountry_growth_kernel.py`
- `multicountry_growth_nn.py`

Each driver in `replication/` writes Parquet data and summary JSON. Figure
drivers read those saved results, and `replication/collect_json.py` assembles
the authoritative paper-number file. Core NN baselines use 100 seeds; the two
open-economy paths included only for the transition figure use one displayed
seed each. A deterministic kernel specification is solved five times with the
persistent JAX cache enabled; its JSON reports median solve time and its five
raw timings are retained in Parquet.

## Canonical solvers

The main kernel modules are:

- `asset_pricing_kernel.py`: asset-pricing kernel solution and closed-form
  benchmark.
- `asset_pricing_nn.py`: sequence-space MLP trained only on the pricing ODE;
  the no-bubble benchmark is used only for ex-post diagnostics.
- `neoclassical_growth_kernel.py`: smooth or kinked neoclassical growth.
- `neoclassical_growth_concave_convex_kernel.py`: the butterfly-production
  threshold experiment.
- `optimal_advertising_kernel.py`: optimal advertising.
- `human_capital_kernel.py`: the two-capital human-capital model.
- `open_economy_growth_kernel.py`: the one-country open-economy DAE.
- `multicountry_growth_kernel.py`: the reduced one-good multicountry kernel
  formulation, with the country consumption and investment FOCs substituted.

All kernel filenames now refer to the NLLS formulation. Where algebraic DAE
blocks are retained, the kernel metric acts on their RKHS paths as well as the
dynamic paths and selects among the underdetermined solutions. The small
systems use dense kernel matrices, a float64 metric solve, and the library's
default linear solver.

The main NN modules are deliberately limited:

- `neoclassical_growth_nn.py`: direct sequence-space MLP.
- `neoclassical_growth_nn_recursive.py`: fixed-step RK4 state-space investment
  policy used as the recursive NN baseline.
- `open_economy_growth_nn_recursive.py`: the open-economy state-space policy.
- `multicountry_growth_nn.py`: the reduced one-good multicountry sequential NN;
  consumption, investment, and the common rate are calculated from its four
  country path blocks inside the residual evaluation.

Shared activation and transform helpers are in `utilities.py`.

Every solver returns `(results, extras)`, writes nothing itself, and exposes a
`jsonargparse` CLI. For example:

```bash
uv run python neoclassical_growth_kernel.py --kernel_set.ell 5.0
uv run python neoclassical_growth_nn_recursive.py --seed 3 --data_set.train_steps 48
uv run python multicountry_growth_kernel.py --a_0 '[-0.5, 0.5]'
uv run python multicountry_growth_nn.py --a_0 '[-0.5, 0.5]' --seed 0
```

or import directly:

```python
from neoclassical_growth_kernel import neoclassical_growth_kernel

results, extras = neoclassical_growth_kernel()
print(results["k_rel_error_mean"])
```

## Optional examples

[`extra/`](extra/README.md) contains the UNO kernel implementations, the
integrated-derivative MLP, the separate RFF NN, the adaptive recursive kernel
and NN, the unreduced one-good multicountry DAE, method-comparison drivers, and
HPO sweep material. None are part of the main replication command.

[`trade_growth/`](trade_growth/README.md) contains the full multicountry
trade-growth model, kernel and NN solvers, comparative statics, specialized
preconditioners, scaling driver, replication drivers, and tests. The simpler
one-good model and its reduced kernel and NN solvers remain at the top level;
the unreduced DAE variation is under `extra/`.

The design and verification record for the open-economy and multicountry
stepping stones is in [`docs/open_economy_growth.md`](docs/open_economy_growth.md).
The optional adaptive recursive NN is described in
[`extra/docs/neoclassical_growth_nn_recursive_adaptive.md`](extra/docs/neoclassical_growth_nn_recursive_adaptive.md).

## Tests and lint

```bash
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s extra/tests -v
uv run python -m unittest discover -s trade_growth/tests -v
uv run ruff check .
uv run ruff format --check .
```

Enable the committed pre-commit hook once per clone with:

```bash
git config core.hooksPath .githooks
```
