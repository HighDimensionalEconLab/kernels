# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

Replication package for
[Solving Models of Economic Dynamics with Ridgeless Kernel Regressions](https://arxiv.org/pdf/2406.01898)
by Mahdi Ebrahimi Kahou, Jesse Perla, and Geoff Pleiss.

## Setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) for all
Python environment management:

```bash
uv sync
```

The RKHS collocation problems are solved through direct JAX callbacks to UNO
via the `unopy` wheel; the neural-network solvers train pure-JAX MLPs with the
`nlls_gram` Levenberg-Marquardt solver. No optimization DSL, external solver
executable, or conda environment is needed.

## Full replication

```bash
./generate_all_results.sh
```

runs every experiment, renders every figure, and writes the authoritative
paper-number file, all into `.results/` (nothing is written anywhere else):

1. **Data** — one driver per model (`replication/<name>.py`) saves
   `\<prefix>_\<experiment>_results.parquet`, per-solve trajectory frames
   (`*_test_df.parquet`, ...), a `*_hyperparameters.json`, and a JSON of
   cross-seed summary statistics per experiment.
2. **Figures** — `replication/<name>_figures.py` scripts read only those saved
   files and write the paper PDFs.
3. **Numbers** — `replication/collect_json.py` assembles
   `.results/all_results.json` (every number cited in the paper, rounded to
   copy/paste form exactly once, strict JSON) and the robustness table
   `.results/neoclassical_growth_model_nu_rho.tex`.

Paper artifacts produced in `.results/`:

| Artifact | Source |
| --- | --- |
| `neoclassical_growth_model_baseline.pdf` | `neoclassical_growth_kernel` |
| `neoclassical_growth_model_sparse.pdf` | `neoclassical_growth_kernel` |
| `neoclassical_growth_model_far_steady_state.pdf` | `neoclassical_growth_kernel` |
| `neoclassical_growth_model_nu_rho.tex` | `collect_json` (robustness table incl. Gaussian/RBF rows) |
| `neoclassical_growth_model_concave_convex_threshold.pdf` | `neoclassical_growth_concave_convex_kernel` (70 initial conditions) |
| `asset_pricing_contiguous.pdf` | `asset_pricing_kernel` (corrected closed-form benchmark, `x_0 = 1`) |
| `optimal_advertising.pdf` | `optimal_advertising_kernel` |
| `neoclassical_human_capital.pdf` | `human_capital_kernel` |
| `kernel_vs_nn_errors.pdf`, `kernel_vs_nn_basin.pdf` | `kernel_vs_nn` |
| `neoclassical_growth_recursive_policy.pdf`, `neoclassical_growth_recursive_butterfly.pdf` | `neoclassical_growth_nn_recursive` |
| `all_results.json` | `collect_json` |

Note: the paper's `phase_diagram_mu_k.pdf` (referenced only by an appendix that
is not `\input`) is a hand-made illustration and is not produced by this
package.

## Solvers

Flat top-level modules; each returns `(results, extras)` where `results` is a
flat dict of scalars (always including `solution_accepted` and
`rejection_reason`) and `extras` holds `model`, trajectory DataFrames, and the
`hyperparameters` used. Solvers write nothing to disk. Kernel solvers take
`kernel_set` (`nu`, `ell`, `sigma`; `nu = inf` selects the Gaussian/RBF kernel
with a closed-form erf integrated kernel), `data_set`, and `solver_set`
dataclasses.

- `asset_pricing_kernel.py` — linear asset-pricing QP vs the closed-form
  no-bubble fundamental (`asset_pricing_benchmark.py`).
- `neoclassical_growth_kernel.py` — baseline and optional kinked-production
  growth NLP vs the `solve_bvp` benchmark (`neoclassical_growth_benchmark.py`).
- `neoclassical_growth_concave_convex_kernel.py` — kinked-production wrapper
  used for the basin-of-attraction threshold experiment.
- `optimal_advertising_kernel.py` — optimal-advertising NLP.
- `human_capital_kernel.py` — two-capital human-capital NLP
  (`lambda_p`-weighted policy norms).
- `neoclassical_growth_nn.py` — sequential NN: MLP maps `t -> (k, c)`.
- `neoclassical_growth_nn_integrated.py` — sequential NN: the net approximates
  the time derivatives, integrated to paths (MLP or random Fourier features via
  `random_fourier_features.py`).
- `neoclassical_growth_nn_recursive_i.py` — recursive (state-space) NN: the
  net approximates gross investment `i(k)` with a linear head anchored at the
  replacement level (reversible for now — an `i >= 0` irreversibility
  transform is the natural extension); drift and consumption are implied by
  the resource constraint. Collocation states come from a
  differentiated-through free-stepping adaptive Tsit5 rollout of the learned
  policy from `k_0`: the solver's accepted states ARE the collocation points
  (no grid; steps are clipped only at the horizon). No transversality penalty
  enters training: the flat-start parameterization selects the saddle path,
  and the exact log-transversality certificate (computed post hoc on a short
  auxiliary simulation past the training window) gates early stopping and
  acceptance.
- `integrators.py` — the differentiable rollout integrators (free-stepping
  adaptive Tsit5 for training; fixed-grid RK4 for the test/settle evaluation
  paths); forward- and reverse-mode differentiable through the states, array
  states (scalar or vector).

Every solver exposes a `jsonargparse` CLI:

```bash
uv run python neoclassical_growth_kernel.py --kernel_set.ell 5.0
uv run python neoclassical_growth_kernel.py --data_set.train_points 21
uv run python asset_pricing_kernel.py --x_0 1.0
uv run python neoclassical_growth_nn_recursive_i.py --seed 3 --data_set.rtol 1e-8
```

or import directly:

```python
from neoclassical_growth_kernel import neoclassical_growth_kernel

results, extras = neoclassical_growth_kernel()
print(results["k_rel_error_mean"])
```

## Tests

```bash
uv run python -m unittest discover -s tests
```

## Lint

`ruff` runs from the committed pre-commit hook; enable it once per clone with

```bash
git config core.hooksPath .githooks
```
