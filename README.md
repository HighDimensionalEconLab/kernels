# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

Replication package for
[Solving Models of Economic Dynamics with Ridgeless Kernel Regressions](https://arxiv.org/pdf/2406.01898)
by Mahdi Ebrahimi Kahou, Jesse Perla, and Geoff Pleiss.

The design and verification record for the deterministic international-macro example is
in [docs/open_economy_growth.md](docs/open_economy_growth.md). Stage 1 solves a
continuous-time small open economy with capital installation costs and initial net
foreign assets. Stage 3 implements an N-country closed-world stepping stone, evaluated
at two and three countries, with an endogenous world rate and explicit country asset and
marginal-wealth blocks. Stage 5 joins the same generic dynamic blocks to the full
three-sector trade system and verifies the complete DAE at two and three countries. The
kernel and
sequential state-space NN use only finite-time model equations; their respective RKHS
and neural-policy biases select the otherwise free jump variables, while no-Ponzi and
transversality conditions are evaluated only ex post. Independent boundary-value
solutions provide conventional economic benchmarks.

## Setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) for all
Python environment management:

```bash
uv sync
```

The published RKHS baselines remain available through direct JAX callbacks to
UNO via the `unopy` wheel. Parallel `*_nlls.py` variants solve every kernel
model with the released `nlls-gram` Levenberg-Marquardt package. The neural
network solvers use the same pure-JAX optimizer. No optimization DSL, external
solver executable, or conda environment is needed.

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
- `asset_pricing_kernel_nlls.py`, `neoclassical_growth_kernel_nlls.py`,
  `neoclassical_growth_concave_convex_kernel_nlls.py`,
  `optimal_advertising_kernel_nlls.py`, and `human_capital_kernel_nlls.py` —
  metric-aware NLLS variants of every kernel model; the original unopy files
  remain the paper-replication baselines.
- `neoclassical_growth_nn.py` — sequential NN: MLP maps `t -> (k, c)`.
- `neoclassical_growth_nn_integrated.py` — sequential NN: the net approximates
  the time derivatives, integrated to paths (MLP or random Fourier features via
  `random_fourier_features.py`).
- `neoclassical_growth_nn_recursive_i.py` — recursive (state-space) NN: the
  net approximates positive gross investment `i(k)` from the normalized input
  `log(k/k_0)`. Drift and consumption are implied by the resource constraint.
  Training integrates log capital and samples the differentiated-through
  rollout on a fixed 65-point collocation grid. The default 32-by-3 GELU policy
  has 2,209 parameters; wider networks reached the same measured accuracy at
  higher cost. No transversality penalty or callback enters training: the exact
  log-transversality certificate is evaluated once after the solve and gates
  final acceptance.
- `open_economy_growth_nn_recursive.py` — recursive open-economy NN: a shared
  state-space policy maps `log(k/k_0)` to positive investment expenditure and
  the signed current account. It rolls forward capital and NFA, derives Tobin's
  q and consumption algebraically, and enforces only the q Euler equation,
  flat-consumption Euler equation, and the consumption domain. A direct
  consumption head is intentionally avoided because it has no useful bias
  toward the annuity root. Solvency, TVCs, and the BVP comparison are ex-post
  diagnostics and never acceptance gates.
- `multicountry_growth_kernel_nlls.py` — N-country closed-world kernel solve:
  every country retains integrated paths for capital, NFA, Tobin q, and
  marginal wealth. The world rate is an explicit algebraic value at every
  collocation point. All country budget equations remain in the DAE; world
  resource and asset clearing provide the nonredundant global algebraic closure.
  The metric selects the initial q and marginal-wealth values without a relative
  two-country parameterization. Its `(4N+1)n+1` residuals include the initial
  DAE consistency row `G_dot(0)=0` and leave `2N-1` directions to the minimum-RKHS
  metric. They contain no terminal, TVC, no-Ponzi, debt target, or benchmark
  equation. The default two-country run is checked against an independent
  full-Jacobian unstable-projection BVP, and a three-country acceptance run
  verifies that the solved system is generic.
- `multicountry_trade_growth.py` — separate N-country trade-growth mechanics:
  an explicit `8N`-equation static block retains wages, rentals, three sector
  prices, and three sector revenues for every country. Eaton-Kortum shares are
  evaluated as an importer-by-exporter matrix, while both factor markets and
  every country's intermediate-market equation remain explicit. The module
  also supplies the continuous-time capital, NFA, Tobin-q, and marginal-wealth
  shell used by the joint solver; it does not modify the useful one-good
  implementation.
- `multicountry_trade_growth_kernel_nlls.py` — complete generic N-country
  trade-growth DAE, exercised at two and three countries. It retains all `4N`
  dynamic RKHS paths, all `8N` static equations, every country budget, and a
  symmetric consumption-basket numeraire. The `2N` initial Tobin-q and
  marginal-wealth values are selected by the metric. Asset clearing, no-Ponzi,
  and transversality conditions are evaluated only ex post. The replication
  driver uses a 120-year CPU transition window and includes zero, reversed, and
  doubled initial-NFA counterfactuals; its companion figure script plots the
  selected history dependence, discounted NFA, and country-symmetry diagnostics.
- The differentiable rollout integrators (fixed RK4 by measured default, with
  Tsit5, Euler, and Rodas5P available; fixed RK4 for test/settle paths) come from
  [tinydiffeq](https://github.com/HighDimensionalEconLab/tinydiffeq),
  and are forward- and reverse-mode differentiable through the states.

Every solver exposes a `jsonargparse` CLI:

```bash
uv run python neoclassical_growth_kernel.py --kernel_set.ell 5.0
uv run python neoclassical_growth_kernel.py --data_set.train_points 21
uv run python asset_pricing_kernel.py --x_0 1.0
uv run python neoclassical_growth_nn_recursive_i.py --seed 3 --data_set.rtol 1e-8
uv run python open_economy_growth_nn_recursive.py --seed 0 --a_0 -0.5
uv run python multicountry_growth_kernel_nlls.py --a_0 '[-0.5, 0.5]'
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
