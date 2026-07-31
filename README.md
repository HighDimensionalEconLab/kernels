# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

Replication code for [Solving Models of Economic Dynamics with Ridgeless Kernel
Regressions](https://arxiv.org/pdf/2406.01898) by Mahdi Ebrahimi Kahou, Jesse
Perla, and Geoff Pleiss.

The top level contains the canonical examples. Kernel solves use the released
`nlls-gram` Levenberg--Marquardt implementation. Asset pricing and growth also
include constrained UNO comparisons; nonessential variations live in
[`extra/`](extra/README.md). The specialized
multicountry trade-growth project lives in
[`extra/trade_growth/`](extra/trade_growth/README.md).

## Setup

```bash
uv sync
```

The numerical stack requires Python 3.13, `nlls-gram>=2.6.0`, and
`tinydiffeq>=2.2.0`. Small kernel and NN examples use `nlls-gram`'s automatic
linear-solver choice. The reduced one-good multicountry kernel also exposes a
matrix-free preconditioned-CG path for large country counts; the specialized
trade-growth project retains its own explicit solver settings.

## Main replication

```bash
./generate_all_results.sh
```

This runs only the core paper experiments, renders their figures, and writes
`.results/all_results.json`. Generated JSON is ignored by default; add a
targeted `!path/to/file.json` rule when a JSON file is intentionally tracked.

The core pipeline covers:

- `neoclassical_growth_kernel.py`
- `neoclassical_growth_kernel_uno.py`
- `neoclassical_growth_concave_convex_kernel.py`
- `neoclassical_growth_concave_convex_kernel_uno.py`
- `neoclassical_growth_concave_convex_nn.py`
- `neoclassical_growth_concave_convex_nn_recursive.py`
- `asset_pricing_kernel.py`
- `asset_pricing_kernel_uno.py`
- `asset_pricing_nn.py`
- `optimal_advertising_kernel.py`
- `human_capital_kernel.py`
- `neoclassical_growth_nn.py`
- `neoclassical_growth_nn_recursive.py`
- `neoclassical_growth_kernel_recursive.py`
- `open_economy_growth_kernel.py`
- `open_economy_growth_kernel_uno.py`
- `open_economy_growth_nn.py`
- `open_economy_growth_nn_recursive.py`
- `multicountry_growth_kernel.py`
- `multicountry_growth_nn.py`

Each driver in `replication/` writes Parquet data and summary JSON. Figure
drivers read those saved results, and `replication/collect_json.py` assembles
the authoritative paper-number file. Core NN baselines use 100 seeds, except the open-economy sequence-space NN
(21 seeds, at roughly two seconds per solve) and
the multicountry NN replication, which uses five seeds and also pins a
10-country width-32/depth-4 experiment without changing the solver defaults.
Every NN driver allows each seed up to five multi-start restarts
(`stop_set.num_attempts`), so `frac_accepted` measures acceptance within that
restart budget rather than first-try robustness.
The two open-economy paths included only
for the transition figure use one displayed seed each. A deterministic kernel
specification is solved five times with the persistent JAX cache enabled; its
JSON reports median solve time and its five raw timings are retained in Parquet.

## Canonical solvers

The main kernel modules are:

- `asset_pricing_kernel.py`: asset-pricing kernel solution and closed-form
  benchmark.
- `asset_pricing_kernel_uno.py`: constrained asset-pricing kernel comparison.
- `asset_pricing_nn.py`: sequence-space MLP trained only on the pricing ODE;
  the no-bubble benchmark is used only for ex-post diagnostics.
- `neoclassical_growth_kernel.py`: smooth or kinked neoclassical growth.
- `neoclassical_growth_kernel_uno.py`: constrained growth kernel comparison.
- `neoclassical_growth_concave_convex_kernel.py`: the butterfly-production
  threshold experiment. `neoclassical_growth_concave_convex_nn.py` and
  `neoclassical_growth_concave_convex_nn_recursive.py` run the same 70-point
  `k_0` sweep with the two NN formulations. Basin correctness is scored
  independently of solver acceptance in all four: a sequence-space solution
  (kernel, UNO, sequential NN) may still be in motion at `test_T` and is
  classified by the side it ends on plus the direction of its tail, while the
  recursive policy is settled to `settle_T` and classified by the steady-state
  ball it lands in. Neither test feeds back into the solve.
- `optimal_advertising_kernel.py`: optimal advertising.
- `human_capital_kernel.py`: the two-capital human-capital model.
- `open_economy_growth_kernel.py`: the one-country open-economy DAE.
- `open_economy_growth_kernel_uno.py`: the constrained comparison using the
  same open-economy DAE paths, equations, and collocation grid with the
  function-only RKHS norm as its objective.
- `multicountry_growth_kernel.py`: the reduced one-good multicountry kernel
  formulation, with the country consumption and investment FOCs substituted.

Unqualified kernel filenames refer to the NLLS formulation; `_uno` identifies
the constrained comparison. Every driver except `extra/trade_growth` uses
`nlls_gram.RidgeLevenbergMarquardt`: the objective
`||r||^2 + ridge * q(theta)` with the block seminorm
`q = sum_j alpha_j' K alpha_j` supplied as a positive-definite
`nlls_gram.RepeatedFactorMetric` on the upper Cholesky factor of `K`
(structural scalars live in the unpenalized free block, no epsilon shift
anywhere) -- the minimizer is the min-RKHS-norm interpolant up to an
O(ridge) bias, and where algebraic DAE blocks are retained the metric acts
on their RKHS paths as well as the dynamic paths (the reduced multicountry
model's common algebraic interest rate is an RKHS path with the same kernel
metric). The solver runs entirely in the whitened variable, where the
normal matrix keeps a clean spectral floor at the ridge, so every driver
defaults to the dense `Cholesky()` path: the simple ODE drivers
(`asset_pricing_kernel`, `neoclassical_growth_kernel` and its concave-convex
wrapper, `optimal_advertising_kernel`, `human_capital_kernel`) solve at a
fixed small `ridge`, while the harder DAE drivers
(`open_economy_growth_kernel`, `multicountry_growth_kernel`) add
`nlls_gram.ridge_continuation` annealing the ridge to a small floor, with
`QR()` retained as the extreme-conditioning fallback. `multicountry`'s
`lm_set.linear_solver="cg"` switches to matrix-free preconditioned CG on
the whitened normal operator: `nlls_gram.BlockEigenPreconditioner` reads
analytically assembled per-country/rate/free blocks
(`multicountry_growth_preconditioner.py`) from the residual `args`,
rebuilt by the continuation callback at each ridge-level advance; it
crosses over the dense path around N≈15 countries
(`multicountry_growth_scaling.py` is the N=2–50 ladder, ~2 minutes at
N=50 on CPU float64 vs minutes for a single dense Cholesky
factorization). Stopping is the
conjunctive `gtol`/`residual_atol` rule; the selection is resolved to
`gtol / ridge` (the asset driver's free `p_0` needs `gtol ~ 1e-3 * ridge`,
and the DAE drivers' atol is calibrated between the ridge-floor residual and
the last continuation level's so the solve can only stop at the floor).
`extra/trade_growth` keeps the metric-damped `LevenbergMarquardt`
(`repeated_shifted_dense_metric`): its two-phase gram-CG + whitened-LSMR
endgame builds Schur and time-block preconditioners in the metric's whitened
coordinates, a migration deferred until those are re-derived for the ridge
geometry.

UNO comparisons are CPU/float64 reference solves. Their native optimization
state and derivative buffers are host-side float64 arrays; NLLS and NN are the
methods intended for float32 or GPU execution.

The main NN modules are deliberately limited:

- `neoclassical_growth_nn.py`: direct sequence-space MLP.
- `neoclassical_growth_nn_recursive.py`: recursive NN with separate TFP and
  capital states, exogenous `dz/dt = g z`, and a homothetic investment-share
  representation on `log(k) - log(z)`. Training collocation can use exact
  aligned knots from the fixed RK4 rollout or Tsit5's genuinely adaptive
  accepted internal states; both modes retain static residual shapes through an
  accepted-prefix mask and use the same fixed RK4 held-out evaluator. The main
  paper driver retains its legacy `g=0`, single-path and fixed-RK4 defaults;
  `neoclassical_growth_kernel_recursive.py` is the kernel counterpart on the
  same state-space formulation, with positive growth and held-out
  initial-condition draws.
- `open_economy_growth_nn.py`: the open-economy sequence-space MLP. It carries
  the same five RKHS paths the kernel does as network outputs of `t`, imposes
  the same five DAE residuals on the same front-loaded collocation grid, and
  anchors `k(0) = k_0` and `a(0) = a_0` on the network's own value at zero --
  leaving `c`, `mu`, and `q` free there, the NN counterpart of the kernel's
  three unpenalized offsets. Saddle-path selection is the solver's
  minimum-parameter-step bias from a near-zero head, with no ridge or RKHS
  norm; the stable-manifold gap is reported ex post rather than imposed.
- `open_economy_growth_nn_recursive.py`: the open-economy state-space policy.
- `multicountry_growth_nn.py`: the reduced one-good multicountry sequential NN;
  consumption, investment, and the common rate are calculated from its four
  country path blocks inside the residual evaluation.
  `--long_run_diagnostics true` reports the same long-run validity metrics
  as the kernel driver. Unpinned width/depth resolve from the country
  count (verified through N=10, where the drawn heterogeneous initial
  conditions need depth 4); the NN is the small/moderate-N baseline —
  its dense LM works on an iterate-dependent Jacobian with no
  kernel-style preconditioner structure, so the kernel driver's `cg`
  path is the large-N vehicle.

Shared activation and transform helpers are in `utilities.py`.

Every solver returns `(results, extras)`, writes nothing itself, and exposes a
`jsonargparse` CLI. For example:

```bash
uv run python neoclassical_growth_kernel.py --kernel_set.ell 5.0
uv run python neoclassical_growth_nn_recursive.py --seed 3 --data_set.train_steps 48
uv run python neoclassical_growth_nn_recursive.py --seed 3 \
  --data_set.ode_solver tsit5
uv run python multicountry_growth_kernel.py --a_0 '[-0.5, 0.5]'
uv run python multicountry_growth_kernel.py \
  --parameters.populations '[1.0, 1.0, 1.0]' \
  --parameters.productivities '[1.0, 1.0, 1.0]' \
  --lm_set.linear_solver cg \
  --initial_set.seed 3
uv run python multicountry_growth_kernel.py --sensitivity_time 10
uv run python multicountry_growth_nn.py --a_0 '[-0.5, 0.5]' --seed 0
```

The multicountry drivers DRAW heterogeneous initial conditions whenever
`k_0`/`a_0` are not passed explicitly (identical countries make the
cross-country dynamics degenerate): lognormal `k_0` and population-demeaned
dispersed `a_0`, with the seed and dispersions on the CLI
(`--initial_set.seed`, `--initial_set.k_0_log_sigma`,
`--initial_set.a_0_sigma`). The 50-country stress configuration is
`multicountry_growth_scaling.py` — the N=2-50 ladder with drawn
populations, productivities, AND initial conditions (nothing symmetric
anywhere): every rung passes the deterministic acceptance gates, N=50 in
~2 minutes on CPU float64 via `linear_solver="cg"` where the dense
Cholesky factorization alone takes several minutes per solve. On top of
the residual gates, every rung must pass the long-run economic validity
bars at t=100: flat consumption, capital, and debt; capital at the theory
steady state with MPK equalized across countries at `delta + rho/e`;
`r -> rho`; and the exact one-good sharing invariants — constant
consumption ratios and consumption levels proportional to initial total
wealth `a_0 + PV(resources)`. The wealth ordering of long-run consumption
is gated through the largest wealth separation among misordered pairs
(exact rank equality is tie-fragile once 50 drawn wealths land within the
proportionality spread of each other); the a_0-only ranking is reported
but not gated, since drawn `k_0` and productivity heterogeneity
legitimately break it.

`multicountry_growth_kernel.py` threads every structural parameter and
initial condition through the solver's implicit-AD `p` argument, so
`--sensitivity_time 10` reports the comparative statics of the global
interest rate at t=10 with respect to all of them from ONE reverse-mode
pass (`d_rate_d_*` result keys and `extras["sensitivity_df"]`), on the
dense and matrix-free paths alike (the `cg` implicit tangent runs the
block-eigen-preconditioned CG at zero damping). The `a_0` gradient is
projected onto the feasible clearing set; identical homothetic
preferences make the rate exactly neutral to such redistributions, and
the reported gradient confirms it at ~1e-11. The reverse-mode values
carry the Gauss-Newton implicit-sensitivity bias of a non-interpolating
collocation solution — verified ~0.7% against central finite differences
of full re-solves, shrinking proportionally with the collocation
residual. For observables beyond the built-in rate sensitivity,
`extras["parametric_model"]` re-solves at any `p` and evaluates the
paths, so `jax.jvp`/`jax.grad` through it differentiates anything:
`multicountry_growth_transitions.py` is the worked example — the world
saving rate, interest rate, output Gini (sigma-convergence), and NFA
dispersion on a coarse time grid with every `d/d gamma` path from ONE
forward tangent pass, plus the high-dimensional redistribution Jacobian
from one `jax.jacrev` — the consumption Gini and terminal NFA dispersion
respond country-by-country to the initial bond and capital positions
while the r(t=10) row confirms bond-redistribution neutrality at machine
zero.

or import directly:

```python
from neoclassical_growth_kernel import neoclassical_growth_kernel

results, extras = neoclassical_growth_kernel()
print(results["k_rel_error_mean"])
```

## Optional examples

[`extra/`](extra/README.md) contains the advertising and human-capital UNO
comparisons, the integrated-derivative MLP, the separate RFF NN, the adaptive
recursive kernel, the adaptive recursive NN, the unreduced
one-good multicountry DAE,
method-comparison drivers, and HPO sweep material. None are part of the main
replication command.

[`extra/trade_growth/`](extra/trade_growth/README.md) contains the full multicountry
trade-growth model, kernel and NN solvers, comparative statics, specialized
preconditioners, scaling driver, replication drivers, and tests. The simpler
one-good model and its reduced kernel and NN solvers remain at the top level;
the unreduced DAE variation is under `extra/`.

The design and verification record for the open-economy and multicountry
stepping stones is in [`docs/open_economy_growth.md`](docs/open_economy_growth.md).
The optional adaptive recursive NN is described in
[`extra/docs/neoclassical_growth_nn_recursive_adaptive.md`](extra/docs/neoclassical_growth_nn_recursive_adaptive.md).
The accepted-step Tsit5 mode of the canonical two-state recursive NN, including
its AD contract and calibration record, is described in
[`extra/docs/neoclassical_growth_nn_recursive_tsit5.md`](extra/docs/neoclassical_growth_nn_recursive_tsit5.md).
The recursive (state-space) kernel is described in
[`docs/neoclassical_growth_kernel_recursive.md`](docs/neoclassical_growth_kernel_recursive.md).

## Tests and lint

```bash
uv run python -m unittest discover -s extra/tests -v
uv run python -m unittest discover -s extra/trade_growth/tests -v
uv run ruff check .
uv run ruff format --check .
```

Enable the committed pre-commit hook once per clone with:

```bash
git config core.hooksPath .githooks
```
