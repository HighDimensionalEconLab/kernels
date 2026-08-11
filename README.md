# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

Replication code for [Solving Models of Economic Dynamics with Ridgeless Kernel
Regressions](https://arxiv.org/pdf/2406.01898) by Mahdi Ebrahimi Kahou, Jesse
Perla, and Geoff Pleiss.

The top level contains the canonical examples. Kernel solves use the released
`nlls-gram` Levenberg--Marquardt implementation. Asset pricing and growth also
include constrained UNO comparisons; nonessential variations live in
[`extra/`](extra/README.md).

## Setup

```bash
uv sync
```

The numerical stack requires Python 3.13, `nlls-gram>=2.6.0`, and
`tinydiffeq>=2.2.0`. Small kernel and NN examples use `nlls-gram`'s automatic
linear-solver choice. The reduced one-good multicountry kernel also exposes a
matrix-free preconditioned-CG path for large country counts.

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
- `asset_pricing_kernel.py`
- `asset_pricing_kernel_uno.py`
- `asset_pricing_nn.py`
- `optimal_advertising_kernel.py`
- `human_capital_kernel.py`
- `neoclassical_growth_nn.py`
- `neoclassical_growth_nn_recursive.py`
- `neoclassical_growth_kernel_recursive.py`
- `stochastic_growth_nn_recursive.py`
- `open_economy_growth_kernel.py`
- `open_economy_growth_kernel_uno.py`
- `open_economy_growth_nn.py`
- `open_economy_growth_nn_recursive.py`
- `stochastic_open_economy_growth_nn_recursive.py`
- `multicountry_growth_kernel.py`
- `multicountry_growth_nn.py`

Each driver in `replication/` writes Parquet data and summary JSON. Figure
drivers read those saved results, and `replication/collect_json.py` assembles
the authoritative paper-number file. Sequence-space NN baselines use 11 seeds;
recursive NN baselines (neoclassical, stochastic, open-economy) use 5 seeds,
with the recursive growth driver's variant experiments at one seed each; the
multicountry NN replication uses five seeds and also pins a 10-country
width-32/depth-4 experiment without changing the solver defaults.
Every NN driver allows each seed up to five multi-start restarts
(`stop_set.num_attempts`), so `frac_accepted` measures acceptance within that
restart budget rather than first-try robustness.
The two open-economy paths included only
for the transition figure use one displayed seed each. The float64 pass also
runs the multicountry scaling ladder
(`replication/multicountry_growth_scaling.py`) behind the timing figure. A
deterministic kernel
specification is solved with the persistent JAX cache enabled;
`DETERMINISTIC_TIMING_REPETITIONS` in `replication/utilities.py` sets the
repetition count, the timing runs are retained in Parquet, and the JSON reports
their median.

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
  threshold experiment. `neoclassical_growth_concave_convex_nn.py` runs the
  same 70-point `k_0` sweep with the sequence-space NN formulation. Basin
  correctness is scored independently of solver acceptance in all three
  (kernel, UNO, sequential NN): a solution may still be in motion at `test_T`
  and is classified by the side it ends on plus the direction of its tail.
  Neither test feeds back into the solve.
- `optimal_advertising_kernel.py`: optimal advertising.
- `human_capital_kernel.py`: the two-capital human-capital model.
- `open_economy_growth_kernel.py`: the one-country open-economy DAE.
- `open_economy_growth_kernel_uno.py`: the constrained comparison using the
  same open-economy DAE paths, equations, and collocation grid with the
  function-only RKHS norm as its objective.
- `multicountry_growth_kernel.py`: the reduced one-good multicountry kernel
  formulation, with the country consumption and investment FOCs substituted.

Unqualified kernel filenames refer to the NLLS formulation; `_uno` identifies
the constrained comparison. Every driver uses
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
the whitened normal operator. Away from the asset-clearing rows no
residual row couples two countries, so the operator is an arrowhead --
per-country blocks bordered by the shared interest-rate block -- plus a
rank-`n_train` Woodbury term of the whitened clearing columns;
`multicountry_growth_preconditioner.py` assembles exactly that structure
from per-time Jacobian weights (O(N) build, no materialized Jacobian),
and a solve callback rebuilds it from the live iterate every LM step, so
each inner solve is preconditioned by the exact shifted inverse at its
build point. The cg path lands on the dense solution (identical residuals
and step counts) at O(N) cost per step, overtaking dense Cholesky by
N≈3 (2.1s vs 27.1s at N=10 on CPU float64);
`replication/multicountry_growth_scaling.py` runs the N=2-40 timing
ladder behind the paper's `multicountry_growth_scaling.pdf`. Stopping is
the conjunctive `gtol`/`residual_atol` rule; the selection is resolved to
`gtol / ridge` (the asset driver's free `p_0` needs `gtol ~ 1e-3 * ridge`,
and the DAE drivers' atol sits above the ridge-floor residual -- the
multicountry default admits a stop one anneal level above the floor, and
the achieved level is reported as `ridge`).

UNO comparisons are CPU/float64 reference solves. Their native optimization
state and derivative buffers are host-side float64 arrays; NLLS and NN are the
methods intended for float32 or GPU execution.

The main NN modules are deliberately limited:

- `neoclassical_growth_nn.py`: direct sequence-space MLP.
- `neoclassical_growth_nn_recursive.py`: recursive NN with separate TFP and
  capital states and exogenous `dz/dt = g z`. The policy is
  `i = delta * k * unit_softplus(h)` with the head reading `log(k) - log(z)`
  (the only homothetic ingredient), so the zero-initialized policy holds
  capital at rest in levels and no steady state or production structure enters
  the map. Every rollout is fixed-step RK4 and training collocation is the
  integrator's own states: `train_points` knots from `train_points - 1` steps,
  with nothing interpolated inside the differentiated residual. Train and
  test initial conditions are independent draws around the scalar
  `(z_0, k_0)` with per-axis log variances; acceptance is solver convergence
  and everything else (train/test Euler residuals, transversality, benchmark
  path errors) is reported, never gated. The paper driver runs three
  experiments: `single_ic_g0`, `multi_ic_g0`, and `multi_ic_g02`.
  `neoclassical_growth_kernel_recursive.py` is the kernel counterpart on the
  same state-space formulation with the identical output map and
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
- `stochastic_open_economy_growth_nn_recursive.py`: the stochastic small open
  economy on the recursive state `(log z, log k, a / output_scale, rate_gap)`
  with correlated OU TFP and world-rate shocks. Two anchored heads supply
  positive investment and the signed current account; training rolls SDE
  collocation paths from a single initial state by default (trajectories
  differ only by their shock realization -- initial-condition dispersion is a
  CLI experiment) and a shock-continuation ladder is the only homotopy: no
  benchmark, steady-state, or pretraining information enters the residual or
  initialization. The deterministic open-economy benchmark appears only in
  ex-post reported diagnostics.
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

Every solver returns `(results, extras)`, writes nothing itself (beyond the
opt-in policy artifacts below), and exposes a `jsonargparse` CLI. For example:

```bash
uv run python -m models.neoclassical_growth_kernel --kernel_set.ell 5.0
uv run python -m models.neoclassical_growth_nn_recursive --seed 3 --data_set.train_points 49
uv run python -m models.multicountry_growth_kernel --a_0 '[-0.5, 0.5]'
uv run python -m models.multicountry_growth_kernel \
  --parameters.populations '[1.0, 1.0, 1.0]' \
  --parameters.productivities '[1.0, 1.0, 1.0]' \
  --lm_set.linear_solver cg \
  --initial_set.seed 3
uv run python -m models.multicountry_growth_kernel --sensitivity_time 10
uv run python -m models.multicountry_growth_nn --a_0 '[-0.5, 0.5]' --seed 0
```

The multicountry drivers DRAW heterogeneous initial conditions whenever
`k_0`/`a_0` are not passed explicitly (identical countries make the
cross-country dynamics degenerate): lognormal `k_0` and population-demeaned
dispersed `a_0`, with the seed and dispersions on the CLI
(`--initial_set.seed`, `--initial_set.k_0_log_sigma`,
`--initial_set.a_0_sigma`). The scaling ladder is
`replication/multicountry_growth_scaling.py`: symmetric structural
parameters with drawn initial conditions at N = 2, 3, 5, 10, 15, 25, 40,
every rung required to pass the standard acceptance gates with
`linear_solver="cg"` and — up to N = 15 — the dense Cholesky reference on
identical solver settings. The ladder pins `residual_atol = 2e-7` and
deepens the ridge floor at the wide rungs: the converged residual floors
at O(ridge ||dual||) and the dual norm grows with the cross-section, so
the production stopping calibration would breach the fixed train gate
past N ≈ 10. The cg column stays near-linear in N (~7 s at N = 40 on CPU
float64) while dense grows as N³ (~113 s at N = 15, ~29x the cg time);
`multicountry_growth_scaling_figures.py` renders the log-log timing
figure and `collect_json.py` publishes the per-rung times and the
dense/cg ratio at the cap into `all_results.json`.

`multicountry_growth_kernel.py` threads every structural parameter and
initial condition through the solver's implicit-AD `p` argument, so
`--sensitivity_times '[0,10,30,80]'` reports the comparative statics of
the world interest rate, three world statistics (saving rate, NFA
dispersion, output Gini), and every country's consumption and capital
with respect to all of them, as `d_<observable>_T<t>_d_<parameter>`
result keys and `extras["sensitivity_df"]`. `--sensitivity_mode reverse`
builds the whole Jacobian, one reverse pass per output scalar;
`forward` linearizes once and applies the tangent map only along the
directions published — the `(1, ..., 1)` direction in `gamma` and one
bond direction per country. Both cost about the same, since the re-solve
inside the differentiated function dominates either.

Every country block also carries a `_common` entry, the derivative along
`(1, ..., 1)`: the country count cancels, so it is the one country-block
statistic comparable across N. The replication runs it at N=2 and N=10
and the two must agree on `d ln r* / d gamma` despite solving systems of
1,920 and 9,792 rows — the measured gap is 0.57 percentage points, in
`aggregate_sensitivity.max_rate_percent_gap_vs_two_countries`.

The `a_0` gradients are projected onto the feasible clearing set;
identical homothetic preferences make the rate exactly neutral to such
redistributions, and `max_i |d r*/d b_i0|` confirms it below 5e-10 at
t = 10, 30, 80, rising to ~2e-6 at t = 0, where the integrated kernel
vanishes and the rate is carried entirely by its unpenalized offset. The
reverse-mode values carry the Gauss-Newton implicit-sensitivity bias of a
non-interpolating collocation solution — verified ~0.7% against central
finite differences of full re-solves, shrinking proportionally with the
collocation residual.

The stochastic recursive NN drivers (`stochastic_growth_nn_recursive.py`,
`stochastic_open_economy_growth_nn_recursive.py`) additionally accept
`--policy_cache <dir>`: the trained policy is saved there as a
content-addressed `<model>_policy_<problem>-<run>.npz` artifact, and an
existing artifact for the identical configuration is loaded instead of
retraining (`--save_policy <path>` writes one explicit extra copy). The
kernel-recursive consumers glob the problem half of that key under
`--policy_dir` and freeze the best accepted NN policy's own collocation
rollout as their design, so the workflow is two-stage:

```bash
uv run python -m models.stochastic_growth_nn_recursive --policy_cache .results
uv run python -m models.stochastic_growth_kernel_recursive --policy_dir .results
```

The replication drivers deliberately do not pass `--policy_cache` (cached
runs would corrupt the timing tables), so the kernel-recursive stage needs
its NN stage run manually first.

or import directly:

```python
from models.neoclassical_growth_kernel import neoclassical_growth_kernel

results, extras = neoclassical_growth_kernel()
print(results["k_rel_error_mean"])
```

## Optional examples

[`extra/`](extra/README.md) contains the advertising and human-capital UNO
comparisons, the recursive kernel-vs-NN comparison driver, and HPO sweep
material. None are part of the main replication command.

## Tests and lint

```bash
uv run ruff check .
uv run ruff format --check .
```

Enable the committed pre-commit hook once per clone with:

```bash
git config core.hooksPath .githooks
```
