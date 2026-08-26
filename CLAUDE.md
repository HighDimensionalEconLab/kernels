# CLAUDE.md

Editing hints for this repo; user-facing docs are README.md and the
per-directory READMEs.

## Layout

- `models/` — one self-contained solver per model/method: returns
  `(results, extras)`, writes nothing, exposes a jsonargparse CLI. Unqualified
  kernel filenames are the NLLS formulation; `_uno` is the constrained IPOPT
  comparison (host-side float64 only — NLLS/NN are the float32/GPU methods).
  `<model>_shared.py` holds model structure only (constants, the flat
  `StructuralParameters` struct — deep parameters plus initial conditions —
  production/steady-state functions, diagnostics, shared settings, and for
  the DAE models `path_residuals`, the equation system every representation
  consumes); drivers never import other drivers. Each kernel and kernel_uno
  driver owns its `Theta` and `residual_fn` — the residual rows written
  inline — plus a post-solve `evaluate_grid(points)` closure for off-grid
  paths and residual rows. Sequence-space drivers share the solver-args
  structs in `models/utilities.py`: `KernelResidualArgs` (K, K_tilde, the
  optional `batch` dict of standardized always-plain grid arrays — `t`,
  `z_trend`, `g_hat` — and, for the `_uno` drivers, `ravel_pytree`'s unravel
  as a `tree_util.Partial`) and `NNXResidualArgs` (batch, graphdef, nondiff);
  the multicountry kernel's CG path keeps its own
  `PreconditionedKernelResidualArgs` with the whitened preconditioner
  factors, and the recursive drivers keep bespoke `ResidualArgs`.
- `replication/` — ensemble + figure drivers; `collect_json.py` writes
  `.results/all_results.json`, the only place paper numbers are selected and
  rounded. Generated JSON is gitignored; track one with a targeted `!path` rule.
- `notebooks/` — annotated `.qmd` (source of truth) + committed executed
  `.ipynb` + prose-free `*_minimal.py`; fully self-contained, no `models/`
  imports; rebuild with `uv run quarto render notebooks`.
- `extra/` — UNO variants and HPO sweeps outside the main replication.

## Solver invariants

- Kernel solves: `RidgeLevenbergMarquardt` on
  `||r||^2 + ridge * sum_j alpha_j' K alpha_j`, metric =
  `RepeatedFactorMetric` on the upper Cholesky factor of K. The metric
  penalizes the LEADING flat coordinates of theta: alpha blocks first,
  structural scalars (`c_0`, `mu_0`, `p_0`, ...) in the trailing unpenalized
  free block (dict thetas flatten by sorted key — keep alpha keys sorting
  first). Minimizer = min-RKHS-norm interpolant up to an O(ridge) bias.
- Stopping is conjunctive `gtol` AND `atol`; `gtol` must resolve the weakly
  penalized free constants (asset pricing needs `gtol ~ 1e-3 * ridge`).
- Settings taxonomy: `lm_set: OptimizerSettings` carries exactly the LM
  constructor + `solve()` kwargs (`atol`/`gtol`/`max_steps` named as the
  optimizer names them), `ridge_set: RidgeSettings` the RidgeLM extras
  (ridge, continuation, linear_solver), `cg_set: CGSettings` the CG knobs,
  `solver_set: UnoOptimizerSettings` the Uno/IPOPT options; `tolerance_set`
  holds ex-post acceptance gates only (the epoch-recursive NN drivers also
  keep their training-schedule fields there).
- Simple ODE drivers pin a fixed small ridge; the DAE drivers (`open_economy`,
  `multicountry`) anneal via `ridge_set.ridge_init`. `multicountry` with
  `ridge_set.linear_solver="cg"` is the large-N path: matrix-free
  preconditioned CG using the arrowhead + Woodbury structure assembled in
  `multicountry_growth_preconditioner.py`.
- Every driver threads structural parameters and initial conditions through
  the implicit-AD `p` argument and returns `extras["solve"]` (solver,
  residual_fn, theta_0, p, args, options); the sensitivity drivers re-solve
  from that handle and differentiate with respect to `p`.
- Recursive NN/kernel: training collocation is the integrator's own fixed-step
  RK4 states (nothing interpolated inside the residual); the policy is
  `i = delta * k * unit_softplus(h)` on the homothetic state; acceptance is
  solver convergence only — Euler residuals, transversality, and benchmark
  errors are reported, never gated.
- Stochastic recursive NN drivers accept `--policy_cache <dir>`
  (content-addressed `.npz`); the kernel-recursive consumers read it via
  `--policy_dir`. Replication drivers never pass `--policy_cache` — cached
  runs would corrupt the timing tables.

## Conventions

- x64 + `jax_default_matmul_precision="highest"` set by every entrypoint
  before any array is created.
- Seeds: 11 for sequence-space NN ensembles, 5 for recursive NN; each NN seed
  gets up to 5 multi-start attempts.
- Lint: `uv run ruff check .` and `uv run ruff format .`; enable the committed
  hook once per clone with `git config core.hooksPath .githooks`.
