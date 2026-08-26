# CLAUDE.md

Editing hints for this repo; user-facing docs are README.md and the
per-directory READMEs.

## Layout

- `models/` — one self-contained solver per model/method: returns
  `(results, extras)`, writes nothing, exposes a jsonargparse CLI. Unqualified
  kernel filenames are the NLLS formulation; `_uno` is the constrained IPOPT
  comparison (host-side float64 only — NLLS/NN are the float32/GPU methods).
  `<model>_shared.py` holds model structure only (constants, the flat
  `StructuralParameters` pytree — deep parameters plus initial conditions,
  subclassing `models/utilities.ParameterPyTree` for `create(dtype=...)` and
  `to_dict()`; IC fields may default to None for drivers that keep ICs in the
  solver args, and to_dict skips None — production/steady-state functions,
  diagnostics, and shared settings); drivers never import other drivers.
  `results` is the full serializable record: nested input blocks first
  (`"parameters": p.to_dict()` plus one `"<name>_set": asdict(...)` per
  settings object; recursive drivers add an `"initial_conditions"` block for
  draw centers not on `p`), then flat measured outcomes only — never plain
  copies of settings fields or parameters. `extras` carries live objects
  (solve spec, DataFrames, model, result); there is no
  `extras["hyperparameters"]` — `replication.utilities.ensemble` splits the
  dict-valued results entries into `<exp>_hyperparameters.json` and keeps the
  scalar rows for the DataFrame. Each kernel and kernel_uno
  driver owns its `Theta` and `residual_fn` — the residual rows written
  inline. Sequence-space `residual_fn(theta, args, p)` returns
  `(scaled flat residual, values)` where `values` is a dict of paths plus the
  UNSCALED named residual rows; the NLLS solvers run with `has_aux=True`
  (`_uno` `constraints` wrappers take `[0]`), and the post-solve
  `evaluate_grid(points)` / `snapshot` closures build eval-grid args, consume
  `residual_fn(...)[1]`, and add derived diagnostics on top — rows are never
  written twice. Sequence-space drivers share the solver-args
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
- The stochastic recursive NN drivers always train and write
  content-addressed policy `.npz` artifacts via `--policy_cache <dir>` (a
  pure save target); the kernel-recursive consumers read them via
  `--policy_dir`.

## Conventions

- x64 + `jax_default_matmul_precision="highest"` set by every entrypoint
  before any array is created.
- Seeds: 7 for sequence-space NN ensembles, 5 for recursive NN; each NN seed
  gets up to `lm_set.num_attempts` multi-start attempts.
- Lint: `uv run ruff check .` and `uv run ruff format .`; enable the committed
  hook once per clone with `git config core.hooksPath .githooks`.
