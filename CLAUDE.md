# CLAUDE.md

Editing hints for this repo; user-facing docs are README.md and the
per-directory READMEs.

## Layout

- `models/` — one self-contained solver per model/method: returns
  `(results, extras)`, writes nothing, exposes a jsonargparse CLI. Unqualified
  kernel filenames are the NLLS formulation; `_uno` is the constrained IPOPT
  comparison (host-side float64 only — NLLS/NN are the float32/GPU methods).
  `<model>_shared.py` holds everything used by more than one driver (model
  structure plus the kernel Theta/residual/evaluation); drivers never import
  other drivers. The `_uno` drivers map IPOPT's flat vector to the NLLS Theta
  via `ravel_pytree`'s unravel riding in `args` as a `tree_util.Partial`.
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
- Simple ODE drivers pin a fixed small ridge; the DAE drivers (`open_economy`,
  `multicountry`) anneal via `ridge_continuation`. `multicountry` with
  `lm_set.linear_solver="cg"` is the large-N path: matrix-free preconditioned
  CG using the arrowhead + Woodbury structure assembled in
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
