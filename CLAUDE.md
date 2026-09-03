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
  (`"parameters": p.to_dict()`, plus the rare named derived scalar a paper
  table consumes — optimal advertising's `gamma`, the neoclassical kernel's
  `learn_g_hat` — and one `"<name>_set"` dict per settings object holding
  ONLY the fields the driver threads into the solver (`asdict(...)` when it
  threads them all); recursive drivers add an `"initial_conditions"` block for
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
  written twice. Initial-condition rows appear only in the flat vector,
  never in the aux dict. Sequence-space drivers share the solver-args
  structs: `KernelResidualArgs` in `models/utilities.py` (K, K_tilde, the
  optional `batch` dict of standardized always-plain grid arrays — `t`,
  `z_trend`, `g_hat` — and, for the `_uno` drivers, `ravel_pytree`'s unravel
  as a `tree_util.Partial`) and `NNXResidualArgs` (batch, graphdef, nondiff)
  in `models/neural_networks.py`; the multicountry kernel's CG path keeps
  its own `PreconditionedKernelResidualArgs` with the whitened
  preconditioner factors, and the recursive drivers keep bespoke
  `ResidualArgs`. `models/neural_networks.py` is a leaf module (like
  `kernels.py`) holding the NN pieces: the generic batch-aware `TimeMLP`
  map t -> (..., out_features) that every sequence-space NN driver
  instantiates (near-zero head, optional `final_activation` — e.g.
  `unit_softplus`, or the per-output `UnitSoftplusWhere` mask for mixed
  signed/positive heads — `output_scale` Variable, optional trend Param
  `g`); the recursive policy pieces — the generic `StateMLP` map
  state -> (..., out_features) (states arrive in the model's natural
  coordinates, logs of positive states taken by the caller;
  `normalize_state` affine normalization `input_scale * (state - center)`
  — unit `input_scale` on the closed-economy log states, the reciprocal
  standard deviation per axis in the stochastic open economy, whose kernel
  consumer inherits these coordinates — trunk, one near-zero Linear head
  at `head_kernel_std`, optional static
  `final_activation` (bare `unit_softplus`, or
  `UnitSoftplusWhere(local_slope=...)` for mixed heads — one entry per
  output, `None` on the signed slots for a linear pass-through, `1.0` for a
  unit slope, and `flow_unit / start_level` = `1/delta` on the open-economy
  investment slot so every head carries an O(flow-unit) Jacobian at the
  draw), then the `output_scale` Variable multiply; normalization and
  output scales are the non-Param Variables), with
  `normalize_state`/`UnitSoftplusWhere` also imported by the
  stochastic-open-economy kernel policy; plus `PolicySettings` (the
  width/depth/activation/head-init dataclass every recursive NN driver takes
  as `nn_set`), `unit_softplus`/`ACTIVATIONS`, `NNXResidualArgs`, and the
  policy-artifact `save_model`/`load_model`
  (load factories rebuild the `StateMLP` architecture and `final_activation`
  from meta — the open-economy slope from the artifact's
  `structural`/`initial_state` blocks — with placeholder Variables the
  nondiff block overwrites); `utilities.py` keeps the non-NN shared pieces
  and never imports `nnx`.
  Each recursive NN driver builds its `StateMLP` kwargs once as an inline
  `policy_spec` dict and reuses them verbatim in `DrawNNXModule`, so every
  kwarg stays hashable (the load factories in the shared modules spell the
  same architecture out with placeholder Variables). The epoch-recursive
  drivers pass `models/utilities.ResetEpochStream` as `MultiStart`'s
  `reset_args` hook: it re-chains `epoch_key`/`restart_key` from the
  args-carried `restart_key` (data-seed stream, never the multi-start key)
  and zeroes the epoch counter at each retry.
- `replication/` — ensemble + figure drivers; `collect_json.py` writes
  `.results/all_results.json`, the only place paper numbers are selected and
  rounded. Generated JSON is gitignored; track one with a targeted `!path` rule.
  Every number embedded in the paper tex carries a no-op `\jsonpath{slash/path}`
  tag immediately after it, pointing at its `all_results.json` leaf; a number
  without a leaf gets one added in `collect_json.py` first.
  `uv run python -m replication.verify_paper_tags` checks that every tag
  resolves and literally equals the printed value;
  `uv run python -m replication.export_paper_assets` copies the figures and
  generated tables into the paper repo.
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
  penalized free constants (asset pricing needs `gtol ~ 1e-3 * ridge`); the
  converged residual norm floors at O(ridge * ||dual||), so `atol` sits above
  it. Tuning: https://highdimensionaleconlab.github.io/nlls_gram/ridge_lm/
- Settings taxonomy: `lm_set: OptimizerSettings` carries exactly the LM
  constructor + `solve()` kwargs (`atol`/`gtol`/`max_steps` named as the
  optimizer names them), `ridge_set: RidgeSettings` the RidgeLM extras
  (ridge, continuation, linear_solver), `cg_set: CGSettings` the CG knobs,
  `solver_set: UnoOptimizerSettings` the Uno/IPOPT options; `tolerance_set`
  holds the acceptance toggles and bars — `gate_train`/`gate_validation`
  with `train_residual`/`validation_residual` in sequence space,
  `gate_train`/`gate_test` with the driver's training criterion and the
  `test_euler_*` bars in the recursive drivers (the epoch-recursive NN
  drivers also keep their training-schedule fields there).
- Acceptance is ex post and one predicate in every driver:
  `(not gate_train or train_ok) and (not gate_validation or validation_ok)`
  in sequence space, `(not gate_train or train_ok) and (not gate_test or
  test_ok)` in the recursive drivers. Sequence-space `train_ok` /
  `validation_ok` are the max abs residual on the training grid and on the
  held-out validation grid — a finer `linspace(0, train grid end,
  validation_points)` on the same path (same initial conditions), evaluated
  through the post-solve `evaluate_grid`/`snapshot` closure — under
  `tolerance_set.train_residual` / `validation_residual`; `test_*` stays the
  long-horizon diagnostic. Every bar compares with `<=` so a NaN fails;
  feasibility (nonfinite or nonpositive paths, `negative_p_0`, the mpk
  `p_bound`) is a hard, untoggled rejection reason. The optimizer's own
  status is reported as `solver_status` (plus `solution_status` under
  Uno/IPOPT) and never gates; multi-start keys on the optimizer's status
  through the library default accept, and no driver passes an `accept=`
  hook; `max_steps_is_success=False` everywhere except the epoch-recursive
  NN drivers under a multi-epoch schedule, where exhausting the schedule
  is the optimizer's normal stop.
- Every sequence-space driver represents paths in LEVELS (anchored
  representer: value = init + K_tilde @ alpha, the form the RKHS theory
  covers). The sequence-space drivers pin a fixed small ridge, except
  `multicountry`, which keeps the ridge anneal (`ridge_set.ridge_init`):
  its binding row is world bond clearing, whose residual floors at
  O(ridge * ||dual||) with a dual that grows with the country count, so
  the solve homotopes down to a per-configuration ridge floor (the
  scaling ladder runs a uniformly deeper one than the experiment matrix).
  `multicountry` with `ridge_set.linear_solver="cg"` is the large-N path:
  matrix-free preconditioned CG using the arrowhead + Woodbury structure
  assembled in `multicountry_growth_preconditioner.py`.
- Every driver threads structural parameters and initial conditions through
  the implicit-AD `p` argument; the NLLS drivers return `extras["solve"]`
  (solver, residual_fn, theta_0, p, args, options) and the sensitivity
  drivers re-solve from that handle and differentiate with respect to `p`
  (the `_uno` and kernel-recursive drivers return no solve handle).
- Recursive NN/kernel: training collocation is the integrator's own fixed-step
  RK4 states (nothing interpolated inside the residual); every recursive
  policy call is p-free — `policy(state) -> (out_features,)`, single-output
  consumers index `[0]` — with structural constants baked at construction
  (`StateMLP` output scales like `delta * k_0` live in its Variables) and
  the kernel-recursive policy structs carrying `p` as a pytree field
  instead. Every recursive policy is a function of the full state in the
  model's natural coordinates; no driver detrends or reduces the state
  inside the approximation class. The neoclassical and stochastic growth
  kernels share the ray + 2-D ARD state kernel (an RBF in
  `q = log k - ray_slope * log z` plus an anisotropic RBF in
  `(log z, log k)`) with `ray_slope` a tuned hyperparameter like the
  lengthscales and theta = `(alpha, h_const, linear_coeffs)`, alpha alone
  penalized. In the recursive drivers `train_ok` is the driver's training
  criterion (the fresh-batch loss under `train_loss_threshold` for the
  epoch NN drivers, the deterministic open economy's train-path residual
  under `train_residual`, the max training Euler residual under
  `train_euler_max` for the kernels) and `test_ok` the held-out Euler errors
  on the test batch under `test_euler_mean_abs` / `test_euler_max` (plus the
  Ito-term bar in the stochastic open economy). The epoch callback stops the
  solve only on the training criterion (a fresh-batch loss under
  `train_loss_threshold` sets `CONVERGED`; `epoch_early_stopping` toggles
  it), so the held-out batch lives in driver scope and is simulated once ex
  post. Transversality and benchmark errors are reported, never gated.
- The stochastic recursive NN drivers always train and write
  content-addressed policy `.npz` artifacts via `--policy_cache <dir>` (a
  pure save target); the kernel-recursive consumers read them via
  `--policy_dir`.

## Conventions

- x64 + `jax_default_matmul_precision="highest"` set by every entrypoint
  before any array is created.
- Scales multiply: `input_scale` and `output_scale` on `TimeMLP` and
  `StateMLP` are multipliers (sequence-space `input_scale = 0.01`, a
  literal — never derived from a horizon — with asset pricing alone at
  `0.025`), as are every `*_scale` setting.
- Numeric settings are used verbatim — never clamped, floored, or defensively
  adjusted in code; guidance (e.g. keep `train_residual` above ~8x machine
  epsilon for the run dtype) rides as comments on the field defaults.
- Seeds: 7 for sequence-space NN ensembles, 5 for recursive NN; each NN seed
  gets up to `lm_set.num_attempts` multi-start attempts.
- Lint: `uv run ruff check .` and `uv run ruff format .`; enable the committed
  hook once per clone with `git config core.hooksPath .githooks`.
