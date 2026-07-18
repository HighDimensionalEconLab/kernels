# Multicountry trade-growth LLS testbed

This is the development harness for alternative linear solvers and
preconditioners in the full \(N\)-country trade-and-growth model. It generates
deterministic, mildly heterogeneous economies from a typed JAX key. Initial
aggregate asset positions sum exactly to zero, while capital, population,
productivity, value-added shares, and bilateral iceberg costs vary by seed.

Each case exposes the linearized damped dual system

$$
(J M^{-1}J^\top+\lambda I)y=r,
\qquad
\Delta\theta=-M^{-1}J^\top y,
$$

using the production DAE residual and exact float64 RKHS metric (the whole
harness runs with `jax_enable_x64`). The case object provides matrix-free
Jacobian, transpose-Jacobian, metric solve, dual matvec, and step-recovery
callbacks. The dual matvec is parameterized by damping: `case.dual_matvec_base`
is the undamped \(J M^{-1} J^\top\), `case.dual_matvec_with(lambda)` returns the
matvec for a given \(\lambda\) (including \(\lambda=0\)), and
`case.dense_dual_matrix(damping=...)` materializes it densely. It also provides:

- native and time-major residual permutations;
- the \(12N+1\) residual block size at each collocation date;
- dynamic-kernel, interest-rate, global-offset, and static-coordinate
  boundaries in the flattened unknown vector;
- optional linearization after a chosen number of dense LM steps;
- a dense Cholesky reference and an intentionally weak identity-CG baseline;
- compile time, repeated synchronized solve time, iterations, linear residual,
  predicted residual, and error relative to the dense step.

Run a small CPU sweep with:

```bash
JAX_PLATFORMS=cpu uv run python \
  -m experiments.multicountry_trade_growth_lls_testbed \
  --country_counts '[2,3,4]' \
  --train_point_counts '[5,9]' \
  --seeds '[0,1]' \
  --algorithms '[dense_cholesky,cg_identity]' \
  --dampings '[1e-6,1e-2,0.0]' \
  --output_path .results/multicountry_trade_growth_lls.json
```

The `--dampings` list benchmarks a single built case across the whole LM
\(\lambda\) range without rebuilding it; each \(\lambda\) is compared against its
own per-\(\lambda\) dense reference. Include `0.0` to exercise the undamped
implicit-path operator; at \(\lambda=0\) the dense dual can be singular, so the
reference reports its minimum eigenvalue and falls back to a least-squares solve
instead of crashing. Omitting `--dampings` keeps the single `--damping` value
(default `1e-6`).

Pass `--matrix_dump_dir PATH` to save, per case, a compressed
`{case.name}_matrices.npz` holding the stacked dense dual matrices (one per
requested damping, keyed alongside a `dampings` array), the dense Jacobian, the
residual, the native/time-major permutations, and the unknown-layout block
boundaries. The dump reuses `case.dense_dual_matrix()` / `case.dense_jacobian()`
and is gated by `--matrix_dump_max_residuals` (default `3000`); larger cases are
skipped with a printed message rather than writing an \(m^2\) dense matrix.

For larger cases, lower `dense_reference_max_residuals` or set it to zero so
the harness does not materialize the dense dual. Vary `linearization_steps` to
test the warm start and later nonlinear iterates separately.

An external algorithm can be loaded without editing the harness by passing
`module:function` in `algorithms`. Its callable contract is:

```python
def solve(case, rhs, settings):
    # case.dual_matvec accepts vectors and column matrices.
    # Return JAX values so the complete call can be jitted and timed.
    return dual_solution, iterations, converged
```

For a method with a nontrivial factorization, sketch, or preconditioner build,
the imported object can instead provide `prepare(case, settings)` and
`solve(case, prepared, rhs, settings)` methods. The harness then reports setup
and solve compilation and steady-state times separately. This distinction is
important for frozen or periodically refreshed preconditioners.

Block methods should use `case.to_time_major` and `case.to_native_order`
instead of assuming that the production residual is already time-major. An
approximate preconditioner is allowed; `case.metric_solve` and
`case.recover_step` must remain exact because the metric defines solution
selection.
