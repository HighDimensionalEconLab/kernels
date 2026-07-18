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

using the production DAE residual and exact mixed-precision RKHS metric. The
case object provides matrix-free Jacobian, transpose-Jacobian, metric solve,
dual matvec, and step-recovery callbacks. It also provides:

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
  --algorithms '[dense_cholesky,pcg_identity]' \
  --output_path .results/multicountry_trade_growth_lls.json
```

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
