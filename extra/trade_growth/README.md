# Multicountry trade and growth

This subproject solves the trade-growth transition as one global sequence-space
differential-algebraic system. It uses the current `nlls-gram` interfaces and
the same scalable parameter and random-initial-condition patterns as the
multicountry-growth examples. It does not use a time-stepping DAE solver.

## Coordinates and grid

For each of the N countries, the kernel represents four differential paths
(`k`, net foreign assets, Tobin's `q`, and marginal wealth) and eight algebraic
trade paths. The common interest rate is one additional global path. A grid
with T collocation dates therefore gives `(12N + 1)T` equations and RKHS
coefficient unknowns.

Here “grid” means the dates at which the global RKHS path must satisfy the DAE,
not time-stepping nodes. The maintained production grid has 48 dates over
`[0, 100]`: it is dense through year 40 and has eight tail dates. Diagnostics
evaluate the interpolated path at 401 evenly spaced dates over the full
horizon, so passing the equations only at the collocation dates is not enough.

The initial capital stocks and NFA positions are anchored by `p["k_0"]` and
`p["a_0"]`. The actual initial log values of `q`, marginal wealth, all eight
static variables, and the common interest rate are free unknowns. The static
equilibrium routine supplies only a numerical starting guess.

Every economic input that may be differentiated later is in `p`:

- all scalar structural parameters;
- country populations, productivities, and value-added shares;
- the full N by N bilateral iceberg-trade-cost matrix;
- `k_0` and `a_0`.

The NFA vector is projected onto population-weighted world clearing inside the
parametric path map. Derivatives with respect to `a_0` therefore remain on the
economically feasible clearing manifold.

## RidgeLM and MetricLM

`SolverSettings(optimizer="ridge_lm")` retains the conservative baseline. Its
objective is

\[
  \lVert R(\theta,p)\rVert_2^2
  + \lambda \sum_{j=1}^{12N+1} \alpha_j^{\mathsf T}K\alpha_j.
\]

Only the RKHS coefficients are penalized. The `10N + 1` free initial values,
including `mu_0` and every static `y_0`, are not penalized.

`SolverSettings(optimizer="metric_lm")` instead minimizes the unmodified
residual objective and uses the positive-definite damping metric

\[
 W_\epsilon = \operatorname{blockdiag}\left(
   \underbrace{K+\epsilon I,\ldots,K+\epsilon I}_{12N+1\text{ paths}},
   \epsilon I_{10N+1}
 \right).
\]

The epsilon block on the free initial values is required to make the metric
invertible. It affects the LM correction geometry but is not a ridge penalty
in the objective. MetricLM is path-dependent: it selects minimum-metric
corrections at each step, not a globally minimum-norm root. The historical
`epsilon=1e-7` remains the retained value.

Dense `Cholesky()` is the verified baseline; `QR()` is an independent dense
check. Both solve the same equations, but on the production problem their
accepted MetricLM paths differ by about `7.6e-5`, so MetricLM remains opt-in.

## Iterative preconditioners

Two matrix-free families are available.

`linear_solver="cg"` uses parameter-space CG with country modes. The aggregate
country mode is grouped with the common interest-rate path, while contrast
modes capture trade reallocation. The balanced inverse retains all
aggregate-to-contrast couplings and is SPD.

`linear_solver="gram_cg"` is MetricLM-only and works in residual space. Its
retained `hybrid_blocks` preconditioner combines:

- full within-date blocks, which retain all country, trade, numeraire, and
  common-rate interactions, plus the exact global low-rank contribution of
  the initial values;
- full-horizon blocks for each economic residual stream, which retain the
  kernel's temporal stiffness.

The hybrid is refreshed after accepted steps. It is correct for N=2 and N=3,
but production timings deteriorate from about 31 seconds to 94 seconds, while
dense MetricLM grows from about 6 seconds to 13 seconds. Consequently no runs
above N=3 are maintained and neither iterative method is the default.

## Differentiation

`extra["solve_parametric"]` and `extra["parametric_model"]` use the matching
implicit AD rule. All structural parameters, bilateral trade costs, `k_0`, and
`a_0` remain differentiable through `p`. Tests cover forward/reverse duality,
the shifted metric maps, and dense versus iterative tangents on small grids.

`comparative_statics.py` is a statistics layer over this same full DAE; it
does not construct a second parameterization.

## Running

From the repository root, the conservative baseline is:

```bash
uv run python -m extra.trade_growth.kernel
```

The retained dense MetricLM N=2/N=3 study is:

```bash
uv run python -m extra.trade_growth.scaling \
  --country_counts '[2,3]' \
  --linear_solvers '["auto"]' \
  --optimizer metric_lm \
  --metric_eps 1e-7 \
  --damping_decrease 0.5 \
  --residual_atol 2e-8
```

The iterative fallback needs its looser stopping threshold and is best run
separately:

```bash
uv run python -m extra.trade_growth.scaling \
  --country_counts '[2,3]' \
  --linear_solvers '["gram_cg"]' \
  --optimizer metric_lm \
  --metric_eps 1e-7 \
  --damping_decrease 0.5 \
  --residual_atol 7e-7 \
  --preconditioner_refresh_steps 1 \
  --cg_tol 1e-8 \
  --cg_maxiter 2000
```

Tests:

```bash
JAX_PLATFORMS=cpu uv run python -m unittest discover \
  -s extra/trade_growth/tests -v
```

Accepted and rejected configurations, timings, and stop decisions are in
[VALIDATION.md](VALIDATION.md).
