# Trade-growth DAE validation

Validation date: 2026-07-26. All production runs used CPU float64, N=2 or N=3
heterogeneous economies, seed-0 random initial conditions, a Matérn-5/2
kernel, 48 collocation dates over a 100-year horizon, and 401 validation dates.
No country count above three was attempted after the iterative scaling test
failed the preset performance gate.

Acceptance requires solver convergence, maximum train residual at most
`1e-5`, maximum validation residual at most `5e-3`, and asset clearing,
common-rate consistency, and world trade balance at most `5e-5`.

## Metric and algorithm screening

The first dense MetricLM sweep used the original path controls, disabled
homotopy, and varied only epsilon. All three cases were rejected. Epsilon did
not cure the selected path's off-grid rate inconsistency.

| Epsilon | Steps | Solve (s) | Max train | Max validation | Max rate gap | Decision |
|---:|---:|---:|---:|---:|---:|---|
| `1e-6` | 100 | 8.61 | `7.21e-9` | `3.43e-4` | `1.11e-4` | rejected: max steps and rate gap |
| `1e-7` | 100 | 6.64 | `5.37e-8` | `3.40e-4` | `1.11e-4` | rejected: max steps and rate gap |
| `1e-8` | 100 | 6.47 | `8.18e-7` | `3.31e-4` | `1.11e-4` | rejected: max steps and rate gap |

Initial damping, damping decrease, geodesic acceleration, and the residual
stop were then varied one factor at a time. The important findings were:

- initial damping `1e-9` fixed the rate gap but still needed an explicit
  residual stop;
- damping decrease `0.5` selected a comparably accurate path with fewer
  steps;
- residual tolerance `1e-6` converged quickly but left the train residual too
  close to the acceptance boundary;
- residual tolerance `1e-10` exhausted 100 steps;
- `1e-8` for N=2 and `2e-8` for N=3 were retained dense settings.

## Dense production results

| N | Optimizer | Initialization | Steps | Solve (s) | Max train | Max validation | Max asset | Max rate | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 2 | MetricLM | direct | 64 | 5.66 | `7.47e-8` | `1.42e-4` | `6.52e-8` | `1.60e-5` | accepted |
| 3 | MetricLM | direct | 79 | 13.44 | `2.07e-7` | `1.48e-4` | `5.21e-7` | `5.17e-6` | accepted |
| 2 | RidgeLM | direct | 100 | 9.29 | `2.66e-7` | `6.77e-3` | `1.57e-5` | `1.01e-3` | rejected |
| 3 | RidgeLM | direct | 100 | 17.31 | `2.58e-7` | `4.60e-3` | `1.24e-5` | `9.74e-4` | rejected |
| 2 | RidgeLM | asset homotopy | 8 final | 25.49 | `2.87e-8` | `1.19e-4` | `1.88e-7` | `3.60e-6` | accepted |
| 3 | RidgeLM | asset homotopy | 8 final | 39.15 | `5.15e-8` | `1.33e-4` | `4.01e-7` | `3.61e-6` | accepted |

MetricLM is substantially faster and avoids homotopy on these two draws. It
is not promoted to the default because the independent dense solver parity
gate failed: N=2 Cholesky and QR both passed all economic checks, but their
full-horizon paths differed by up to `7.64e-5`. This is a root-selection
difference in a path-dependent method, not a residual-equation failure.

## Preconditioner attempts

The shifted coefficient maps were audited as
`K_tilde F^{-1}` and `K F^{-1}` with `F'F=K+epsilon I`; the latter must not be
replaced by `F'`. Initial-value columns were scaled by `1/sqrt(epsilon)`.

| Attempt | Structure | Outcome | Decision |
|---|---|---|---|
| Country modes | Parameter-space aggregate and contrast blocks | N=2 accepted in 39.03 s, 26 steps, max train `9.24e-6` | correct fallback, too slow |
| Within-date blocks | All economic coupling at each date plus exact initial-value Woodbury update | N=2 small-grid max train about `2.13e-2` after 100 steps | rejected alone; missed temporal stiffness |
| Equation-time blocks | Full temporal block for each repeated economic equation | N=2 small-grid max train about `3.96e-3` after 100 steps | rejected alone; missed cross-equation coupling |
| Additive hybrid | Equal-weight sum of the preceding inverse actions, refreshed after accepted steps | N=2 and N=3 production runs accepted | retained correctness fallback |
| Multiplicative hybrid | Symmetric Schwarz composition using the exact frozen dual operator | excellent small-grid result; N=2 production max train `4.19e-5` and asset gap `3.77e-4` in 43.64 s | rejected |
| Loose-to-tight forcing | CG tolerance `1e-2` to `1e-8`, budget 60 to 500 | N=2 max train `2.90e-5` after 100 steps | rejected; early inexact steps changed the MetricLM path |

The retained hybrid keeps two complementary pieces of economic structure.
Within-date blocks retain the numeraire, world asset balance, common rate, and
bilateral trade interactions. Equation-time blocks retain the smooth RKHS
covariance across the full horizon. Either piece alone is insufficient.

## Iterative production results and stop decision

| N | Solver | Steps | Solve (s) | Max train | Max validation | Max asset | Max rate | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2 | balanced parameter CG | 26 | 39.03 | `9.24e-6` | `1.42e-4` | `2.64e-5` | `1.37e-5` | accepted |
| 2 | hybrid GramCG | 47 | 31.14 | `8.87e-6` | `1.43e-4` | `2.11e-5` | `1.37e-5` | accepted |
| 3 | hybrid GramCG | 95 | 94.23 | `8.90e-6` | `1.48e-4` | `5.70e-6` | `4.02e-6` | accepted |

The hybrid is about five times the N=2 dense solve and seven times the N=3
dense solve. Its solve time grows by about three times from N=2 to N=3, while
dense grows by about 2.4 times. This fails the performance gate before N=5,
so the requested N=5, N=10, and larger experiments were not run.

## AD and algebra checks

The tests verify:

- the exact shifted repeated metric and epsilon-scaled free block;
- dense local block assembly against a materialized shifted dual Jacobian;
- SPD preconditioner actions and improved damped spectrum;
- dense Cholesky, QR, and GramCG path agreement on small grids;
- finite forward-mode derivatives and forward/reverse directional duality;
- all structural parameters, bilateral trade costs, `k_0`, and `a_0` remain
  leaves of `p`.

The free initial values are not penalized under RidgeLM. Under MetricLM they
receive only the epsilon damping metric needed for an invertible correction
geometry; the residual objective itself remains unmodified.
