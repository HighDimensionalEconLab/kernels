# NLLS solver experiments

One-off measurement scripts behind the NLLS solver decisions (2026-07-09,
CPU, float64). Not part of the replication pipeline or the test suite. Run
from the repo root as modules, e.g. `uv run python -m experiments.qr_sweep`.

## `qr_sweep.py` — Cholesky vs QR across the dense configurations

Both models x {matern12, matern32, matern52, rbf} x n in {41, 101, 501,
1001}, explicit `linear_solver="cholesky"` vs `"qr"`. Headline rows:

| model  | kernel   | n    | cholesky                  | qr                          |
| ------ | -------- | ---- | ------------------------- | --------------------------- |
| growth | matern32 | 101  | 50-step cap, res 1.3e-7   | 11 steps, res 3.6e-10       |
| growth | matern52 | 101  | err 3.1e-5 (T=40)         | err 3.3e-7 (~100x lower)    |
| growth | matern52 | 501  | err 1.2e-6, rejected      | err 7.3e-9 (165x), rejected |
| growth | matern32 | 501  | err 1.4e-7                | err 4.3e-8, accepted        |

Everywhere else (all n = 41 runs, every asset-pricing dense case) the two
solvers agree to ~1e-13 and cholesky is the same speed or faster; qr costs
2-6x more time at n >= 101 (growth n = 501: 8.2s vs 1.3s). rbf evidence is
mixed (qr worse at T = 40, better at T = 50).

**Decision (commit 431b430):** growth `linear_solver="auto"` picks qr for
matern32/matern52 on the dense branch at n >= `DENSE_QR_MIN_POINTS` (101);
rbf and asset pricing keep cholesky.

## `qr_horizon_followup.py` — the n = 101 boundary at both horizons

The qr wins above hold at T = 40 and T = 50. One caveat observed while
sweeping: growth matern32/rbf (n = 101, T = 50) under qr once stalled at the
warm start for all 50 steps inside a long-lived process (train residual
0.137 — the kernel-independent warm-start residual) yet converges
reproducibly in fresh processes. Likely a knife-edge first-step
accept/reject at `init_damping = 1e-8`; a reason to keep the auto-qr
boundary conservative and possibly report upstream to nlls-gram.

## `eps_sensitivity.py` — is the UNO error edge the metric shift?

No. Sweeping `metric_eps` from 1e-7 to 1e-10 on the growth baseline leaves
the matern12 gap at ~15% (3.9e-4 vs UNO's 3.39e-4) and degrades
matern32/matern52 convergence. The residual gap is an LM-path selection
difference on the interpolation manifold (UNO's KKT system lands exactly on
the min-seminorm point), inside the O(dt^2) grid-selection bias at n = 41.
`metric_eps = 1e-7` stays the right default.

## `uno_vs_nlls.py` — the replication configurations, side by side

Every replication experiment with an NLLS port (10 growth + 1 asset), UNO
vs NLLS with auto settings. Snapshot in `uno_vs_nlls_results.json`.

| experiment       | UNO err | NLLS err | winner      | NLLS solve speedup |
| ---------------- | ------- | -------- | ----------- | ------------------ |
| baseline         | 3.39e-4 | 3.93e-4  | UNO by 16%  | 26x                |
| sparse           | 2.93e-3 | 3.34e-3  | UNO by 14%  | 13x                |
| far_steady_state | 1.53e-2 | 1.06e-2  | NLLS by 31% | 11x                |
| nu15_ell10       | 2.37e-4 | 3.01e-4  | UNO by 27%  | 24x                |
| nu25_ell10       | 5.14e-4 | 5.14e-4  | tie         | 31x                |
| nu05_ell2        | 8.11e-4 | 8.39e-4  | UNO by 3%   | 18x                |
| nu05_ell20       | 3.71e-4 | 4.11e-4  | UNO by 11%  | 24x                |
| rbf_ell10/ell20  | reject  | reject   | tie ("--")  | --                 |
| rbf_ell2         | 1.96e-3 | 2.54e-5  | NLLS by 77x | 88x                |
| asset baseline   | 1.14e-3 | 1.14e-3  | tie         | 11x                |

**Verdict:** NLLS strictly dominates on solve time (11-88x) and matches the
accept/reject pattern exactly, wins decisively on the numerically hard
cases (rbf_ell2, far_steady_state), but UNO keeps a 3-27% error edge on
five Matern rows — same accuracy class, not strict domination. The
concave-convex growth, optimal-advertising, and human-capital experiments
have no NLLS ports. Decision: replication stays on UNO for now; revisit
switching later.
