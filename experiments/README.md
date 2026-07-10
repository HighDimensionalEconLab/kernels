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
reproducibly in fresh processes.

**Root cause found (2026-07-10): jaxlib heap corruption, not nlls-gram.**
Instrumented reruns of the same workload produced SIGSEGV/SIGTRAP crashes
inside `libjax_common.dylib` (jax 0.10.2, macOS arm64/M4 Max) with three
distinct stacks — MLIR context creation, background-thread compilation,
and an xzone-malloc heap-integrity trap — plus silent wrong-answer runs
(identical solves returning different losses, an 8-step vs 50-step
trajectory flip). The stall was the silent-corruption face of the same
bug: a corrupted buffer makes every LM candidate rejectable without
crashing. nlls-gram's qr path is exonerated — deterministic warm-start
analysis shows matern32's first step improves 8x and rbf recovers by
damping ~1e-2 (cond(R) ~ 2e11 at the warm start explains its early
rejections, a benign and self-correcting effect). Reported upstream to
jax with a dependency-free reproducer; see the jax issue linked in the
repo discussion.

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

## `nn_many_points.py` — oversampled NN collocation + the Nystrom preconditioner

The sequential growth NN (width 32, depth 3, p = 2242 params) trained on
far more collocation points than it needs — the kernel models' 1001-node
stress test — comparing the in-model dense cholesky against cg with
nlls-gram 1.5's `nystrom_preconditioner` built from the NTK dual J J' at
the warm start (its canonical target). Same seed-0 initialization for every
solver. All runs remain underdetermined (m = 2n + 1 <= 2003 < p).

| n    | m    | cholesky                             | cg + nystrom (r=64/256, 50 CG iters)   |
| ---- | ---- | ------------------------------------ | -------------------------------------- |
| 41   | 83   | certified, 169 steps, k_err 1.1e-6, 0.15s | stalls at loss ~9e-13, k_err 1.5e-5, 2s |
| 251  | 503  | certified, 191 steps, k_err 2.1e-6, 1.7s  | stalls at loss ~2e-12, k_err 1.6e-5, 10s |
| 1001 | 2003 | certified, 186 steps, k_err 2.4e-6, 26s   | stalls at loss ~2e-12, k_err 1.6e-5, 29-32s |

Findings:

1. **Oversampling is harmless to quality.** With 24x unnecessary points the
   NN still certifies at the 1e-15 loss bar in a flat number of LM steps
   (169/191/186) with only mild error growth (1.1e-6 -> 2.4e-6). The cost
   is wall time, which follows the m x m dual (0.15s -> 26s).
2. **cg + Nystrom loses to cholesky at these sizes.** At every n it burns
   the full 400-step budget, plateaus ~3 orders above the loss bar (the
   plateau error k_err ~1.6e-5 matches the documented loss->error relation,
   so it "works", it just cannot certify), and is slower than cholesky even
   at m = 2003. Rank 64 vs 256 is indistinguishable — the bottleneck is not
   the sketch rank.
3. **The bottleneck is the inexact inner solve.** A diagnostic with a 4x CG
   budget (200 iters, r=256, n=1001) reaches loss 7.8e-15 and the best test
   error of the table (k_err 5.7e-7) — but at 124s vs cholesky's 26s and
   still shy of the bar within 400 steps. The warm-start-frozen
   preconditioner plus a finite CG budget throttles LM's terminal
   convergence; exactness is what the certification bar buys.

Verdict: for these NN duals (m <= ~2000, fast solves) dense cholesky is
simply the right solver — consistent with the prior that the NN models
don't benefit from the new solver machinery in practice. The Nystrom+cg
path only becomes interesting once forming and factoring the dense dual is
prohibitive — confirmed below.

**Big-net follow-up** (single test; `run()` takes width/depth): a 128x4 net
(p = 50050) on n = 10001 points (m = 20003, overparameterized again) under
cg + nystrom (r = 256, 100 CG iters) reaches loss 1.2e-13 with k_err
7.6e-6, c_err 1.2e-6, and a passing TVC in 100 LM steps — 346s total,
~3.5s per step. It did not hit the 1e-15 certification bar within that
step budget (the loss was still descending), but the quality already beats
the small net's documented 1e-12-loss tier. Dense cholesky at this size
would form a 3.2 GB dual from a 20003 x 50050 Jacobian at ~2e13 flops per
LM step — hours per solve — so m ~ 2e4 is past the crossover: there the
matrix-free Nystrom-preconditioned dual is the only practical option, and
it delivers near-certification accuracy in minutes.

**Staleness is the preconditioner's real bottleneck** (`run_refreshed`):
rebuilding the Nystrom sketch at the current iterate every 25 LM steps
(chunked warm-restarted solves, ~1-3s per rebuild) strictly dominates the
frozen sketch on the same 10001-point big-net problem: loss 7.4e-15 vs
1.2e-13 (16x), k_err 2.6e-6 vs 7.6e-6 (3x), in 283s vs 346s — faster
despite four extra solver compiles, because the fresher preconditioner
makes the accepted steps better. Per-chunk losses 1.2e-12 -> 4.0e-14 ->
1.5e-14 -> 7.4e-15, still descending at the 100-step budget. The
principled zero-rebuild version of this (Krylov recycling / deflated CG
across LM steps) is filed upstream as nlls_gram#15 with a self-contained
tracking benchmark.

## `dual_solve_dtype_float32.py` — where the 1.7 wide-dual flag matters

A float32 rebuild of the asset-pricing dense collocation solve (matrices
computed in float64 then cast, so every configuration sees identical
float32-representable inputs). The eps = 1e-7 unified metric puts the
dense dual at condition ~1/eps — upstream's validated target for
`dual_solve_dtype=jnp.float64`. Three configurations against the float64
truth: plain f32, f32 + wide dual, full f64. Forward accuracy (`p_0`,
train residual) and the implicit-AD path (`d p_0 / d r` through the
solve). True IEEE float32 throughout (`JAX_DEFAULT_MATMUL_PRECISION=highest`
on GPU, so Ampere TF32 does not contaminate the comparison).

CPU (Apple Silicon, n = 41):

| case | rel_err(p_0) | rel_err(d p_0/d r) | solve_s |
| --- | --- | --- | --- |
| f32 | 9.8e-8 | 5.3e-4 | 0.0018 |
| f32+wide | 2.1e-8 | **1.1e-7** | 0.0027 |

GPU (RTX 3090, driver 580, CUDA jax):

| n | case | rel_err(p_0) | max_res | rel_err(grad) | s/step |
| --- | --- | --- | --- | --- | --- |
| 41 | f32 | 2.1e-8 | 9.7e-8 | **2.3e-2** | 0.0002 |
| 41 | f32+wide | 9.8e-8 | 1.3e-7 | 1.1e-7 | 0.0004 |
| 2001 | f32 | 6.5e-5 | **3.1e-2** | **NaN** | 0.007 |
| 2001 | f32+wide | 5.7e-8 | 3.9e-7 | 1.4e-7 | 0.082 |
| 2001 | f64 | 0 | 1.1e-14 | 0 | 0.156 |

Findings:

1. **Forward solves at small n are fine in plain float32; derivatives are
   not.** On CPU the implicit gradient is wrong in the fourth digit
   (5.3e-4); on the GPU it is wrong by **2.3%** at n = 41 and **NaN** at
   n = 2001 (the undamped implicit Gram at condition 1e7 loses its f32
   Cholesky). The wide dual restores ~1e-7 everywhere.
2. **At n = 2001 on the GPU, plain float32 fails outright even forward**:
   the train residual floor is 3.1e-2 — the acceptance gates would reject
   — vs 3.9e-7 with the wide dual.
3. **The economics work on consumer GPUs.** Per accepted LM step at
   n = 2001 the wide dual costs 0.082s vs full float64's 0.156s (~2x
   cheaper: only the n·m² dual assembly and m x m factorization run wide,
   on a card with 1/64 native f64 throughput) and 12x plain f32 (which
   buys a wrong answer). Raw `solve_s` in the script output is not
   step-normalized: f64 stops at atol in 2 steps while the f32 residual
   floor keeps both f32 configurations running the full budget — use a
   dtype-appropriate atol in production.

Verdict: whenever these models move to float32 (GPU especially),
`dual_solve_dtype="float64"` (now surfaced in both kernel NLLS
`NLLSGramSettings`) should be considered mandatory: it is the difference
between rejected/NaN results and float64-grade accuracy, at half the cost
of solving fully in float64 on consumer hardware.
