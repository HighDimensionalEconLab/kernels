# Cluster HPO record — recursive i(k) solver

## Sweep

- **Sweep**: `highdimensionaleconlab/kernels/9j88zxne`
  (https://wandb.ai/highdimensionaleconlab/kernels/sweeps/9j88zxne), bayes,
  minimize median-seed `train_time_success` (time to a certified solve; NaN
  when not accepted), 11 seeds/trial, `min_successful_seeds=4`, hard 2 s
  budget, single attempt. Search space: `recursive_i_hpo.yaml`.
- **Acceptance calibration (done first)**: scipy `solve_bvp`'s default
  `tol=1e-3` was the entire 3.7e-5 reported "accuracy floor"; the benchmark
  now solves at `tol=1e-10` (true policy error ~1.5–3e-6). The sweep fixes
  `train_loss_threshold=1e-14` so an accepted solve certifies k-error ~2e-6,
  and caps swept rtol/atol at 1e-6 where integration error is non-binding.
- **Cluster**: Fir, `sbatch --account=def-jlperla_cpu --cpus-per-task=4
  --mem=32G --time=01:00:00 --array=1-50`, `VENV=.venv-cpu`, unbounded agents
  (ended TIMEOUT as expected). Jobs: 47575834 (abandoned, see below),
  47576830 (the real run).
- **Yield**: 3,868 trials, 3,157 certified (82%), 1,726 with 11/11 seeds.

## Winning configuration (recommended)

`tts=0.271s`, 11/11 seeds, k-err 1.27e-6, 3 epochs — 4x faster than the
pre-sweep defaults (1.1 s) at slightly better accuracy:

```
nn_set.width=8  nn_set.depth=2  nn_set.hidden_activation=gelu
lm_set.init_damping=3.26e-05  lm_set.damping_decrease=0.482
lm_set.damping_increase=2.71  lm_set.geodesic_acceleration=true
lm_set.geodesic_acceptance_ratio=0.541
data_set.rtol=6.28e-07  data_set.atol=2.23e-07
data_set.adaptive_steps=32  data_set.dt=1.0
stop_set.train_loss_threshold=1e-14
```

(The raw #1 at 0.259 s used dt=2, whose coarse RK4 *evaluation* grid inflates
the measured k-error to 2.1e-5 — a measurement artifact, not a worse policy;
the dt=1 runner-up is the clean pick.)

## Marginal effects (directional; bayes oversamples good regions)

median certified tts | certification rate:

- **geodesic_acceleration**: on 0.48s|91% vs off 0.79s|73% — largest single effect.
- **width**: 8 → 0.55s|84%, monotone worse to 128 → 1.83s|20% (capacity hurts
  under a hard budget).
- **activation**: gelu 0.54s|93% > silu 86% > tanh 81% > softplus 0.82s|45%.
- **adaptive_steps**: 12 → 0% and 16 → 1% certification (budget cannot reach
  the horizon: structurally infeasible); 32 is the sweet spot (88%).
- **atol**: below ~6e-9 certification collapses to 28% (over-tight absolute
  tolerance burns the budget); [2e-7, 1e-6] best (85%).
- **train_steps_per_epoch**: 2 → 46% (per-epoch certificate overhead), 10 → 84%.
- **Flat / insensitive**: init_damping (1e-8..1e-2!), damping_increase,
  damping_decrease, geodesic_acceptance_ratio, dt in {1, 2}.

## Operational notes

- Job 47575834 failed structurally because the first jit compile (~2.8 s on Fir
  cores vs ~0.6 s locally) entered the old 2 s timed budget. That historical
  sweep used an explicit warm-up; current replication instead enables the
  persistent cache and reports medians across repeated solves.
- The sbatch exports `JAX_COMPILATION_CACHE_DIR` (shared persistent XLA cache
  on scratch) to amortize per-config compiles across seeds/trials/agents.
- 12 of 50 array tasks died at startup (3 s) with a partially-visible `yaml`
  package import on the parallel FS — a launch race; the other 38 agents did
  all the work. If it recurs, stagger task start or resubmit the failed ids.
