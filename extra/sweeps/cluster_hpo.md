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

## Solve-time sweeps, stochastic recursive NNs (Fir, 2026-09-01)

Objective: median-seed `solve_time_gated` (solve time when the driver
accepts the solution and the wrapper's ex-post gates hold), 3 seeds per
trial, `min_successful_seeds` 2, per-seed cap 1.3x the default config's
Fir total fit time (calibration job 57782142, 4 cores: closed 624 s, open
72 s). Venv: wheelhouse jax/jaxlib 0.11.1, flax 0.12.9 from PyPI
`--no-deps`, nlls-gram 2.10.0, tinydiffeq 2.6.1.

| sweep | config | id | array | resources |
|---|---|---|---|---|
| closed economy | `stochastic_growth_nn_time_hpo.yaml` | `highdimensionaleconlab/kernels/qlbyvp59` | 57783361 (1-10) | 4 cores, 16 GB, 6 h |
| open economy | `stochastic_open_economy_nn_time_hpo.yaml` | `highdimensionaleconlab/kernels/623si085` | 57783362 (1-10) | 4 cores, 16 GB, 3 h |

    sbatch --account=def-jlperla_cpu --cpus-per-task=4 --mem=16G --time=06:00:00 --array=1-10 \
      --export=ALL,PROJECT_DIR=/scratch/$USER/kernel_econ_alignment,SWEEP_ID=highdimensionaleconlab/kernels/qlbyvp59,VENV=.venv-cpu,WANDB_MODE=online \
      extra/sweeps/wandb_sweep_agent.sbatch
    # open economy: --time=03:00:00, SWEEP_ID=highdimensionaleconlab/kernels/623si085

Unbounded agents (no `AGENT_COUNT`); the arrays end as `TIMEOUT`.

Stopped after 35 minutes (`wandb sweep --stop`, `scancel`): closed 106
trials / 33 accepted, open 188 / 174. Per-seed timeouts fired 14 times.

### Results

Times are Fir 4-core wall clock, median seed of 3; the defaults measured
624 s (closed) and 72 s (open) on the same allocation.

| sweep | best | held-out Euler (mean, max) | config vs default |
|---|---|---|---|
| closed | 11.6 s (54x) | 5.6e-8, 2.2e-5 (default 3.7e-8, 1.8e-5) | width 8, depth 2, silu, 4 x 250 steps, loss bar 5.6e-12, 16 trajectories, 65 points, euler_maruyama |
| closed | 12.1 s (52x) | 5.4e-8, 3.9e-5 | width 8, depth 3, silu, 4 x 125, bar 1.2e-12, 33 points, euler_maruyama |
| open | 8.4 s (9x) | 1.2e-5, 2.2e-4 / 8.9e-5 (default 3e-6) | width 8, depth 2, silu, 2 x 100 steps, one ladder epoch, 8 trajectories, 97 points, euler_maruyama |
| open | 9.2 s (8x) | 4.0e-6, 3.9e-5 / 3.8e-5 | width 8, depth 2, silu, 2 x 250, one ladder epoch, 8 trajectories, 49 points, euler_maruyama |

Marginal effects (accepted share, median accepted time; a Bayes sweep
oversamples good regions, so directional only):

- Closed: width 8 22/58 at 28 s, 16 11/33 at 102 s, 32 0/15; trajectories
  16 32/70, 8 1/23, 4 0/13 (the dispersed-IC design needs the batch);
  `max_steps` 25 and 50 never accepted, 125 6/25, 250 27/49;
  `num_epochs` 2 never, 4 best (18/32); `train_points` 17 never, 33 and
  65 alike; `euler_maruyama` 33/81, `sra1` 0/25 (confounded with the
  early width-32 probes); gelu, silu, tanh all accept, silu fastest.
  Accepted `train_loss_threshold` spans 1e-13 to 5e-11 (median 4e-12);
  the LM damping quartet is flat.
- Open: width 8 167/174 at 16 s vs 16 7/12 at 42 s; depth 2 slightly
  faster than 3 (15 vs 18 s); one ladder epoch 14 s vs two 18 s;
  `max_steps` 100 13 s vs 250 17 s; `train_points` 33 to 49 (14 s) vs
  97 (19 s); 8 trajectories accept as often as 4, 2 do not (1/7);
  `euler_maruyama` 130/136 vs `sra1` 44/52; activations and the LM
  quartet flat.

Adopting a sweep value: `ic_set.num_train_trajectories` enters the policy
artifact's problem hash, so the kernel consumer's default must follow.

### float32 GPU smoke (Fir, job 57785088, one `1g.10gb` H100 MIG slice)

`.venv-gpu` is the wheelhouse jax 0.10.2 stack with flax 0.12.7. Both
drivers at their defaults, `use_float64 false`, thresholds from the local
float32 ladder (closed `train_loss_threshold` 1e-11, open 1e-9; the
float32 fresh-batch floors sit near 1e-12 and 1e-11):

| driver | steps | solve time | s / step | held-out Euler mean | float64 default |
|---|---|---|---|---|---|
| closed | 750 (accept hook, final loss 6.4e-13) | 136 s | 0.18 | 1.9e-6 | 3.7e-8 |
| open | 750 | 80 s | 0.11 | 1.6e-5 | 3e-6 |

Local CPU float32 at the same thresholds: closed 0.24 s/step (converged at
the first boundary), open 0.04 s/step. The networks are too small to fill
a GPU; the slice only helps the closed economy's Jacobian, and the open
economy is launch-bound and slower than a laptop core.
