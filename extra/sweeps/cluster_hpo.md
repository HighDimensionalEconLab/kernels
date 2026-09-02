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
| open | 8.4 s (9x) | 1.2e-5, 2.2e-4 / 8.9e-5 (default 3e-6) | width 8, depth 2, silu, 2 x 100 steps, no shock homotopy, 8 trajectories, 97 points, euler_maruyama |
| open | 9.2 s (8x) | 4.0e-6, 3.9e-5 / 3.8e-5 | width 8, depth 2, silu, 2 x 250, no shock homotopy, 8 trajectories, 49 points, euler_maruyama |

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
  faster than 3 (15 vs 18 s); training the true model from the first
  epoch 14 s vs a reduced-volatility first epoch 18 s (that shock homotopy
  has since been removed from the driver); `max_steps` 100 13 s vs 250
  17 s; `train_points` 33 to 49 (14 s) vs
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

Adopted defaults (2026-09-02): closed width 8, depth 3, silu, 4 x 125
steps, `euler_maruyama`; open width 8, depth 2, silu, 5 x 100 steps, 49
train points, `euler_maruyama`. Trajectory counts and
loss bars unchanged. Locally, five seeds: closed held-out Euler mean
7.3e-8 median / 2.1e-7 worst at 6.8 s (was 3.7e-8 / 4.6e-8 at 82 to
165 s); open 5.9e-6 / 1.0e-5 at 1.2 s (was 4.2e-6 / 5.0e-6 at 23 to
34 s). The kernel consumers distilled from the new artifacts reproduce
their previous held-out errors to three digits.

Acceptance bars (2026-09-02): each recursive driver's held-out bar is 3
x the worst accepted seed over its experiments under the defaults above,
rounded up to one significant figure (10 x wherever the 3 x arm
rejected; none did). The mean bars in the stochastic open economy apply
per output, so the worst is the larger of the c and q means.

| driver | metric | worst | bar |
| --- | --- | --- | --- |
| neoclassical NN | test mean / max | 4.2e-7 / 1.1e-5 | 2e-6 / 4e-5 |
| closed NN | test mean / max | 2.1e-7 / 6.0e-5 | 7e-7 / 2e-4 |
| open NN | test mean / max / diffusion max | 1.4e-5 / 5.8e-4 / 9.6e-4 | 5e-5 / 2e-3 / 3e-3 |
| deterministic open NN | test max | 1.6e-6 | 5e-6 |
| neoclassical kernel | test mean / max | 1.6e-5 / 1.3e-4 | 5e-5 / 4e-4 |
| closed kernel | test mean / max | 5.1e-5 / 3.2e-3 | 2e-4 / 1e-2 |
| open kernel | test mean / max | 1.8e-3 / 1.2e-1 | 6e-3 / 4e-1 |

Closed NN `test_only` arm (`gate_train=False`, five seeds): at these
bars 5/5 accept, epoch early stopping ends three seeds after one epoch
(125 steps, 1.6 s) and the rest by 375 steps, held-out mean 3.8e-7
median / 5.6e-7 worst, max 8.1e-5 / 1.5e-4; the gated arm runs 375 to
500 steps (4.7 to 7.5 s) for 7.3e-8 / 2.1e-7.

## Stochastic open-economy kernel (local, 2026-09-02)

Diagnosis: the kernel's held-out error tracked the distance to the
nearest design state, not the fit (tightening the stop and ridge cut the
design error 13x and the held-out error 2x), because four shock paths
from one initial condition leave the held-out paths ~0.8 normalized
units outside the design box along log z and the rate gap, and half of
a 60-year design sat past the capital transition. The decoupled
`reference_ic_set` lets the kernel choose its own design density
without a matching NN artifact, and the horizons drop to train 40 /
test 50 (transition dynamics only).

Sweep (`stochastic_open_economy_kernel_local_hpo.py`, three data seeds
per config): 277 random configs over trajectory count {2, 4, 8, 16},
train points {11, 17, 21, 33, 41} (designs above 400 centers skipped),
ARD RBF / Matern-5/2, free constant, linear mean, log-uniform
lengthscales (shared 0.7 to 5, log k as a 0.3x to 3x ratio, assets 0.1
to 5), amplitude, metric epsilon, ridge, atol; then two rounds of 60
log-normal jitters (sd log 1.5) around the top three to five. Marginal
effects: 11 points never accept and 21 (dt 2) beats every other
density; 16 paths beat fewer; ARD RBF beats Matern; the linear mean and
lengthscales of 3 to 7 on the log z and rate axes do the extrapolation.

Adopted (16 x 21, ARD RBF, linear mean, no free constant, lengthscales
6.7 / 3.45 / 0.62 / 6.7, amplitude 1.2, epsilon 4.3e-6, ridge 2.8e-12,
atol 2.2e-4, gtol 1e-9), five data seeds, all accepted:

| | held-out mean (c, q) | held-out max | capital error mean / max | solve |
| --- | --- | --- | --- | --- |
| adopted | 7.4e-5 worst, 5.1e-5 median | 3.7e-3 | 0.055% / 0.56% | 0.8 to 2.5 s |
| previous defaults (4 x 49, lengthscale 1.69) | 1.8e-3 | 1.2e-1 | 1.7% / 6.3% | 0.7 s |
| adopted geometry, 32 paths | 6.2e-5 | 6.0e-3 | 0.037% / 0.29% | 8.9 s |

Acceptance bars (2026-09-02, transition horizons): 3 x the worst
accepted seed under the defaults above, rounded up to one significant
figure. The neoclassical kernel moves to lengthscale 1.1 with the
60-year path (0.6 stalls the LM with the free constant, and without it
the multi-IC g = 0 design stalls instead); the closed kernel keeps its
geometry (its replication seed accepts; two of five data seeds reject
on the max bar, a coverage limit of the single-IC design that no
density or lengthscale in {11..41 points, 8..32 paths, z lengthscale
0.4..1.0} moves).

| driver | metric | worst | bar |
| --- | --- | --- | --- |
| neoclassical NN | test mean / max | 9.1e-7 / 2.9e-5 | 3e-6 / 9e-5 |
| closed NN | test mean / max | 9.2e-8 / 3.2e-5 | 3e-7 / 1e-4 |
| open NN | test mean / max / diffusion max | 1.4e-5 / 3.1e-4 / 9.7e-4 | 5e-5 / 1e-3 / 3e-3 |
| neoclassical kernel | test mean / max | 7.5e-6 / 8.4e-5 | 3e-5 / 3e-4 |
| closed kernel | test mean / max | 7.7e-5 / 5.2e-3 | 3e-4 / 2e-2 |
| open kernel | test mean / max | 7.4e-5 / 3.7e-3 | 3e-4 / 2e-2 |
