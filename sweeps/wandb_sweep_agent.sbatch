#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=sweeps/logs/sweep_%A_%a.out
# One wandb-agent worker per array task. Resources + --array come from the sbatch CLI
# flags (see sweeps/README.md); behaviour comes from env vars passed via --export:
#   PROJECT_DIR   absolute path to the synced repo (where the venv + the sweep yaml live)
#   SWEEP_ID      entity/kernels/<sweep_id> returned by `wandb sweep`
#   VENV          optional: venv dir to activate (default .venv-cpu -- this repo's sweeps
#                 are CPU-only; the solver's MLPs are tiny).
#   WANDB_MODE    online (default) on internet-enabled compute nodes (Fir verified)
#   AGENT_COUNT   optional: runs this worker pulls before exiting. Unset -> unbounded
#                 (runs until the walltime kills the job; ends TIMEOUT, which is expected).
# The arrow module provides pyarrow (the wheelhouse wheel is a dummy); it must be loaded
# before the venv activates, matching build_cluster_venv.sh.
set -euo pipefail
cd "${PROJECT_DIR}"
module load gcc arrow/24.0.0 python/3.12
source "${VENV:-.venv-cpu}/bin/activate"
export WANDB_MODE="${WANDB_MODE:-online}"
# Persistent XLA compilation cache: amortizes the ~2.8s per-config compiles
# across seeds, trials, and array tasks (shared over the parallel FS).
export JAX_COMPILATION_CACHE_DIR="${PROJECT_DIR}/.jax_cache"
echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID:-NA} sweep=${SWEEP_ID} count=${AGENT_COUNT:-unlimited} mode=${WANDB_MODE}"
python -c "import jax; print('jax devices:', jax.devices())" || true
if [ -n "${AGENT_COUNT:-}" ]; then
  wandb agent --count "${AGENT_COUNT}" "${SWEEP_ID}"
else
  wandb agent "${SWEEP_ID}"
fi
