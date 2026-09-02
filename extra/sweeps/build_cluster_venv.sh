#!/bin/bash
# One-time, login-node build of a persistent Alliance-wheelhouse venv.
#   Usage: extra/sweeps/build_cluster_venv.sh [cpu|gpu]
#     cpu (default) -> .venv-cpu : jax on CPU, no cuda module/plugin
#     gpu           -> .venv-gpu : + module load cuda/cudnn + jax_cuda12_plugin/pjrt
#                      (runs on a GPU node; falls back to CPU elsewhere)
#
# WHEELHOUSE RULE (Alliance): install EVERYTHING from the wheelhouse with --no-index --
# never the public PyPI registry when a wheel exists. flax, nlls-gram and tinydiffeq
# are installed LAST with --no-deps so pip cannot re-resolve jax/jaxlib from PyPI.
# Their runtime dependencies are installed from the wheelhouse first.
#
# pyarrow is the exception to "everything from the wheelhouse": the wheelhouse ships only
# a DUMMY pyarrow wheel that errors on install. The real pyarrow lives in the `arrow`
# MODULE, which MUST be `module load`ed BEFORE the venv is activated -- at BOTH build and
# run time (the sbatch loads the same modules before activating).
#
# The sweep path only needs the recursive i(k) solver's imports (jax, flax, scipy,
# pandas, nlls-gram, tinydiffeq) + wandb/jsonargparse -- no UNO or plotting stack.
#
# cuda/cudnn/arrow versions are Fir's (2026-06) -- adjust per cluster. On a login node
# jax.devices() shows cpu even for .venv-gpu (no GPU there).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-cpu}"

WHEELS="jax jaxlib flax jsonargparse numpy pandas scipy wandb"

case "$MODE" in
  cpu) module load gcc arrow/24.0.0 python/3.12;                           VENV=.venv-cpu ;;
  gpu) module load gcc arrow/24.0.0 python/3.12 cuda/12.9 cudnn/9.13.1.26; VENV=.venv-gpu
       WHEELS="$WHEELS jax_cuda12_plugin jax_cuda12_pjrt" ;;
  *)   echo "usage: extra/sweeps/build_cluster_venv.sh [cpu|gpu]" >&2; exit 1 ;;
esac

echo "=== building ${VENV} (${MODE}) ==="
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index $WHEELS
# The wheelhouse flax lags its own jax (0.12.6 imports jax.core.Effect, gone in
# jax 0.11); the wheelhouse install above supplies flax's deps, PyPI the release
# whose jax floor matches the wheelhouse jax (0.12.9 for jax 0.11.1).
pip install --no-deps 'flax>=0.12.9' 'nlls-gram>=2.10.0' 'tinydiffeq>=2.6.1'
python -c "import jax, flax, jsonargparse, pandas, pyarrow, scipy, nlls_gram, tinydiffeq, wandb; print('venv ok | jax', jax.__version__, '| devices', jax.devices())"
echo "BUILD_DONE (${VENV})"
