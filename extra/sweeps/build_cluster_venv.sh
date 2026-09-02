#!/bin/bash
# One-time, login-node build of a persistent Alliance-wheelhouse venv.
#   Usage: extra/sweeps/build_cluster_venv.sh [cpu|gpu]
#     cpu (default) -> .venv-cpu : jax on CPU, no cuda module/plugin
#     gpu           -> .venv-gpu : + module load cuda/cudnn + jax_cuda12_plugin/pjrt
#                      (runs on a GPU node; falls back to CPU elsewhere)
#
# WHEELHOUSE RULE (Alliance): install EVERYTHING from the wheelhouse with --no-index --
# never the public PyPI registry when a wheel exists. flax, nlls-gram and tinydiffeq
# are installed LAST with --no-deps so pip cannot re-resolve jax/jaxlib from PyPI
# (the site pip constraints pin jax/jaxlib and the CUDA plugin to wheelhouse
# versions anyway). Their runtime dependencies are installed from the wheelhouse first.
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

WHEELS="flax jsonargparse numpy pandas scipy wandb"

# The wheelhouse CUDA plugin lags its jaxlib (plugin 0.10.2, jaxlib 0.11.1),
# so the GPU venv takes the wheelhouse's complete 0.10.2 jax stack; each venv
# then gets the flax release whose jax floor matches its jax.
case "$MODE" in
  cpu) module load gcc arrow/24.0.0 python/3.12;                           VENV=.venv-cpu
       JAX="jax jaxlib"; FLAX="flax>=0.12.9" ;;
  gpu) module load gcc arrow/24.0.0 python/3.12 cuda/12.9 cudnn/9.13.1.26; VENV=.venv-gpu
       JAX="jax==0.10.2 jaxlib==0.10.2 jax_cuda12_plugin==0.10.2 jax_cuda12_pjrt==0.10.2"
       FLAX="flax==0.12.7" ;;
  *)   echo "usage: extra/sweeps/build_cluster_venv.sh [cpu|gpu]" >&2; exit 1 ;;
esac

echo "=== building ${VENV} (${MODE}) ==="
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index $JAX $WHEELS
# The wheelhouse flax (0.12.6) cannot import against jax 0.11; the wheelhouse
# install above supplies flax's deps, PyPI the matching flax release.
pip install --no-deps "$FLAX" 'nlls-gram>=2.10.0' 'tinydiffeq>=2.6.1'
python -c "import jax, flax, jsonargparse, pandas, pyarrow, scipy, nlls_gram, tinydiffeq, wandb; print('venv ok | jax', jax.__version__, '| devices', jax.devices())"
echo "BUILD_DONE (${VENV})"
