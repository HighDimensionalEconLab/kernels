#!/bin/bash
# One-time, login-node build of a persistent Alliance-wheelhouse venv.
#   Usage: sweeps/build_cluster_venv.sh [cpu|gpu]   (run from anywhere in the repo)
#     cpu (default) -> .venv-cpu : jax on CPU, no cuda module/plugin
#     gpu           -> .venv-gpu : + module load cuda/cudnn + jax_cuda12_plugin/pjrt
#                      (runs on a GPU node; falls back to CPU elsewhere)
#
# WHEELHOUSE RULE (Alliance): install EVERYTHING from the wheelhouse with --no-index --
# never the public PyPI registry when a wheel exists. The ONLY package not in the
# wheelhouse is nlls-gram: install it LAST with --no-deps so pip cannot pull its deps
# (or re-resolve jax/jaxlib) from PyPI. Its runtime deps (jaxtyping wadler-lindig lineax
# equinox) are installed from the wheelhouse FIRST, so --no-deps leaves a consistent env.
#
# pyarrow is the exception to "everything from the wheelhouse": the wheelhouse ships only
# a DUMMY pyarrow wheel that errors on install. The real pyarrow lives in the `arrow`
# MODULE, which MUST be `module load`ed BEFORE the venv is activated -- at BOTH build and
# run time (the sbatch loads the same modules before activating).
#
# The sweep path only needs the recursive i(k) solver's imports (jax, flax, scipy,
# pandas, nlls-gram) + wandb/jsonargparse -- no unopy/quadax/matplotlib on the cluster.
#
# cuda/cudnn/arrow versions are Fir's (2026-06) -- adjust per cluster. On a login node
# jax.devices() shows cpu even for .venv-gpu (no GPU there).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this file lives in sweeps/)
MODE="${1:-cpu}"

WHEELS="jax jaxlib flax jsonargparse numpy pandas scipy wandb"
GRAM_DEPS="jaxtyping wadler-lindig lineax equinox"   # nlls-gram runtime deps, all in wheelhouse

case "$MODE" in
  cpu) module load gcc arrow/24.0.0 python/3.12;                           VENV=.venv-cpu ;;
  gpu) module load gcc arrow/24.0.0 python/3.12 cuda/12.9 cudnn/9.13.1.26; VENV=.venv-gpu
       WHEELS="$WHEELS jax_cuda12_plugin jax_cuda12_pjrt" ;;
  *)   echo "usage: sweeps/build_cluster_venv.sh [cpu|gpu]" >&2; exit 1 ;;
esac

echo "=== building ${VENV} (${MODE}) ==="
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index $WHEELS
pip install --no-index $GRAM_DEPS
pip install --no-deps 'nlls-gram>=0.5.1'
python -c "import jax, flax, jsonargparse, pandas, pyarrow, scipy, nlls_gram, wandb; print('venv ok | jax', jax.__version__, '| devices', jax.devices())"
echo "BUILD_DONE (${VENV})"
