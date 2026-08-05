#!/bin/bash
# One-time, login-node build of a persistent Alliance-wheelhouse venv for the
# replication pipeline (generate_all_results.sh).
#   Usage: ./build_replication_venv.sh [cpu|gpu]
#     cpu (default) -> .venv-cpu : jax on CPU, no cuda module/plugin
#     gpu           -> .venv-gpu : + module load cuda/cudnn + jax cuda plugin
#                      (runs on a GPU node; falls back to CPU elsewhere)
#
# WHEELHOUSE RULE (Alliance): install EVERYTHING from the wheelhouse with
# --no-index -- never the public PyPI registry when a wheel exists. nlls-gram
# and tinydiffeq are installed LAST with --no-deps so pip cannot re-resolve
# jax/jaxlib from PyPI. Their runtime dependencies are installed from the
# wheelhouse first.
#
# pyarrow is the exception to "everything from the wheelhouse": the wheelhouse
# ships only a DUMMY pyarrow wheel that errors on install. The real pyarrow
# lives in the `arrow` MODULE, which MUST be `module load`ed BEFORE the venv
# is activated -- at BOTH build and run time.
#
# The replication path needs the solver stack (jax, flax, scipy, pandas,
# nlls-gram, tinydiffeq, jsonargparse) plus matplotlib for the figure scripts.
# No unopy: the float32 pipeline skips the uno/IPOPT baselines.
#
# Module and wheel versions below are Vulcan's -- confirm with `module spider`
# and `avail_wheels jax jax_cuda12_plugin jax_cuda12_pjrt` per cluster.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-cpu}"

WHEELS="jax jaxlib flax jsonargparse numpy pandas scipy matplotlib"
GRAM_DEPS="jaxtyping wadler-lindig lineax equinox"   # nlls-gram runtime deps

case "$MODE" in
  cpu) module load gcc arrow python/3.12;                     VENV=.venv-cpu ;;
  gpu) module load gcc arrow python/3.12 cuda cudnn;          VENV=.venv-gpu
       WHEELS="$WHEELS jax_cuda12_plugin jax_cuda12_pjrt" ;;
  *)   echo "usage: ./build_replication_venv.sh [cpu|gpu]" >&2; exit 1 ;;
esac

echo "=== building ${VENV} (${MODE}) ==="
rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index $WHEELS
pip install --no-index $GRAM_DEPS
pip install --no-deps 'nlls-gram>=2.4.0' 'tinydiffeq>=2.4.0'
python -c "import jax, flax, jsonargparse, matplotlib, pandas, pyarrow, scipy, nlls_gram, tinydiffeq; print('venv ok | jax', jax.__version__, '| devices', jax.devices())"
echo "BUILD_DONE (${VENV})"
