# Sweeps — wandb HPO

All wandb sweeps for this repo log to the **`kernels`** project. Every sweep yaml sets
`project: kernels`; keep that consistent when adding new configs.

The sweep configs are `stochastic_open_economy_kernel_hpo.yaml` and
`stochastic_open_economy_nn_hpo.yaml`, run via `wandb_sweep_wrapper.py`.
Smoke-test one point locally with no wandb credentials by passing the chosen
yaml's `--sweep_settings.script` and parameters to the wrapper with
`--sweep_settings.wandb_mode=disabled`.

## Cluster venv
`build_cluster_venv.sh [cpu|gpu]` builds a persistent venv from the Alliance wheelhouse
(`--no-index`). `cpu` (default) builds `.venv-cpu` — the sweeps here are CPU-only (the
solver's MLPs are tiny). Only `nlls-gram` is outside the wheelhouse — it is installed
last with `--no-deps`; pyarrow comes from `module load arrow`, not pip. Build once on a
login node:

    extra/sweeps/build_cluster_venv.sh cpu   # -> .venv-cpu

## Launching on Digital Alliance Clusters
1. `rsync` the repo to `/scratch/$USER/kernel_econ_alignment/` and build the venv (above).
2. `wandb sweep extra/sweeps/<config>.yaml` → capture `entity/kernels/<id>`; write it to
   `sweep_ids.json` (`{ "stochastic_open_economy_nn_hpo": "entity/kernels/<id>" }`).
3. Submit one `wandb agent` per array task with `extra/sweeps/wandb_sweep_agent.sbatch`
   (CPU on Fir: `--account=def-jlperla_cpu --cpus-per-task=4 --mem=32G --time=01:00:00
   --array=1-50`, `VENV=.venv-cpu`, unbounded — omit `AGENT_COUNT`). See the `clusters`
   skill `workflows/wandb-sweep.md`.

## Monitoring
`monitor_top5.py` reads `sweep_ids.json` and prints the top-5 solves per sweep every 5
minutes (also appended to `logs/monitor.log`). Run locally (auth via `~/.netrc`):

    uv run --group sweep python extra/sweeps/monitor_top5.py        # 5-min loop
    uv run --group sweep python extra/sweeps/monitor_top5.py --once # single pass
