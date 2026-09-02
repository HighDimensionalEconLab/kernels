# Sweeps — wandb HPO

All wandb sweeps for this repo log to the **`kernels`** project. Every sweep yaml sets
`project: kernels`; keep that consistent when adding new configs.

The sweep configs, all run via `wandb_sweep_wrapper.py`:
`stochastic_growth_nn_time_hpo.yaml` and `stochastic_open_economy_nn_time_hpo.yaml`
(accepted solve time, `solve_time_gated`), `stochastic_open_economy_nn_hpo.yaml` and
`stochastic_open_economy_kernel_hpo.yaml` (held-out Euler error). The wrapper's
`--sweep_settings.seed_timeout` caps each seed's wall clock in a spawned process and
`--sweep_settings.gates` / `gated_key` derive `<gated_key>_gated` from ex-post
conditions on the results dict. Smoke-test one point locally with no wandb
credentials by passing the chosen yaml's `--sweep_settings.script` and parameters
to the wrapper with `--sweep_settings.wandb_mode=disabled` from the repo root.

Local sweeps without wandb: `multicountry_growth_hpo.py` and
`stochastic_open_economy_kernel_local_hpo.py` (jsonargparse CLIs; invocation in each header).

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
   `sweep_ids.json` (`{ "stochastic_growth_nn_time_hpo": "entity/kernels/<id>" }`).
3. Submit one `wandb agent` per array task with `extra/sweeps/wandb_sweep_agent.sbatch`
   (CPU on Fir: `--account=def-jlperla_cpu --cpus-per-task=4 --mem=16G --time=03:00:00
   --array=1-10`, `VENV=.venv-cpu`, unbounded — omit `AGENT_COUNT`). See the `clusters`
   skill `workflows/wandb-sweep.md`.

## Monitoring
`monitor_top5.py` reads `sweep_ids.json` and prints the top-5 solves per sweep every 5
minutes (also appended to `logs/monitor.log`). Run locally (auth via `~/.netrc`):

    uv run --group sweep python extra/sweeps/monitor_top5.py        # 5-min loop
    uv run --group sweep python extra/sweeps/monitor_top5.py --once # single pass
