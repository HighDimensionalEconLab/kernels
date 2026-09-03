import json
import sys
import time
from pathlib import Path

import wandb

# Local monitor: every 5 minutes, print the top-5 solves (smallest median
# solve_time_gated = solve time of an accepted, gate-passing solve) for each
# sweep named in sweep_ids.json, and append the same block to
# logs/monitor.log. Runs off ~/.netrc auth and works live while the cluster
# agents are still filling the sweeps. Pass --once for a single pass.

METRIC = "solve_time_gated"
HERE = Path(__file__).resolve().parent
IDS_FILE = HERE / "sweep_ids.json"
LOG_FILE = HERE / "logs" / "monitor.log"
# Union over the sweeps; keys absent from a run's config are skipped.
CFG_KEYS = [
    "nn_set.width",
    "nn_set.depth",
    "nn_set.hidden_activation",
    "tolerance_set.max_steps",
    "tolerance_set.num_epochs",
    "tolerance_set.train_loss_threshold",
    "ic_set.num_train_trajectories",
    "data_set.train_points",
    "sde_solver",
    "lm_set.init_damping",
    "lm_set.damping_decrease",
    "lm_set.damping_increase",
    "lm_set.geodesic_acceptance_ratio",
]


def fmt(v):
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def top5(sweep_id):
    sweep = wandb.Api().sweep(sweep_id)
    runs = list(sweep.runs)
    done = [
        r
        for r in runs
        if isinstance(r.summary.get(METRIC), (int, float))
        and r.summary.get(METRIC) == r.summary.get(METRIC)  # not NaN
    ]
    done.sort(key=lambda r: float(r.summary[METRIC]))
    return runs, done


def main():
    once = "--once" in sys.argv
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    while True:
        lines = [f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} ====="]
        try:
            ids = json.loads(IDS_FILE.read_text())
        except FileNotFoundError:
            ids = {}
            lines.append(f"(no {IDS_FILE.name} yet — write it after `wandb sweep`)")
        for name, sweep_id in ids.items():
            try:
                runs, done = top5(sweep_id)
            except Exception as e:
                lines.append(f"\n## {name} [{sweep_id}] -- query error: {e!r}")
                continue
            best = done[:5]
            lines.append(
                f"\n## {name} [{sweep_id}] -- {len(runs)} runs, "
                f"{len(done)} finished with {METRIC}"
            )
            if not best:
                lines.append("   (no finished trials yet)")
            for i, r in enumerate(best, 1):
                cfg = " ".join(
                    f"{k.split('.')[-1]}={fmt(r.config[k])}"
                    for k in CFG_KEYS
                    if k in r.config
                )
                nsc = r.summary.get("num_seeds_completed")
                ncr = r.summary.get("num_crashes")
                lines.append(
                    f"   {i}. {METRIC}={float(r.summary[METRIC]):.3f}s  "
                    f"seeds_ok={nsc} crashes={ncr}  {cfg}"
                )
        block = "\n".join(lines)
        print(block, flush=True)
        with LOG_FILE.open("a") as f:
            f.write(block + "\n")
        if once:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
