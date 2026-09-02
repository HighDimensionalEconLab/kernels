import importlib.util
import multiprocessing
import queue as queue_module
import re
import sys
import time
from pathlib import Path

import jsonargparse
import numpy as np

# Multi-seed HPO wrapper: a wandb sweep invokes this instead of the experiment
# script. It runs the target function for several replicable seeds, sorts the
# successful runs by a criteria key (smaller is better), and logs the median
# seed's results under plain names plus the best seed's results under a `best_`
# prefix.
#
# Target contract: the script contains a jsonargparse-CLI-compatible function
# that returns either a metrics dict, or a tuple whose first element is the
# metrics dict (optionally followed by an extras dict), and accepts a seed
# argument (name set by --sweep_settings.seed_param).
#
# Ex-post gates: --sweep_settings.gates is a comma-separated list of
# `key<=value` / `key>=value` / `key==value` conditions on the results dict and
# --sweep_settings.gated_key names a results key; every seed gains
# `<gated_key>_gated` (the value when all gates hold, NaN otherwise) and a
# `gate_failures` string, so `sort_criteria=<gated_key>_gated` optimizes the
# key conditional on the gates.
#
# --sweep_settings.seed_timeout runs each seed in a spawned process with a
# wall-clock cap; a seed that exceeds it is terminated and counted as a crash.

GATE = re.compile(r"^(\w+)(<=|>=|==|<|>)(.+)$")


def load_function(script, function_name):
    script_path = Path(script).resolve()
    # The target imports its package siblings (`from models.x import ...`), so
    # the repo root (the agent's cwd) and the script's directory are both
    # importable regardless of where the agent launches from.
    sys.path.insert(0, str(script_path.parent))
    sys.path.insert(0, str(Path.cwd()))
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name or module_name)


def split_output(out, extra_keys):
    results = out[0] if isinstance(out, tuple) else out
    extras = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
    return results, {k: extras[k] for k in extra_keys if k in extras}


def run_seed(script, function_name, args, extra_keys, channel):
    fn = load_function(script, function_name)
    try:
        out = jsonargparse.CLI(fn, args=args)
    except Exception as e:
        channel.put(("error", repr(e)))
        return
    channel.put(("ok", *split_output(out, extra_keys)))


def main():
    settings = {
        "script": None,  # required: python file containing the function
        "function_name": None,  # defaults to the script filename stem
        "sort_criteria": None,  # required: key in the results dict; smaller is better
        "num_seeds": 5,
        "seed_param": "seed",
        "starting_seed": 42,
        "seed_timeout": None,  # seconds per seed; None runs in-process, uncapped
        "gates": "",
        "gated_key": None,
        "wandb_mode": "online",  # "online" | "offline" | "disabled" (print only)
        # Log the per-seed seed_results / extra_log_keys wandb.Tables (default true).
        # Set false for large PRODUCTIVE grid sweeps where each run is one data point:
        # the tables bloat every run's summary and make the bulk API pull slow, and a
        # 1-seed run's table is redundant with its own scalar metrics.
        "log_tables": "true",
        "extra_log_keys": "",  # comma-separated keys from the median run's extras
        # If set, and fewer than this many seeds produced a finite sort_criteria,
        # blank (NaN) the logged sort_criteria -- both the median value and its
        # best_ -- so an HPO optimizer skips this under-supported config while
        # every other metric still logs for inspection.
        "min_successful_seeds": None,
    }
    int_keys = {"num_seeds", "starting_seed", "min_successful_seeds", "seed_timeout"}

    # Manual argv walk: --sweep_settings.* args (both --key=value and --key value
    # forms) configure the wrapper; everything else passes through verbatim.
    prefix = "--sweep_settings."
    passthrough = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith(prefix):
            if "=" in arg:
                key, value = arg[len(prefix) :].split("=", 1)
            else:
                key = arg[len(prefix) :]
                i += 1
                if i >= len(argv):
                    raise SystemExit(f"Missing value for {arg}")
                value = argv[i]
            if key not in settings:
                raise SystemExit(f"Unknown sweep_settings key: {key}")
            settings[key] = int(value) if key in int_keys else value
        else:
            passthrough.append(arg)
        i += 1

    if settings["script"] is None:
        raise SystemExit("--sweep_settings.script is required")
    if settings["sort_criteria"] is None:
        raise SystemExit("--sweep_settings.sort_criteria is required")

    gates = []
    for clause in (c.strip() for c in settings["gates"].split(",") if c.strip()):
        match = GATE.match(clause)
        if match is None:
            raise SystemExit(f"Malformed gate: {clause!r}")
        gates.append((match.group(1), match.group(2), float(match.group(3))))
    if gates and settings["gated_key"] is None:
        raise SystemExit("--sweep_settings.gated_key is required with gates")
    compare = {
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
    }

    disabled = settings["wandb_mode"] == "disabled"
    if not disabled:
        import wandb

        # Under a sweep agent this picks up the sweep config automatically.
        wandb.init(mode=settings["wandb_mode"])

    extra_keys = [k.strip() for k in settings["extra_log_keys"].split(",") if k.strip()]
    timeout = settings["seed_timeout"]
    if timeout is None:
        fn = load_function(settings["script"], settings["function_name"])

    # Replicable uint32 seeds derived from starting_seed.
    seeds = np.random.SeedSequence(settings["starting_seed"]).generate_state(
        settings["num_seeds"]
    )

    sort_criteria = settings["sort_criteria"]
    runs = []
    num_crashes = 0
    num_timeouts = 0
    gate_failures = "none"
    for s in seeds:
        seed = int(s)
        args = passthrough + [f"--{settings['seed_param']}={seed}"]
        # A bad passthrough arg raises SystemExit (a BaseException), which
        # propagates and fails the whole trial loudly: a config error, not
        # seed noise. Only Exception counts as a per-seed crash.
        if timeout is None:
            try:
                message = (
                    "ok",
                    *split_output(jsonargparse.CLI(fn, args=args), extra_keys),
                )
            except Exception as e:
                message = ("error", repr(e))
        else:
            context = multiprocessing.get_context("spawn")
            channel = context.Queue()
            process = context.Process(
                target=run_seed,
                args=(
                    settings["script"],
                    settings["function_name"],
                    args,
                    extra_keys,
                    channel,
                ),
            )
            process.start()
            message = None
            deadline = time.monotonic() + timeout
            while message is None and time.monotonic() < deadline:
                try:
                    message = channel.get(timeout=1.0)
                except queue_module.Empty:
                    if not process.is_alive():
                        break
            if message is None:
                if process.is_alive():
                    process.terminate()
                    process.join()
                    num_timeouts += 1
                    message = ("error", f"timed out after {timeout}s")
                else:
                    process.join()
                    raise SystemExit(f"seed {seed} exited with code {process.exitcode}")
            process.join()
        if message[0] == "error":
            print(f"seed {seed} crashed: {message[1]}", file=sys.stderr)
            num_crashes += 1
            continue
        _, results, extras = message
        if gates:
            failed = [
                f"{key}{op}{bound:g}"
                for key, op, bound in gates
                if key not in results or not compare[op](float(results[key]), bound)
            ]
            results[f"{settings['gated_key']}_gated"] = (
                float(results[settings["gated_key"]]) if not failed else float("nan")
            )
            gate_failures = ",".join(failed) or "none"
            results["gate_failures"] = gate_failures
        try:
            criteria = float(results[sort_criteria])
        except (KeyError, TypeError, ValueError):
            criteria = float("nan")
        if not np.isfinite(criteria):
            print(
                f"seed {seed} produced no finite {sort_criteria!r}; counting as crash",
                file=sys.stderr,
            )
            num_crashes += 1
            continue
        runs.append(
            {"seed": seed, "criteria": criteria, "results": results, "extras": extras}
        )

    if not runs:
        # No seed produced a finite objective. Log only the crash counts (no
        # metric, so the sweep optimizer skips this config) and exit 0: exiting
        # non-zero counts as a crash toward wandb agent's flapping protection,
        # which kills a worker after a few consecutive failures -- fatal for
        # high-failure-rate sweeps that must run the full walltime.
        log = {
            "num_crashes": num_crashes,
            "num_timeouts": num_timeouts,
            "gate_failures": gate_failures,
        }
        if disabled:
            print(f"[wandb disabled] all seeds failed; would log: {log}")
            return
        wandb.log(log)
        wandb.finish()
        return

    runs.sort(key=lambda r: r["criteria"])
    best = runs[0]
    median = runs[len(runs) // 2]  # rounds up to the worse middle for even n

    log = dict(median["results"])
    log.update({f"best_{k}": v for k, v in best["results"].items()})
    log["num_crashes"] = num_crashes
    log["num_timeouts"] = num_timeouts
    log["num_seeds_completed"] = len(runs)
    log["median_seed"] = median["seed"]
    log["best_seed"] = best["seed"]

    # Too few seeds reached a finite sort_criteria: keep every other metric so the
    # run stays inspectable, but NaN the HPO objective (median + best_) so the
    # sweep optimizer ignores this under-supported config.
    min_successful = settings["min_successful_seeds"]
    if min_successful is not None and len(runs) < min_successful:
        log[sort_criteria] = float("nan")
        log[f"best_{sort_criteria}"] = float("nan")

    def is_scalar(v):
        return isinstance(v, (bool, int, float, str, np.integer, np.floating))

    columns = ["seed"] + sorted(
        {k for r in runs for k, v in r["results"].items() if is_scalar(v)}
    )
    table_rows = [
        [r["seed"]] + [r["results"].get(k) for k in columns[1:]] for r in runs
    ]

    extra_log = {}
    for key in extra_keys:
        if key not in median["extras"]:
            print(f"extra_log_keys: {key!r} not in extras; skipping", file=sys.stderr)
            continue
        extra_log[key] = median["extras"][key]

    if disabled:
        print(f"[wandb disabled] would log: {log}")
        print(f"[wandb disabled] seed_results columns: {columns}")
        for row in table_rows:
            print(f"[wandb disabled]   {row}")
        for key, value in extra_log.items():
            print(f"[wandb disabled] extra {key!r}: {type(value).__name__}")
        return

    if settings["log_tables"].lower() not in ("false", "0", "no"):
        log["seed_results"] = wandb.Table(columns=columns, data=table_rows)
        for key, value in extra_log.items():
            # Duck-typed DataFrame check keeps pandas out of the dependencies.
            if hasattr(value, "columns") and hasattr(value, "itertuples"):
                log[key] = wandb.Table(dataframe=value)
            else:
                log[key] = value
    wandb.log(log)
    wandb.finish()


if __name__ == "__main__":
    main()
