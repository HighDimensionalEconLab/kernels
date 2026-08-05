from __future__ import annotations

import json
import pathlib
import time
import traceback
from dataclasses import asdict

import jax
import jsonargparse

from stochastic_open_economy_growth_nn_recursive import (
    WORLD_RATE_SIGMA,
    DataSettings,
    InitialConditionSettings,
    InitializationSettings,
    SequentialPolicySettings,
    StoppingCriteria,
    stochastic_open_economy_growth_nn_recursive,
)
from utilities import NEOCLASSICAL_GROWTH_CONSTANTS, OptimizerSettings


def append_record(path, record):
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=True, sort_keys=True) + "\n")


def run_tuning_trial(
    trial: str,
    output_path: str = ".results_stochastic_open_economy/tuning.jsonl",
    seed: int = 0,
    data_seed: int = 0,
    sigma_z: float = NEOCLASSICAL_GROWTH_CONSTANTS.sigma_z_high,
    sigma_r: float = WORLD_RATE_SIGMA,
    width: int = 32,
    depth: int = 3,
    hidden_activation: str = "gelu",
    num_train_trajectories: int = 8,
    num_test_trajectories: int = 16,
    num_tvc_trajectories: int = 128,
    train_points: int = 21,
    test_points: int = 41,
    tvc_points: int = 41,
    train_T: float = 10.0,
    test_T: float = 10.0,
    tvc_T: float = 20.0,
    max_steps: int = 12,
    num_epochs: int = 8,
    continuation_epochs: int = 3,
    num_attempts: int = 1,
    train_loss_threshold: float = 1e-8,
    test_diffusion_max_abs_threshold: float = 5e-2,
    test_terminal_capital_relative_threshold: float = 0.25,
    epoch_early_stopping: bool = True,
    init_damping: float = 3.26e-5,
    damping_decrease: float = 0.482,
    damping_increase: float = 2.71,
    geodesic_acceleration: bool = True,
    geodesic_acceptance_ratio: float = 0.541,
    diagnose_jacobian: bool = False,
    deterministic_reference: bool = True,
    initialize_from_reference: bool = True,
    initialization_grid_size: int = 65,
    initialization_max_steps: int = 40,
    initialization_loss_threshold: float = 1e-10,
):
    data_set = DataSettings(
        train_points=train_points,
        test_points=test_points,
        tvc_points=tvc_points,
        train_T=train_T,
        test_T=test_T,
        tvc_T=tvc_T,
    )
    ic_set = InitialConditionSettings(
        data_seed=data_seed,
        num_train_trajectories=num_train_trajectories,
        num_test_trajectories=num_test_trajectories,
        num_tvc_trajectories=num_tvc_trajectories,
    )
    nn_set = SequentialPolicySettings(
        width=width,
        depth=depth,
        hidden_activation=hidden_activation,
    )
    stop_set = StoppingCriteria(
        max_steps=max_steps,
        train_loss_threshold=train_loss_threshold,
        test_diffusion_max_abs_threshold=test_diffusion_max_abs_threshold,
        test_terminal_capital_relative_threshold=(
            test_terminal_capital_relative_threshold
        ),
        num_attempts=num_attempts,
        num_epochs=num_epochs,
        continuation_epochs=continuation_epochs,
        epoch_early_stopping=epoch_early_stopping,
    )
    init_set = InitializationSettings(
        enabled=initialize_from_reference,
        grid_size=initialization_grid_size,
        max_steps=initialization_max_steps,
        loss_threshold=initialization_loss_threshold,
    )
    opt_set = OptimizerSettings(
        init_damping=init_damping,
        damping_decrease=damping_decrease,
        damping_increase=damping_increase,
        geodesic_acceleration=geodesic_acceleration,
        geodesic_acceptance_ratio=geodesic_acceptance_ratio,
    )
    config = {
        "seed": seed,
        "data_seed": data_seed,
        "sigma_z": sigma_z,
        "sigma_r": sigma_r,
        "data_set": asdict(data_set),
        "ic_set": asdict(ic_set),
        "nn_set": asdict(nn_set),
        "stop_set": asdict(stop_set),
        "init_set": asdict(init_set),
        "opt_set": asdict(opt_set),
        "diagnose_jacobian": diagnose_jacobian,
        "deterministic_reference": deterministic_reference,
        "timing_scope": "cold_start_including_jit_compilation",
    }
    started = time.perf_counter()
    try:
        results = stochastic_open_economy_growth_nn_recursive(
            seed=seed,
            sigma_z=sigma_z,
            sigma_r=sigma_r,
            data_set=data_set,
            ic_set=ic_set,
            nn_set=nn_set,
            stop_set=stop_set,
            init_set=init_set,
            opt_set=opt_set,
            diagnose_jacobian=diagnose_jacobian,
            deterministic_reference=deterministic_reference,
            verbose=False,
        )[0]
        record = {
            "trial": trial,
            "run_status": "completed",
            "wall_time": time.perf_counter() - started,
            "config": config,
            "results": results,
        }
    except Exception as error:
        record = {
            "trial": trial,
            "run_status": "failed",
            "wall_time": time.perf_counter() - started,
            "config": config,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        append_record(output_path, record)
        raise
    append_record(output_path, record)
    print(json.dumps(record, indent=2, allow_nan=True, sort_keys=True))
    return record


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jsonargparse.CLI(run_tuning_trial)
