# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes goldens.json for the C++ scheduler and RNG tests.

Run from the repository root with torch, diffusers and executorch installed:

    python extension/diffusers/runner/test/generate_goldens.py
"""

import json
import math
from pathlib import Path

import diffusers
import torch

SD15_PNDM = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "num_train_timesteps": 1000,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "trained_betas": None,
}
SDXL_EULER = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False,
}
FLUX_DEV = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}
FLUX_SCHNELL = {**FLUX_DEV, "shift": 1.0, "use_dynamic_shifting": False}


def flux_sigmas(steps):
    return torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64).tolist()


def flux_mu(image_sequence_length):
    return 0.5 + (image_sequence_length - 256) * (1.15 - 0.5) / (4096 - 256)


SCHEDULER_CASES = [
    ("sd15_pndm", "PNDMScheduler", SD15_PNDM, [{"steps": 50}, {"steps": 4}]),
    (
        "pndm_v_prediction_trailing",
        "PNDMScheduler",
        {
            **SD15_PNDM,
            "prediction_type": "v_prediction",
            "timestep_spacing": "trailing",
        },
        [{"steps": 7}],
    ),
    ("sdxl_euler", "EulerDiscreteScheduler", SDXL_EULER, [{"steps": 50}, {"steps": 3}]),
    (
        "euler_v_prediction_trailing",
        "EulerDiscreteScheduler",
        {
            **SDXL_EULER,
            "prediction_type": "v_prediction",
            "timestep_spacing": "trailing",
        },
        [{"steps": 3}, {"steps": 25}],
    ),
    (
        "euler_linear_linspace",
        "EulerDiscreteScheduler",
        {"num_train_timesteps": 1000},
        [{"steps": 10}],
    ),
    (
        "flux_dev",
        "FlowMatchEulerDiscreteScheduler",
        FLUX_DEV,
        [
            {"steps": 28, "sigmas": flux_sigmas(28), "mu": flux_mu(4096)},
            {"steps": 4, "sigmas": flux_sigmas(4), "mu": flux_mu(1024)},
        ],
    ),
    (
        "flux_schnell",
        "FlowMatchEulerDiscreteScheduler",
        FLUX_SCHNELL,
        [{"steps": 4, "sigmas": flux_sigmas(4), "mu": flux_mu(4096)}],
    ),
    (
        "flow_static_shift",
        "FlowMatchEulerDiscreteScheduler",
        {"num_train_timesteps": 1000, "shift": 3.0},
        [{"steps": 28}, {"steps": 5, "sigmas": flux_sigmas(5)}],
    ),
    (
        "flow_shift_terminal",
        "FlowMatchEulerDiscreteScheduler",
        {**FLUX_DEV, "shift_terminal": 0.02},
        [{"steps": 20, "sigmas": flux_sigmas(20), "mu": flux_mu(4096)}],
    ),
    (
        "flow_linear_time_shift",
        "FlowMatchEulerDiscreteScheduler",
        {**FLUX_DEV, "time_shift_type": "linear"},
        [{"steps": 6, "sigmas": flux_sigmas(6), "mu": 1.3}],
    ),
]

# Inputs shared with the C++ test, which derives the per-step model output
# from them with exact float32 arithmetic.
SAMPLE = [math.sin(i) for i in range(16)]
NOISE = [math.cos(3 * i) for i in range(16)]
RECORDED_STEPS = 4


def model_output(step):
    sample = torch.tensor(SAMPLE, dtype=torch.float32)
    return torch.roll(sample, step) * 0.5


def run_scheduler(class_name, config, run):
    scheduler = getattr(diffusers, class_name).from_config(config)
    kwargs = {k: run[k] for k in ("sigmas", "mu") if k in run}
    scheduler.set_timesteps(run["steps"], **kwargs)
    timesteps = scheduler.timesteps.to(torch.float32)

    sample = torch.tensor(SAMPLE, dtype=torch.float32)
    samples = []
    scaled = []
    for index, timestep in enumerate(scheduler.timesteps):
        scaled_input = sample
        if hasattr(scheduler, "scale_model_input"):
            scaled_input = scheduler.scale_model_input(sample, timestep)
        sample = scheduler.step(
            model_output(index), timestep, sample, return_dict=False
        )[0]
        if index < RECORDED_STEPS:
            scaled.append(scaled_input.tolist())
            samples.append(sample.tolist())
    result = {
        "steps": run["steps"],
        "sigmas": run.get("sigmas", []),
        "mu": run.get("mu"),
        "timesteps": timesteps.tolist(),
        "init_noise_sigma": float(getattr(scheduler, "init_noise_sigma", 1.0)),
        "scaled_inputs": scaled,
        "samples": samples,
        "final_sample": sample.tolist(),
    }
    if hasattr(scheduler, "add_noise"):
        noisy = getattr(diffusers, class_name).from_config(config)
        noisy.set_timesteps(run["steps"], **kwargs)
        index = min(1, len(noisy.timesteps) - 1)
        if hasattr(noisy, "set_begin_index"):
            noisy.set_begin_index(index)
        result["add_noise_index"] = index
        result["add_noise"] = noisy.add_noise(
            torch.tensor(SAMPLE, dtype=torch.float32),
            torch.tensor(NOISE, dtype=torch.float32),
            noisy.timesteps[index : index + 1],
        ).tolist()
    return result


RNG_CASES = [
    (0, [("float32", 37)]),
    (42, [("float32", 100)]),
    (42, [("float16", 37)]),
    (7, [("bfloat16", 100)]),
    (2**32 + 5, [("float32", 16)]),
    (3, [("float32", 37), ("float16", 50), ("float32", 32)]),
]


def bits(tensor):
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32).tolist()
    return [value & 0xFFFF for value in tensor.view(torch.int16).tolist()]


def main():
    schedulers = []
    for name, class_name, config, runs in SCHEDULER_CASES:
        schedulers.append(
            {
                "name": name,
                "class_name": class_name,
                "config": config,
                "runs": [run_scheduler(class_name, config, run) for run in runs],
            }
        )
    rng = []
    for seed, draws in RNG_CASES:
        generator = torch.Generator().manual_seed(seed)
        rng.append(
            {
                "seed": seed,
                "draws": [
                    {
                        "dtype": dtype,
                        "bits": bits(
                            torch.randn(
                                numel, generator=generator, dtype=getattr(torch, dtype)
                            )
                        ),
                    }
                    for dtype, numel in draws
                ],
            }
        )
    goldens = {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "sample": SAMPLE,
        "noise": NOISE,
        "schedulers": schedulers,
        "rng": rng,
    }
    path = Path(__file__).with_name("goldens.json")
    path.write_text(json.dumps(goldens, separators=(",", ":")) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
