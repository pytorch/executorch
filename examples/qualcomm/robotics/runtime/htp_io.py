# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host side of qnn_smolvla_runner: one adb execute runs one plan.

The PTEs and the runner are pushed once with SimpleADB.push(). Each plan writes
the SmolVLAInputs fields as <field>.raw, pushes only those files, executes the
runner, and pulls action_chunk.raw back.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from executorch.backends.qualcomm.export_utils import QnnConfig
from executorch.examples.qualcomm.robotics.policies.smolvla.policy import (
    SmolVLAInputs,
)

RUNNER = "examples/qualcomm/robotics/runtime/qnn_smolvla_runner"
ACTION_CHUNK_FILENAME = "action_chunk.raw"
_RUNNER_TIMING = re.compile(
    r"\[qnn_smolvla_runner\] plan_ms=(?P<plan_ms>\S+) execute_ms=(?P<execute_ms>\S+)"
)


def write_raw_inputs(inputs: SmolVLAInputs, directory: Path) -> list[Path]:
    """Write each SmolVLAInputs field as <field>.raw."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for field, tensor in zip(dataclasses.fields(inputs), inputs.as_model_args()):
        path = directory / f"{field.name}.raw"
        path.write_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())
        paths.append(path)
    return paths


def push_inputs(qnn_config: QnnConfig, files: Sequence[Path], workspace: str) -> None:
    """Push only the plan inputs; SimpleADB.push() would also re-push every PTE."""
    command = ["adb"]
    if qnn_config.host:
        command += ["-H", qnn_config.host]
    command += ["-s", qnn_config.device, "push", *map(str, files), workspace]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def runner_command(
    workspace: str,
    *,
    vision_pte: str,
    prefix_pte: str,
    action_pte: str,
    num_steps: int,
) -> str:
    runner = Path(RUNNER).name
    return " ".join(
        [
            f"cd {workspace} &&",
            f"chmod +x {runner} &&",
            "export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. &&",
            f"rm -f {ACTION_CHUNK_FILENAME} &&",
            f"./{runner}",
            f"--vision_pte {vision_pte}",
            f"--prefix_pte {prefix_pte}",
            f"--action_pte {action_pte}",
            "--input_dir .",
            f"--output_path {ACTION_CHUNK_FILENAME}",
            f"--num_steps {num_steps}",
        ]
    )


def read_action_chunk(path: Path, shape: Sequence[int]) -> Tensor:
    data = bytearray(path.read_bytes())
    expected_bytes = torch.Size(shape).numel() * torch.float32.itemsize
    if len(data) != expected_bytes:
        raise ValueError(
            f"{path} must be {expected_bytes} bytes for float32 {tuple(shape)}, "
            f"got {len(data)}"
        )
    return torch.frombuffer(data, dtype=torch.float32).reshape(shape).clone()


def parse_runner_timing(stdout: str) -> tuple[float, float]:
    """Return (plan_ms, execute_ms) reported by qnn_smolvla_runner."""
    match = _RUNNER_TIMING.search(stdout)
    if match is None:
        raise RuntimeError(f"qnn_smolvla_runner did not report its timing:\n{stdout}")
    return float(match["plan_ms"]), float(match["execute_ms"])
