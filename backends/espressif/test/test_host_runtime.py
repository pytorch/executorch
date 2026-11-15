# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Runs fully delegated programs on the host build of the ESP-NN delegate
(espnn_runtime built outside ESP-IDF, over esp-nn's portable C kernels) and
checks every output against ProgramEmulator, the bit-exact Python reference.
Skipped unless ESPNN_HOST_RUNNER points to espnn_host_runner; see
backends/espressif/test/host/CMakeLists.txt.

    ESPNN_HOST_RUNNER=cmake-out-espnn-host/espnn_host_runner \
        pytest backends/espressif/test/test_host_runtime.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torchao")

import torch

from executorch.backends.espressif import to_espnn_edge
from executorch.backends.espressif.reference import (
    esp_nn_reference as reference,
    ProgramEmulator,
)
from executorch.backends.espressif.test import (
    test_int8_lowering as lowering_cases,
    test_int8_ops as op_cases,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

RUNNER = os.environ.get("ESPNN_HOST_RUNNER")
pytestmark = pytest.mark.skipif(
    not RUNNER, reason="Set ESPNN_HOST_RUNNER to espnn_host_runner"
)

# The runner links no CPU kernels, so only fully delegated programs run on it;
# transpose-select-stack leaves a cat on the CPU.
CASES = [
    ("conv-act-pool-linear", lowering_cases.ConvActModel(), [(1, 3, 16, 16)]),
    ("concat-mean-linear", lowering_cases.ConcatModel(), [(1, 4, 6, 6)]),
    ("se-lut-mul-softmax", lowering_cases.SEBlockModel(), [(1, 3, 16, 16)]),
] + [
    (name, model, shapes)
    for name, model, shapes, _ in op_cases.CASES
    if name != "transpose-select-stack"
]


@pytest.fixture(autouse=True)
def disable_grad():
    with torch.no_grad():
        yield


def lower(model, shapes, planned_io: bool = True):
    _, program = op_cases.quantize(model, shapes)
    edge, report = to_espnn_edge(program, allow_cpu_fallback=False)
    assert not report["cpu_ops"]
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=planned_io, alloc_graph_output=planned_io
        )
    )
    return edge.exported_program(), edge.to_executorch(config).buffer


def as_list(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def run_host(tmp_path: Path, program: bytes, steps, *args):
    """
    Runs one list of input tensors per step. Returns the output files per
    instance and step, and the delegates' memory statistics.
    """
    (tmp_path / "program.pte").write_bytes(program)
    for step, inputs in enumerate(steps):
        for i, tensor in enumerate(inputs):
            (tmp_path / f"in_{step}_{i}.bin").write_bytes(
                tensor.contiguous().numpy().tobytes()
            )
    result = subprocess.run(
        [RUNNER, str(tmp_path / "program.pte"), str(tmp_path), str(len(steps))]
        + [str(arg) for arg in args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (stats_line,) = [
        line for line in result.stdout.splitlines() if line.startswith("STATS ")
    ]
    stats = {
        key: int(value)
        for key, value in (field.split("=") for field in stats_line.split()[1:])
    }
    instances = 1
    if "--instances" in args:
        instances = int(args[args.index("--instances") + 1])
    outputs = [
        [sorted(tmp_path.glob(f"out_{k}_{step}_*.bin")) for step in range(len(steps))]
        for k in range(instances)
    ]
    return outputs, stats


def assert_matches(paths: list[Path], expected: list[torch.Tensor]):
    assert len(paths) == len(expected)
    for path, tensor in zip(paths, expected):
        actual = torch.frombuffer(
            bytearray(path.read_bytes()), dtype=tensor.dtype
        ).reshape(tensor.shape)
        if tensor.dtype == torch.float32:
            torch.testing.assert_close(actual, tensor, atol=1e-5, rtol=1e-5)
        else:
            assert torch.equal(actual, tensor)


def random_steps(shapes, count: int, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(shape, generator=generator) for shape in shapes]
        for _ in range(count)
    ]


@pytest.mark.parametrize("name, model, shapes", CASES, ids=[c[0] for c in CASES])
def test_host_delegate_matches_reference(tmp_path, name, model, shapes):
    exported, program = lower(model, shapes)
    steps = random_steps(shapes, 2)
    (outputs,), stats = run_host(tmp_path, program, steps)
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
    # With ESP-NN scratch planned in the arena, the arena also holds scratch.
    planned = sum(
        reference.plan_activation_arena(delegate.graph)[0]
        for delegate in emulator.delegates.values()
    )
    assert stats["delegates"] == len(emulator.delegates)
    if stats["scratch"]:
        assert stats["arena"] == planned
    else:
        assert stats["arena"] >= planned


def test_instances_run_independently(tmp_path):
    model, shapes = lowering_cases.ConvActModel(), [(1, 3, 16, 16)]
    exported, program = lower(model, shapes)
    steps = random_steps(shapes, 3)
    (first, second), stats = run_host(tmp_path, program, steps, "--instances", 2)
    assert stats["delegates"] == 2
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        expected = as_list(emulator(*inputs))
        assert_matches(first[step], expected)
        assert_matches(second[step], expected)


def test_unplanned_io_shared_with_the_caller(tmp_path):
    model, shapes = op_cases.TwoInputTwoOutput(), [(1, 3, 12, 12), (1, 6)]
    exported, program = lower(model, shapes, planned_io=False)
    steps = random_steps(shapes, 2)
    (outputs,), _ = run_host(tmp_path, program, steps, "--share-io")
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
