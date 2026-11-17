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
import sys
from pathlib import Path

import pytest

pytest.importorskip("torchao")

import torch

from executorch.backends.espressif import to_espnn_edge
from executorch.backends.espressif.quantizer.espnn_quantizer import EspNNQuantizer
from executorch.backends.espressif.reference import (
    esp_nn_reference as reference,
    ProgramEmulator,
)
from executorch.backends.espressif.serialization import xts3
from executorch.backends.espressif.test import (
    test_cpu_fallback as fallback_cases,
    test_int8_lowering as lowering_cases,
    test_int8_ops as op_cases,
)
from executorch.backends.espressif.test.test_mutable_buffers import (
    InputAsState,
    lower_input_as_state,
    lower_stateful,
    ShiftRegister,
    StateReadLate,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quantize_io_pass import quantize_input, quantize_output
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

RUNNER = os.environ.get("ESPNN_HOST_RUNNER")
pytestmark = pytest.mark.skipif(
    not RUNNER, reason="Set ESPNN_HOST_RUNNER to espnn_host_runner"
)


class RepeatedOutput(torch.nn.Module):
    """Returns one tensor twice, so two delegate outputs share a graph tensor."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, x):
        y = self.conv(x)
        return y, y


class LongCat(torch.nn.Module):
    """A cat of seven tensors, which two COPY operations write into one output."""

    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList(torch.nn.Conv2d(3, 2, 1) for _ in range(7))

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=2)


class AddChain(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def forward(self, x):
        x = InputAsState.qdq(x, 0.1)
        for _ in range(self.count):
            x = InputAsState.qdq(x + x, 0.1)
        return x


class CpuBetweenAdds(torch.nn.Module):
    def forward(self, x):
        x = InputAsState.qdq(x, 0.1)
        x = InputAsState.qdq(x + x, 0.1)
        x = InputAsState.qdq(torch.cumsum(x, dim=1), 0.1)
        return InputAsState.qdq(x + x, 0.1)


# Fully delegated programs, checked bit-exactly; transpose-select-stack leaves a
# cat on the CPU and runs with the CPU-fallback programs below. single-conv has
# no activation besides its input and output.
CASES = [
    ("single-conv", torch.nn.Conv2d(3, 4, 1), [(1, 3, 8, 8)]),
    ("repeated-output", RepeatedOutput(), [(1, 3, 8, 8)]),
    ("seven-way-cat", LongCat(), [(1, 3, 4, 4)]),
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


def lower(
    model,
    shapes,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
    xts3_version: int | None = None,
):
    _, program = op_cases.quantize(model, shapes)
    edge, report = to_espnn_edge(program, allow_cpu_fallback=False)
    assert not report["cpu_ops"]
    if xts3_version is not None:
        for _, module in edge.exported_program().graph_module.named_children():
            if isinstance(module, LoweredBackendModule):
                graph = xts3.parse_graph(module.processed_bytes)
                graph.version = xts3_version
                module._processed_bytes = xts3.serialize_graph(graph)
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=alloc_graph_input, alloc_graph_output=alloc_graph_output
        )
    )
    return edge.exported_program(), edge.to_executorch(config).buffer


def as_list(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def to_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's bytes in memory order, which may be channels-last."""
    if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor.contiguous().numpy().tobytes()


def from_bytes(data: bytes, like: torch.Tensor, channels_last: bool) -> torch.Tensor:
    channels_last = channels_last and like.dim() == 4
    shape = tuple(like.permute(0, 2, 3, 1).shape) if channels_last else like.shape
    tensor = torch.frombuffer(bytearray(data), dtype=like.dtype).reshape(shape)
    return tensor.permute(0, 3, 1, 2) if channels_last else tensor


def run_host(tmp_path: Path, program: bytes, steps, *args):
    """
    Runs one list of input tensors per step. Returns the output files per
    instance and step, and the delegates' memory statistics.
    """
    (tmp_path / "program.pte").write_bytes(program)
    for step, inputs in enumerate(steps):
        for i, tensor in enumerate(inputs):
            (tmp_path / f"in_{step}_{i}.bin").write_bytes(to_bytes(tensor))
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


def assert_matches(
    paths: list[Path], expected: list[torch.Tensor], channels_last: bool = False
):
    """Compares output files, in the program's dim order, with the reference."""
    assert len(paths) == len(expected)
    for path, tensor in zip(paths, expected):
        actual = from_bytes(path.read_bytes(), tensor, channels_last)
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
    # Version 7 graphs keep every requantization table in place.
    assert stats["parameters"] == 0
    if stats["scratch"]:
        assert stats["arena"] == planned
    else:
        assert stats["arena"] >= planned


@pytest.mark.parametrize("alloc_graph_input", [False, True])
@pytest.mark.parametrize("alloc_graph_output", [False, True])
@pytest.mark.parametrize(
    "input_int8, output_int8, channels_last, legacy, adds, arena_slots",
    [
        pytest.param(False, False, False, False, 2, 2, id="float-nchw"),
        pytest.param(False, False, True, False, 2, 2, id="float-nhwc"),
        pytest.param(True, True, False, False, 2, 2, id="int8-nchw"),
        pytest.param(True, True, True, False, 2, 2, id="int8-nhwc"),
        pytest.param(False, True, True, False, 1, 1, id="float-input-only"),
        pytest.param(True, False, True, False, 1, 2, id="float-output-only"),
        pytest.param(False, False, False, True, 2, 2, id="legacy-float"),
        pytest.param(True, True, True, True, 2, 2, id="legacy-int8"),
    ],
)
def test_conversion_buffers_share_the_arena(
    tmp_path,
    input_int8,
    output_int8,
    channels_last,
    legacy,
    adds,
    arena_slots,
    alloc_graph_input,
    alloc_graph_output,
):
    shape = (1, 16, 64, 64)
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    sample = torch.zeros(shape).contiguous(memory_format=memory_format)
    program = torch.export.export(AddChain(adds), (sample,), strict=True)
    if input_int8:
        quantize_input(program, 0)
    if output_int8:
        quantize_output(program, 0)
    edge, _ = to_espnn_edge(program, allow_cpu_fallback=False)
    if legacy:
        for module in edge.exported_program().graph_module.children():
            if isinstance(module, LoweredBackendModule):
                module.compile_specs[:] = [
                    spec
                    for spec in module.compile_specs
                    if spec.key != "espnn_io_conversion_slots"
                ]
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=alloc_graph_input, alloc_graph_output=alloc_graph_output
        )
    )
    buffer = edge.to_executorch(config).buffer
    _, loaded = run_host(tmp_path, buffer, [])
    assert loaded["arena"] == arena_slots * sample.numel()
    assert loaded["staging"] == 0
    steps = random_steps([shape], 2)
    steps = [
        [
            ((x * 20).round().to(torch.int8) if input_int8 else x).contiguous(
                memory_format=memory_format
            )
        ]
        for (x,) in steps
    ]
    (outputs,), stats = run_host(tmp_path, buffer, steps)
    emulator = ProgramEmulator(edge.exported_program())
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)), channels_last)
    assert stats["arena"] == loaded["arena"]
    assert stats["staging"] == 0


def test_partitions_keep_independent_conversion_plans(tmp_path):
    program = torch.export.export(CpuBetweenAdds(), (torch.zeros(1, 4),))
    quantize_input(program, 0)
    edge, _ = to_espnn_edge(program)
    steps = [[torch.full((1, 4), value, dtype=torch.int8)] for value in (1, 3)]
    (outputs,), stats = run_host(tmp_path, edge.to_executorch().buffer, steps)
    emulator = ProgramEmulator(edge.exported_program())
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
    assert stats["delegates"] == 2
    # Inputs are always copied; both partitions' outputs need conversion.
    assert stats["arena"] == 32 + 32
    assert stats["staging"] == 0


def test_inputs_are_copied_for_exports_that_bound_them(tmp_path):
    # Exports that recorded no conversion slots for compatible inputs.
    sample = torch.zeros(1, 16, 8, 8).contiguous(memory_format=torch.channels_last)
    program = torch.export.export(AddChain(2), (sample,), strict=True)
    quantize_input(program, 0)
    quantize_output(program, 0)
    edge, _ = to_espnn_edge(program, allow_cpu_fallback=False)
    for module in edge.exported_program().graph_module.children():
        if isinstance(module, LoweredBackendModule):
            module.compile_specs[:] = [
                (
                    CompileSpec(spec.key, bytes(4))
                    if spec.key == "espnn_io_conversion_slots"
                    else spec
                )
                for spec in module.compile_specs
            ]
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False, alloc_graph_output=False
        )
    )
    generator = torch.Generator().manual_seed(2)
    steps = [
        [
            torch.randint(
                -128, 128, sample.shape, generator=generator, dtype=torch.int8
            ).contiguous(memory_format=torch.channels_last)
        ]
        for _ in range(2)
    ]
    (outputs,), stats = run_host(
        tmp_path, edge.to_executorch(config).buffer, steps, "--share-io"
    )
    emulator = ProgramEmulator(edge.exported_program())
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)), True)
    assert stats["arena"] == 2 * sample.numel()
    assert stats["staging"] == 0


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
    exported, program = lower(
        model, shapes, alloc_graph_input=False, alloc_graph_output=False
    )
    steps = random_steps(shapes, 2)
    (outputs,), _ = run_host(tmp_path, program, steps, "--share-io")
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))


def test_version_6_graphs_still_run(tmp_path):
    model, shapes = lowering_cases.SEBlockModel(), [(1, 3, 16, 16)]
    exported, program = lower(model, shapes, xts3_version=6)
    steps = random_steps(shapes, 2)
    (outputs,), stats = run_host(tmp_path, program, steps)
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
    assert all(d.graph.version == 6 for d in emulator.delegates.values())
    assert stats["parameters"] > 0


FALLBACK_CASES = [
    ("transpose-select-stack", op_cases.TransposeSelect(), [(1, 3, 4, 5)])
] + [(name, model, shapes) for name, model, shapes, _ in fallback_cases.CASES]


@pytest.mark.parametrize(
    "name, model, shapes", FALLBACK_CASES, ids=[c[0] for c in FALLBACK_CASES]
)
def test_programs_with_cpu_operators(tmp_path, name, model, shapes):
    # Delegates sit between portable kernels, which receive and return their
    # tensors (some delegates return one tensor as several outputs); the
    # kernels' float rounding gets the emulator suite's tolerance.
    _, program = op_cases.quantize(model, shapes)
    edge, report = to_espnn_edge(program)
    assert report["cpu_ops"]
    exported = edge.exported_program()
    steps = random_steps(shapes, 2)
    (outputs,), _ = run_host(tmp_path, edge.to_executorch().buffer, steps)
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        expected = [t.to(torch.float32) for t in as_list(emulator(*inputs))]
        peak = max(float(t.abs().max()) for t in expected)
        for path, tensor in zip(outputs[step], expected):
            actual = from_bytes(path.read_bytes(), tensor, False)
            torch.testing.assert_close(
                actual, tensor, atol=0.01 * max(peak, 1.0), rtol=0.01
            )


class StreamingConv(torch.nn.Module):
    """One step of a streaming convolution: the state holds the last frames."""

    def __init__(self, channels_last: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, (3, 1))
        self.memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )

    def forward(self, x, state):
        window = torch.cat([state, x], dim=2)
        new_state = window[:, :, 1:].clone(memory_format=self.memory_format)
        return self.conv(window).clone(memory_format=self.memory_format), new_state


class LateStateRead(StreamingConv):
    """Writes its new state before it reads the old one."""

    def __init__(self, channels_last: bool):
        super().__init__(channels_last)
        self.conv = torch.nn.Conv2d(4, 8, (2, 1))
        self.frame = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x, state):
        new_state = (
            self.frame(x).expand(-1, -1, 2, -1).clone(memory_format=self.memory_format)
        )
        return self.conv(state).clone(memory_format=self.memory_format), new_state


class EarlyOutput(LateStateRead):
    """LateStateRead with its output computed before its new state."""

    def forward(self, x, state):
        output = self.conv(state).clone(memory_format=self.memory_format)
        new_state = (
            self.frame(x).expand(-1, -1, 2, -1).clone(memory_format=self.memory_format)
        )
        return output, new_state


class ChainedOutputs(StreamingConv):
    """Returns a tensor and two functions of it, computed one after the other."""

    def forward(self, x, state):
        a = self.conv(torch.cat([state, x], dim=2)).clone(
            memory_format=self.memory_format
        )
        b = torch.relu(a).clone(memory_format=self.memory_format)
        c = torch.sigmoid(a).clone(memory_format=self.memory_format)
        return a, b, c


def lower_streaming(channels_last: bool, model_class=StreamingConv, patch=None):
    """
    int8 inputs and outputs, not memory planned, optionally channels-last.
    patch, if given, edits each delegate's XTS3 graph before serialization.
    """
    model = model_class(channels_last).eval()
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    shapes = [(1, 4, 1, 1), (1, 4, 2, 1)]

    def sample():
        return tuple(
            torch.randn(shape).to(memory_format=memory_format) for shape in shapes
        )

    example = sample()
    prepared = prepare_pt2e(
        torch.export.export(model, example, strict=True).module(), EspNNQuantizer()
    )
    for _ in range(16):
        prepared(*sample())
    program = torch.export.export(convert_pt2e(prepared), example, strict=True)
    for index in range(len(program.graph_signature.user_inputs)):
        quantize_input(program, index)
    for index in range(len(program.graph_signature.user_outputs)):
        quantize_output(program, index)
    edge, report = to_espnn_edge(program, allow_cpu_fallback=False)
    assert not report["cpu_ops"]
    if patch is not None:
        for _, module in edge.exported_program().graph_module.named_children():
            if isinstance(module, LoweredBackendModule):
                graph = xts3.parse_graph(module.processed_bytes)
                patch(graph)
                module._processed_bytes = xts3.serialize_graph(graph)
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False, alloc_graph_output=False
        )
    )
    return edge.exported_program(), edge.to_executorch(config).buffer, sample


@pytest.mark.parametrize("channels_last", [False, True])
def test_device_layout_io_is_used_in_place(tmp_path, channels_last):
    exported, program, sample = lower_streaming(channels_last)
    emulator = ProgramEmulator(exported)
    steps = []
    for _ in range(2):
        floats = sample()
        steps.append(
            [torch.clamp((t * 20).round(), -128, 127).to(torch.int8) for t in floats]
        )
    (outputs,), stats = run_host(tmp_path, program, steps, "--share-io")
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)), channels_last)
    # Inputs and NCHW outputs use the arena; channels-last outputs are bound
    # directly.
    assert stats["staging"] == 0
    if channels_last:
        assert stats["arena"] == sum(
            reference.plan_activation_arena(delegate.graph, external_io=True)[0]
            for delegate in emulator.delegates.values()
        )


@pytest.mark.parametrize("channels_last", [False, True])
def test_output_sharing_an_input_buffer(tmp_path, channels_last):
    # The state output is written into the state input's buffer, as an
    # application updating its state in place would bind them.
    exported, program, _ = lower_streaming(channels_last, LateStateRead)
    emulator = ProgramEmulator(exported)
    generator = torch.Generator().manual_seed(3)
    steps = [
        [
            torch.randint(-128, 128, shape, generator=generator, dtype=torch.int8)
            for shape in [(1, 4, 1, 1), (1, 4, 2, 1)]
        ]
        for _ in range(4)
    ]
    if channels_last:
        steps = [
            [t.contiguous(memory_format=torch.channels_last) for t in inputs]
            for inputs in steps
        ]
    (outputs,), _ = run_host(tmp_path, program, steps, "--share-io", "--alias", "1:1")
    state = steps[0][1]
    for step, inputs in enumerate(steps):
        expected = as_list(emulator(inputs[0], state))
        assert_matches(outputs[step], expected, channels_last)
        state = expected[1]


def int8_steps(count: int, channels_last: bool, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    return [
        [
            torch.randint(
                -128, 128, shape, generator=generator, dtype=torch.int8
            ).contiguous(memory_format=memory_format)
            for shape in [(1, 4, 1, 1), (1, 4, 2, 1)]
        ]
        for _ in range(count)
    ]


def test_misaligned_buffers_are_staged(tmp_path):
    exported, program, _ = lower_streaming(channels_last=True)
    emulator = ProgramEmulator(exported)
    steps = int8_steps(2, channels_last=True)
    (outputs,), stats = run_host(tmp_path, program, steps, "--share-io", "--misalign")
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)), True)
    assert stats["staging"] > 0


@pytest.mark.parametrize("channels_last", [False, True])
def test_outputs_read_by_operators_keep_their_slots(tmp_path, channels_last):
    # Output 1 is written over output 0's buffer before an operator reads
    # output 0 for output 2, so output 0 must stay in its arena slot.
    exported, program, _ = lower_streaming(channels_last, ChainedOutputs)
    emulator = ProgramEmulator(exported)
    steps = int8_steps(2, channels_last)
    (outputs,), stats = run_host(
        tmp_path, program, steps, "--share-io", "--share-output", "0:1"
    )
    for step, inputs in enumerate(steps):
        expected = as_list(emulator(*inputs))
        assert_matches(outputs[step][2:], expected[2:], channels_last)
    assert stats["staging"] == 0
    if channels_last:
        assert stats["arena"] == sum(
            reference.plan_activation_arena(delegate.graph, external_io=True)[0]
            for delegate in emulator.delegates.values()
        )


def return_state_input(graph):
    """Makes the graph return its state input instead of its new state."""
    state_in = graph.inputs[1]
    shape = graph.tensors[state_in].shape
    (index,) = [
        k for k, t in enumerate(graph.outputs) if graph.tensors[t].shape == shape
    ]
    graph.tensors[graph.outputs[index]].flags &= ~xts3.TENSOR_FLAG_GRAPH_OUTPUT
    graph.tensors[state_in].flags |= xts3.TENSOR_FLAG_GRAPH_OUTPUT
    graph.outputs[index] = state_in


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("alias", [[], ["--alias", "1:0"]], ids=["", "aliased"])
def test_input_returned_as_output(tmp_path, channels_last, alias):
    # With the alias, the first output is written over the returned input's
    # buffer, which must not change the second output.
    exported, program, _ = lower_streaming(
        channels_last, EarlyOutput, patch=return_state_input
    )
    emulator = ProgramEmulator(exported)
    steps = int8_steps(1 if alias else 2, channels_last)
    (outputs,), _ = run_host(tmp_path, program, steps, "--share-io", *alias)
    for step, inputs in enumerate(steps):
        expected = as_list(emulator(*inputs))
        assert torch.equal(expected[1], inputs[1])
        assert_matches(outputs[step], expected, channels_last)


def stateful_steps(count: int):
    return random_steps([(1, 4, 1, 1)], count, seed=7)


@pytest.mark.parametrize(
    "model_class, in_place", [(ShiftRegister, 1), (StateReadLate, 0)]
)
def test_delegate_owned_state_persists(tmp_path, model_class, in_place):
    exported, program, report = lower_stateful(model_class())
    assert report["delegated_buffers"] == ["state"]
    assert len(exported.graph_signature.user_inputs) == 1
    assert not exported.graph_signature.buffers_to_mutate
    steps = stateful_steps(5)
    (outputs,), stats = run_host(tmp_path, program, steps)
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
    assert stats["states"] == 1
    assert stats["states_in_place"] == in_place
    # The state changes the outputs, so the check above depends on it.
    replay = ProgramEmulator(exported)
    stateful = [as_list(replay(*inputs))[0] for inputs in steps]
    fresh = [as_list(ProgramEmulator(exported)(*inputs))[0] for inputs in steps]
    assert not all(torch.equal(a, b) for a, b in zip(stateful, fresh))


def test_state_updated_from_a_graph_input(tmp_path):
    exported, program, _ = lower_input_as_state()
    steps = stateful_steps(4)
    (outputs,), stats = run_host(tmp_path, program, steps)
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
    assert stats["states"] == 1


def test_state_reset_and_instances(tmp_path):
    exported, program, _ = lower_stateful(ShiftRegister())
    steps = stateful_steps(6)
    first, second = run_host(
        tmp_path, program, steps, "--instances", 2, "--reset-before", 4
    )[0]
    emulator = ProgramEmulator(exported)
    for step, inputs in enumerate(steps):
        if step == 4:
            emulator.reset_states()
        expected = as_list(emulator(*inputs))
        assert_matches(first[step], expected)
        assert_matches(second[step], expected)


def test_streaming_example_runs_without_copies(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/espressif"))
    import export_streaming_model as example

    edge = example.lower(calibration_frames=32)
    program = edge.to_executorch(example.executorch_config()).buffer
    generator = torch.Generator().manual_seed(1)
    steps = [
        [
            torch.randint(
                -128,
                128,
                (1, example.FEATURES, 1, 1),
                generator=generator,
                dtype=torch.int8,
            )
        ]
        for _ in range(6)
    ]
    (outputs,), stats = run_host(tmp_path, program, steps, "--share-io")
    emulator = ProgramEmulator(edge.exported_program())
    for step, inputs in enumerate(steps):
        assert_matches(outputs[step], as_list(emulator(*inputs)))
    assert stats["states"] == 2 and stats["staging"] == 0
