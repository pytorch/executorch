# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Lowering of mutable buffers (streaming state) with delegate_mutable_buffers:
which buffers the ESP-NN delegate owns, and the stateful host reference.
test_host_runtime.py runs the same models on the delegate.

    pytest backends/espressif/test/test_mutable_buffers.py
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("torchao")

import torch

from executorch.backends.espressif import to_espnn_edge
from executorch.backends.espressif.partition import espnn_partitioner as partitioner
from executorch.backends.espressif.quantizer.espnn_quantizer import EspNNQuantizer
from executorch.backends.espressif.reference import ProgramEmulator
from executorch.backends.espressif.serialization import xts3
from executorch.exir.lowered_backend_module import LoweredBackendModule
from torch.fx.passes.infra.partitioner import Partition
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class ShiftRegister(torch.nn.Module):
    """A streaming convolution over the last frames, kept in a buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 2, 1))
        self.conv = torch.nn.Conv2d(4, 8, (3, 1))

    def forward(self, x):
        window = torch.cat([self.state, x], dim=2)
        self.state.copy_(window[:, :, 1:])
        return self.conv(window)


class StateReadLate(ShiftRegister):
    """Computes its new state before an operator reads the old one."""

    def forward(self, x):
        new_state = torch.cat([x, x], dim=2)
        y = self.conv(torch.cat([self.state, x], dim=2))
        self.state.copy_(new_state)
        return y


class InputAsState(torch.nn.Module):
    """
    Explicit int8 Q/DQ (no PT2E): the new state is the quantized input itself,
    a graph input of the delegate, which the delegate copies into the state.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 1, 1))

    @staticmethod
    def qdq(value, scale):
        args = (scale, 0, -128, 127, torch.int8)
        quantized = torch.ops.quantized_decomposed.quantize_per_tensor(value, *args)
        return torch.ops.quantized_decomposed.dequantize_per_tensor(quantized, *args)

    def forward(self, x):
        frame = self.qdq(x, 0.05)
        y = self.qdq(self.qdq(self.state, 0.05) + frame, 0.1)
        self.state.copy_(frame)
        return y


class ReturnsOldState(InputAsState):
    """InputAsState that also returns the state it read."""

    def forward(self, x):
        old = self.qdq(self.state, 0.05)
        frame = self.qdq(x, 0.05)
        y = self.qdq(old + frame, 0.1)
        self.state.copy_(frame)
        return y, old


class ReturnsClonedOldState(InputAsState):
    """ReturnsOldState with the old state returned through a clone."""

    def forward(self, x):
        old = self.qdq(self.state, 0.05)
        frame = self.qdq(x, 0.05)
        y = self.qdq(old + frame, 0.1)
        self.state.copy_(frame)
        return y, self.qdq(old.clone(), 0.05)


class StateOnlyInput(torch.nn.Module):
    """Explicit int8 Q/DQ recurrence whose only input is its state."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 1, 1))

    def forward(self):
        qdq = InputAsState.qdq
        state = qdq(self.state, 0.1)
        update = qdq(state + state, 0.1)
        self.state.copy_(update)
        return qdq(update + state, 0.1)


class StateOnlyOutput(torch.nn.Module):
    """Explicit int8 Q/DQ accumulator whose only delegated output is its state."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 1, 1))

    def forward(self, x):
        qdq = InputAsState.qdq
        self.state.copy_(qdq(qdq(self.state, 0.1) + qdq(x, 0.1), 0.1))
        return torch.cumsum(x, dim=1)


class StateOnlyOutputWithConstant(StateOnlyOutput):
    """StateOnlyOutput whose CPU operator reads a constant."""

    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.full((1, 4, 1, 1), 0.2))

    def forward(self, x):
        qdq = InputAsState.qdq
        self.state.copy_(qdq(qdq(self.state, 0.1) + qdq(x, 0.1), 0.1))
        return torch.cumsum(x + self.offset, dim=1)


def lower_input_as_state(model_class=InputAsState):
    program = torch.export.export(
        model_class().eval(), (torch.zeros(1, 4, 1, 1),), strict=True
    )
    edge, report = to_espnn_edge(
        program, allow_cpu_fallback=False, delegate_mutable_buffers=True
    )
    # to_executorch() rewrites the mutations of buffers left in the program.
    exported = copy.deepcopy(edge.exported_program())
    return exported, edge.to_executorch().buffer, report


class ScaledState(ShiftRegister):
    """Its new state has other quantization parameters than the state."""

    def forward(self, x):
        window = torch.cat([self.state, x], dim=2)
        self.state.copy_(window[:, :, 1:] * 0.5)
        return self.conv(window)


class TwoStreams(torch.nn.Module):
    """Two shift registers on separate inputs, which partitions may split."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 4, 2, 1))
        self.register_buffer("other", torch.zeros(1, 4, 2, 1))
        self.conv = torch.nn.Conv2d(4, 8, (3, 1))
        self.other_conv = torch.nn.Conv2d(4, 8, (3, 1))

    def forward(self, x, y):
        window = torch.cat([self.state, x], dim=2)
        self.state.copy_(window[:, :, 1:])
        other = torch.cat([self.other, y], dim=2)
        self.other.copy_(other[:, :, 1:])
        return self.conv(window), self.other_conv(other)


def split_streams(monkeypatch):
    """Makes EspNNPartitioner propose each connected part as a partition."""
    propose = partitioner.CapabilityBasedPartitioner.propose_partitions

    def propose_streams(self):
        streams = []
        for partition in propose(self):
            nodes = set(partition.nodes)
            while nodes:
                stream, frontier = set(), [nodes.pop()]
                while frontier:
                    node = frontier.pop()
                    stream.add(node)
                    for other in [*node.all_input_nodes, *node.users]:
                        if other in nodes:
                            nodes.discard(other)
                            frontier.append(other)
                streams.append(Partition(nodes=stream))
        return streams

    monkeypatch.setattr(
        partitioner.CapabilityBasedPartitioner, "propose_partitions", propose_streams
    )


class ManyStates(torch.nn.Module):
    """More shift registers than one delegate can own."""

    def __init__(self, count: int):
        super().__init__()
        for k in range(count):
            self.register_buffer(f"state_{k}", torch.zeros(1, 4, 2, 1))
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv2d(4, 2, (3, 1)) for _ in range(count)
        )

    def forward(self, x):
        outputs = []
        for k, conv in enumerate(self.convs):
            state = getattr(self, f"state_{k}")
            window = torch.cat([state, x], dim=2)
            state.copy_(window[:, :, 1:])
            outputs.append(conv(window))
        return torch.cat(outputs, dim=1)


def quantize_stateful(model, inputs: int = 1):
    shape = (1, 4, 1, 1)
    generator = torch.Generator().manual_seed(5)
    example = tuple(torch.zeros(shape) for _ in range(inputs))
    prepared = prepare_pt2e(
        torch.export.export(model.eval(), example, strict=True).module(),
        EspNNQuantizer(),
    )
    for _ in range(16):
        prepared(*(torch.randn(shape, generator=generator) for _ in example))
    converted = convert_pt2e(prepared)
    # Calibration ran the model, so start again from the initial state.
    for name, _ in model.named_buffers():
        getattr(converted, name).zero_()
    return converted


def lower_stateful(
    model,
    delegate_mutable_buffers: bool = True,
    quantized=None,
    allow_cpu_fallback: bool = False,
    inputs: int = 1,
):
    converted = (
        copy.deepcopy(quantized) if quantized else quantize_stateful(model, inputs)
    )
    example = tuple(torch.zeros(1, 4, 1, 1) for _ in range(inputs))
    program = torch.export.export(converted, example, strict=True)
    edge, report = to_espnn_edge(
        program,
        allow_cpu_fallback=allow_cpu_fallback,
        delegate_mutable_buffers=delegate_mutable_buffers,
    )
    # to_executorch() rewrites the mutations of buffers left in the program.
    exported = copy.deepcopy(edge.exported_program())
    return exported, edge.to_executorch().buffer, report


def delegate_graphs(exported) -> list[xts3.XTS3Graph]:
    return [
        xts3.parse_graph(module.processed_bytes)
        for _, module in exported.graph_module.named_children()
        if isinstance(module, LoweredBackendModule)
    ]


@pytest.fixture(autouse=True)
def disable_grad():
    with torch.no_grad():
        yield


def test_delegate_owns_the_state():
    exported, _, report = lower_stateful(ShiftRegister())
    assert report["delegated_buffers"] == ["state"]
    assert len(exported.graph_signature.user_inputs) == 1
    assert not exported.graph_signature.buffers_to_mutate
    (graph,) = delegate_graphs(exported)
    assert graph.version == xts3.XTS3_STATE_VERSION
    ((state, update),) = graph.states
    assert graph.tensors[state].flags & xts3.TENSOR_FLAG_STATE
    assert graph.inputs == [graph.inputs[0]] and state not in graph.inputs
    assert xts3.parse_graph(xts3.serialize_graph(graph)).states == graph.states


def test_reference_tracks_the_quantized_model():
    model = quantize_stateful(ShiftRegister())
    exported, _, _ = lower_stateful(None, quantized=model)
    emulator = ProgramEmulator(exported)
    generator = torch.Generator().manual_seed(9)
    for step in range(5):
        x = torch.randn(1, 4, 1, 1, generator=generator)
        if step == 3:
            model.state.zero_()
            emulator.reset_states()
        torch.testing.assert_close(emulator(x), model(x), atol=0.05, rtol=0)


def test_mutable_buffers_stay_in_the_program_by_default():
    exported, _, report = lower_stateful(ShiftRegister(), False)
    assert report["delegated_buffers"] == []
    assert list(exported.graph_signature.buffers_to_mutate.values()) == ["state"]
    (graph,) = delegate_graphs(exported)
    assert graph.version == xts3.XTS3_VERSION and not graph.states


def test_state_with_other_quantization_stays_in_the_program():
    exported, _, report = lower_stateful(ScaledState())
    assert report["delegated_buffers"] == []
    assert list(exported.graph_signature.buffers_to_mutate.values()) == ["state"]


@pytest.mark.parametrize(
    "dtype, reader, writer, writer_scale, owned",
    [
        (torch.float32, "quantize", "dequantize", 0.1, True),
        (torch.int8, "dequantize", "quantize", 0.1, True),
        (torch.int8, "dequantize", "quantize", 0.2, False),
        (torch.int8, "quantize", "dequantize", 0.1, False),
    ],
)
def test_int8_state_compatibility(dtype, reader, writer, writer_scale, owned):
    ops = torch.ops.quantized_decomposed
    graph = torch.fx.Graph()
    buffer = graph.placeholder("buffer")
    buffer.meta["val"] = torch.zeros(2, dtype=dtype)
    read = graph.call_function(
        getattr(ops, f"{reader}_per_tensor").default,
        (buffer, 0.1, 0, -128, 127, torch.int8),
    )
    new_value = graph.call_function(
        getattr(ops, f"{writer}_per_tensor").default,
        (read, writer_scale, 0, -128, 127, torch.int8),
    )
    assert partitioner._int8_state_compatible(buffer, new_value) == owned


@pytest.mark.parametrize("model_class", [ReturnsOldState, ReturnsClonedOldState])
def test_returned_old_state_is_a_copy(model_class):
    # A state cannot also be a graph output, so the graph returns a copy of it.
    exported, _, report = lower_input_as_state(model_class)
    assert report["delegated_buffers"] == ["state"]
    (graph,) = delegate_graphs(exported)
    ((state, _),) = graph.states
    assert state not in graph.outputs
    (copy,) = [op for op in graph.operations if op.inputs == [state]]
    assert copy.opcode == xts3.XTS3OpCode.COPY and copy.outputs[0] in graph.outputs


def test_state_updated_from_a_graph_input():
    exported, _, report = lower_input_as_state()
    assert report["delegated_buffers"] == ["state"]
    (graph,) = delegate_graphs(exported)
    ((state, update),) = graph.states
    # The runtime cannot write a graph input into the state's slot, so the
    # update is a copy of it, made after every other operation.
    assert update not in graph.inputs
    copy = graph.operations[-1]
    assert copy.opcode == xts3.XTS3OpCode.COPY and copy.outputs == [update]
    assert copy.inputs == graph.inputs
    emulator = ProgramEmulator(exported)
    frames = [torch.full((1, 4, 1, 1), 0.1 * k) for k in (1, 2, 3)]
    previous = torch.zeros(1, 4, 1, 1)
    for frame in frames:
        expected = InputAsState.qdq(
            InputAsState.qdq(previous, 0.05) + InputAsState.qdq(frame, 0.05), 0.1
        )
        torch.testing.assert_close(emulator(frame), expected)
        previous = frame


def test_state_in_two_partitions_stays_in_the_program(monkeypatch):
    exported, _, report = lower_stateful(TwoStreams(), inputs=2)
    assert sorted(report["delegated_buffers"]) == ["other", "state"]
    # A reset could reach one partition's state and not yet the other's, so
    # neither delegate owns its buffer.
    split_streams(monkeypatch)
    exported, _, report = lower_stateful(
        TwoStreams(), inputs=2, allow_cpu_fallback=True
    )
    assert report["partitions"] == 2
    assert report["delegated_buffers"] == []
    assert sorted(exported.graph_signature.buffers_to_mutate.values()) == [
        "other",
        "state",
    ]
    assert not any(graph.states for graph in delegate_graphs(exported))


def test_a_delegate_owns_at_most_16_buffers():
    model = ManyStates(xts3.MAX_GRAPH_STATES + 1)
    exported, _, report = lower_stateful(model)
    assert len(report["delegated_buffers"]) == xts3.MAX_GRAPH_STATES
    assert len(exported.graph_signature.buffers_to_mutate) == 1
    (graph,) = delegate_graphs(exported)
    assert len(graph.states) == xts3.MAX_GRAPH_STATES


@pytest.mark.parametrize(
    "model, example",
    [
        (StateOnlyInput(), ()),
        (StateOnlyOutput(), (torch.zeros(1, 4, 1, 1),)),
    ],
    ids=["only-input", "only-output"],
)
def test_state_that_is_a_partition_boundary_stays_in_the_program(model, example):
    # Owning the buffer would leave the partition without a runtime input or
    # output, which an XTS3 graph needs.
    program = torch.export.export(model.eval(), example, strict=True)
    edge, report = to_espnn_edge(program, delegate_mutable_buffers=True)
    assert report["partitions"] == 1 and report["delegated_buffers"] == []
    edge.to_executorch()


def test_constant_shared_with_the_cpu_is_not_a_runtime_output(monkeypatch):
    # tag_constant_data gives a constant that the CPU also reads to the
    # partition, as it would one that the partition reads too, and lowering
    # then duplicates it for the CPU, so it is not a runtime output.
    tag_constants = partitioner.tag_constant_data

    def tag_with_offset(program):
        tag_constants(program)
        buffers = program.graph_signature.inputs_to_buffers
        nodes = list(program.graph.nodes)
        (tag,) = {
            node.meta["delegation_tag"]
            for node in nodes
            if "delegation_tag" in node.meta
        }
        (offset,) = [node for node in nodes if buffers.get(node.name) == "offset"]
        offset.meta["delegation_tag"] = tag

    monkeypatch.setattr(partitioner, "tag_constant_data", tag_with_offset)
    program = torch.export.export(
        StateOnlyOutputWithConstant().eval(), (torch.zeros(1, 4, 1, 1),), strict=True
    )
    edge, report = to_espnn_edge(program, delegate_mutable_buffers=True)
    assert report["delegated_buffers"] == []
    edge.to_executorch()
