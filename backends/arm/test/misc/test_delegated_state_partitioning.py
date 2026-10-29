# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping

import pytest
import torch
from executorch.backends.arm.arm_vela import VelaCompileResult
from executorch.backends.arm.ethosu import (
    backend as ethosu_backend,
    EthosUCompileSpec,
    EthosUPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase
from tosa.DType import DType  # type: ignore[import-untyped]
from tosa.Op import Op  # type: ignore[import-untyped]
from tosa.TosaGraph import TosaGraph  # type: ignore[import-untyped]

_HISTORY, _CHANNELS = 4, 8


class _KeepAbsOnHost(OperatorSupportBase):
    """Keep abs on the host, so a test can move a state access out."""

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return node.target != exir_ops.edge.aten.abs.default


class _Stateful(torch.nn.Module):
    # An int8 state is not quantized, so its placeholder feeds each reader
    # directly and the partitioner sees every access to it.
    def __init__(self) -> None:
        super().__init__()
        self.state: torch.Tensor
        initial = torch.arange(_HISTORY * _CHANNELS, dtype=torch.int8)
        self.register_buffer("state", initial.reshape(1, _HISTORY, _CHANNELS))


class _StateOnlyInDelegate(_Stateful):
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        window = torch.cat([self.state, frame], dim=1)
        self.state.copy_(window[:, 1:, :])
        return window


class _StateAlsoReadOnHost(_Stateful):
    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        side = torch.abs(self.state)
        window = torch.cat([self.state, frame], dim=1)
        self.state.copy_(window[:, 1:, :])
        return side, window


class _StateReadByTwoDelegates(_Stateful):
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        between = torch.abs(self.state[:, :1, :])
        window = torch.cat([self.state, between, frame], dim=1)
        self.state.copy_(window[:, 2:, :])
        return window


class _StateUpdatedOnHost(_Stateful):
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        window = torch.cat([self.state, frame], dim=1)
        self.state.copy_(torch.abs(window[:, 1:, :]))
        return window


def _partition_state_tag(module: torch.nn.Module) -> str | None:
    """Partition ``module`` and return its state's delegation tag."""
    compile_spec = EthosUCompileSpec("ethos-u85-256").set_delegate_mutable_buffers()
    partitioner = EthosUPartitioner(compile_spec, additional_checks=[_KeepAbsOnHost()])
    frame = torch.zeros(1, 1, _CHANNELS, dtype=torch.int8)
    exported = torch.export.export(module, (frame,), strict=True)
    keep, _ = partitioner.ops_to_not_decompose(exported)
    program = to_edge(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, preserve_ops=keep),
    ).exported_program()
    result = partitioner.partition(program)
    assert result.partition_tags, "nothing was delegated"
    buffers = program.graph_signature.inputs_to_buffers
    (state,) = [
        node
        for node in program.graph.find_nodes(op="placeholder")
        if buffers.get(node.name) == "state"
    ]
    return state.meta.get("delegation_tag")


def test_state_only_in_delegate_is_delegate_owned_u85_INT():
    assert _partition_state_tag(_StateOnlyInDelegate()) is not None


def test_int8_state_lowers_as_persistent_variable_u85_INT(monkeypatch):
    compile_spec = EthosUCompileSpec("ethos-u85-256").set_delegate_mutable_buffers()
    frame = torch.zeros(1, 1, _CHANNELS, dtype=torch.int8)
    exported = torch.export.export(_StateOnlyInDelegate(), (frame,), strict=True)
    tosa_flatbuffers = []

    def capture_tosa(tosa_flatbuffer, compile_flags, **kwargs):
        tosa_flatbuffers.append(tosa_flatbuffer)
        return VelaCompileResult(processed_bytes=b"stub", scratch_size=0)

    monkeypatch.setattr(ethosu_backend, "vela_compile", capture_tosa)

    to_edge_transform_and_lower(
        exported,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    assert len(tosa_flatbuffers) == 1
    graph = TosaGraph.GetRootAs(tosa_flatbuffers[0], 0)
    block = graph.Regions(0).Blocks(0)
    variables = [
        block.Tensors(index)
        for index in range(block.TensorsLength())
        if block.Tensors(index).Variable()
    ]
    assert len(variables) == 1
    assert variables[0].Type() == DType.INT8
    assert variables[0].VariableName()
    assert [
        variables[0].Data(index) for index in range(variables[0].DataLength())
    ] == list(range(_HISTORY * _CHANNELS))
    operators = {
        block.Operators(index).Op() for index in range(block.OperatorsLength())
    }
    assert {Op.VARIABLE_READ, Op.VARIABLE_WRITE} <= operators


@pytest.mark.parametrize(
    "module_type",
    [_StateAlsoReadOnHost, _StateReadByTwoDelegates, _StateUpdatedOnHost],
    ids=["also_read_on_host", "read_by_two_delegates", "updated_on_host"],
)
def test_shared_state_stays_a_program_buffer_u85_INT(module_type):
    assert _partition_state_tag(module_type()) is None
