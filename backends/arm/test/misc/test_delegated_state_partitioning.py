# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping

import pytest
import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase

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


@pytest.mark.parametrize(
    "module_type",
    [_StateAlsoReadOnHost, _StateReadByTwoDelegates, _StateUpdatedOnHost],
    ids=["also_read_on_host", "read_by_two_delegates", "updated_on_host"],
)
def test_shared_state_stays_a_program_buffer_u85_INT(module_type):
    assert _partition_state_tag(module_type()) is None
