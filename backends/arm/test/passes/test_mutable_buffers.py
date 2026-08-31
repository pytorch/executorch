# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast
from unittest.mock import patch

import torch
from executorch.backends.arm._passes import DeduplicateGetAttrPass, mutable_buffer_utils
from executorch.backends.arm._passes.mutable_buffer_utils import (
    collect_mutable_buffer_infos,
    restore_mutable_buffer_targets,
)
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)


class MutableBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(2, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        new_state = state + x.mean()
        state.copy_(new_state)
        return new_state.mean(dim=1)


class NestedMutableBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.child = MutableBufferModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.child(x)


class IdenticalMutableBuffersModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state_a", torch.zeros(4))
        self.register_buffer("state_b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_a = cast(torch.Tensor, self.state_a)
        state_b = cast(torch.Tensor, self.state_b)
        state_a.copy_(state_a + x)
        state_b.copy_(state_b - x)
        return state_a + state_b


class SharedStorageBuffersModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        storage = torch.arange(8, dtype=torch.float32)
        self.register_buffer("state", storage[:4])
        self.register_buffer("other", storage[4:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = cast(torch.Tensor, self.state)
        other = cast(torch.Tensor, self.other)
        state.copy_(state + x)
        return state + other


def _quantizer() -> EthosUQuantizer:
    quantizer = EthosUQuantizer(
        EthosUCompileSpec(target="ethos-u85-256", memory_mode="Shared_Sram")
    )
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


def test_restore_nested_mutable_buffer_target() -> None:
    graph_module = torch.export.export(
        NestedMutableBufferModel().eval(), (torch.ones(4),), strict=False
    ).module()
    mutable_buffers = collect_mutable_buffer_infos(graph_module)

    restore_mutable_buffer_targets(graph_module, mutable_buffers)

    assert "child.state" in dict(graph_module.named_buffers())
    assert any(
        node.op == "get_attr" and node.target == "child.state"
        for node in graph_module.graph.nodes
    )


def test_collect_skips_graph_traversal_without_registered_buffers() -> None:
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    value = graph.placeholder("value")
    copy = graph.call_function(torch.ops.aten.copy_.default, (state, value))
    graph.output(copy)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with patch.object(
        mutable_buffer_utils,
        "_mutable_buffer_get_attrs",
        side_effect=AssertionError("unexpected graph traversal"),
    ):
        assert collect_mutable_buffer_infos(graph_module) == {}


def test_transform_restores_identical_mutable_buffers() -> None:
    graph_module = torch.export.export(
        IdenticalMutableBuffersModel().eval(), (torch.ones(4),), strict=False
    ).module()

    transformed = _quantizer().transform_for_annotation(graph_module)

    buffers = dict(transformed.named_buffers())
    assert "state_a" in buffers
    assert "state_b" in buffers
    get_attr_targets = {
        node.target for node in transformed.graph.nodes if node.op == "get_attr"
    }
    assert "state_a" in get_attr_targets
    assert "state_b" in get_attr_targets

    transformed(torch.ones(4))
    torch.testing.assert_close(cast(torch.Tensor, transformed.state_a), torch.ones(4))
    torch.testing.assert_close(cast(torch.Tensor, transformed.state_b), -torch.ones(4))


def test_transform_restores_nested_mutable_buffer_module() -> None:
    graph_module = torch.export.export(
        NestedMutableBufferModel().eval(), (torch.ones(4),), strict=False
    ).module()

    transformed = _quantizer().transform_for_annotation(graph_module)

    buffers = dict(transformed.named_buffers())
    assert "child.state" in buffers
    transformed(torch.ones(4))
    torch.testing.assert_close(buffers["child.state"], torch.ones(2, 4))


def test_restore_identical_buffers_after_mutation_reordering() -> None:
    graph_module = torch.export.export(
        IdenticalMutableBuffersModel().eval(), (torch.ones(4),), strict=False
    ).module()
    mutable_buffers = collect_mutable_buffer_infos(graph_module)
    result = DeduplicateGetAttrPass(tfa_pass=True)(graph_module)
    assert result is not None
    transformed = result.graph_module
    copy_nodes = [
        node
        for node in transformed.graph.nodes
        if node.target == torch.ops.aten.copy_.default
    ]
    copy_nodes[1].append(copy_nodes[0])
    transformed.graph.lint()
    transformed.recompile()

    restore_mutable_buffer_targets(transformed, mutable_buffers)
    transformed(torch.ones(4))

    torch.testing.assert_close(cast(torch.Tensor, transformed.state_a), torch.ones(4))
    torch.testing.assert_close(cast(torch.Tensor, transformed.state_b), -torch.ones(4))


def test_restore_does_not_merge_disjoint_buffer_views() -> None:
    graph_module = torch.export.export(
        SharedStorageBuffersModel().eval(), (torch.ones(4),), strict=False
    ).module()
    mutable_buffers = collect_mutable_buffer_infos(graph_module)

    restore_mutable_buffer_targets(graph_module, mutable_buffers)

    get_attr_targets = {
        node.target for node in graph_module.graph.nodes if node.op == "get_attr"
    }
    assert "state" in get_attr_targets
    assert "other" in get_attr_targets
    torch.testing.assert_close(
        graph_module(torch.ones(4)), torch.tensor([5.0, 7.0, 9.0, 11.0])
    )
