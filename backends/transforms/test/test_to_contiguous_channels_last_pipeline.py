# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.transforms.to_contiguous_channels_last_pass import (
    ToContiguousChannelsLastPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class ConvChain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.relu(self.conv1(x)))


class ConvOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvThenLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.linear = torch.nn.Linear(4 * 8 * 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(x).flatten(1))


class UserPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class ConvChannelBias(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.register_buffer("channel_bias", torch.randn(1, 4, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.channel_bias


class PadConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.nn.functional.pad(x, (1, 1, 1, 1)))


class ConvSoftmax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.conv(x), dim=-1)


class ConvRuntimeBiasConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + bias)


class ConvSpatialBufferConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.register_buffer("bias", torch.randn(4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + self.bias)


def _edge(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    exported = torch.export.export(module.eval(), inputs)
    return to_edge(
        exported,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )


def _count(graph_module: torch.fx.GraphModule, target: object) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def test_conv_chain_folds_to_boundary_copies() -> None:
    torch.manual_seed(0)
    module = ConvChain().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])
    graph_module = transformed.exported_program().graph_module

    assert _count(graph_module, exir_ops.edge.channels_last.convolution.default) == 2
    assert _count(graph_module, exir_ops.edge.channels_last.permute_copy.default) == 2
    assert layout_pass.report.inserted_copy_count == 4
    assert layout_pass.report.eliminated_copy_count == 2
    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 0
    assert layout_pass.report.unknown_copy_count == 0
    assert layout_pass.report.boundary_copy_bytes == 2048
    assert layout_pass.report.internal_copy_bytes == 0
    assert layout_pass.report.unknown_copy_bytes == 0
    assert layout_pass.report.copies_with_unknown_size == 0
    actual = transformed.exported_program().module()(*inputs)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_strict_rejects_internal_layout_copy() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _edge(ConvThenLinear().eval(), inputs)

    with pytest.raises(RuntimeError, match="left .* internal"):
        ToContiguousChannelsLastPass(edge.exported_program(), strict=True).call(
            edge.exported_program().graph_module
        )


def test_strict_rejects_unknown_boundary_copy_size() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    exported = torch.export.export(
        ConvOnly().eval(),
        inputs,
        dynamic_shapes={"x": {2: torch.export.Dim("height", min=4, max=16)}},
    )
    edge = to_edge(
        exported,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program(), strict=True)

    with pytest.raises(RuntimeError, match="unknown sizes"):
        layout_pass.call(edge.exported_program().graph_module)

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.boundary_copy_bytes == 0
    assert layout_pass.report.copies_with_unknown_size == 2


def test_backend_can_block_layout_propagation_at_a_node() -> None:
    torch.manual_seed(0)
    module = ConvChain().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(
        edge.exported_program(),
        can_propagate=lambda node: node.target != exir_ops.edge.aten.relu.default,
    )

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 2
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("module", [ConvChannelBias(), PadConv()])
def test_one_sided_propagation_reaches_graph_boundary(
    module: torch.nn.Module,
) -> None:
    torch.manual_seed(0)
    module.eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 0
    assert layout_pass.report.unknown_copy_count == 0
    actual = transformed.exported_program().module()(*inputs)
    assert torch.allclose(actual, expected, atol=1e-6)

    if isinstance(module, PadConv):
        assert (
            _count(
                transformed.exported_program().graph_module,
                exir_ops.edge.channels_last.constant_pad_nd.default,
            )
            == 1
        )


def test_softmax_blocks_layout_propagation() -> None:
    module = ConvSoftmax().eval()
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 1
    assert layout_pass.report.internal_copy_count == 1
    softmax = next(
        node
        for node in transformed.exported_program().graph.nodes
        if node.target
        in (exir_ops.edge.aten._softmax.default, exir_ops.edge.aten.softmax.int)
    )
    assert softmax.args[1] in (-1, 3)
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "module, inputs",
    [
        (
            ConvRuntimeBiasConv(),
            (torch.randn(1, 4, 8, 8), torch.randn(4, 1, 1)),
        ),
        (ConvSpatialBufferConv(), (torch.randn(1, 4, 8, 8),)),
    ],
)
def test_boundary_propagation_rejects_unsafe_broadcast_rewrites(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
) -> None:
    module.eval()
    expected = module(*inputs)
    edge = _edge(module, inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program())

    transformed = edge.transform([layout_pass])

    assert layout_pass.report.boundary_copy_count == 2
    assert layout_pass.report.internal_copy_count == 2
    actual = transformed.exported_program().module()(*inputs)
    torch.testing.assert_close(actual, expected)


def test_layout_copy_report_is_idempotent() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _edge(ConvChain().eval(), inputs)
    first_pass = ToContiguousChannelsLastPass(edge.exported_program())
    transformed = edge.transform([first_pass])
    second_pass = ToContiguousChannelsLastPass(transformed.exported_program())

    transformed.transform([second_pass])

    assert second_pass.report.inserted_copy_count == 0
    assert second_pass.report.eliminated_copy_count == 0


def test_user_permute_is_not_reported_as_layout_copy() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _edge(UserPermute(), inputs)
    layout_pass = ToContiguousChannelsLastPass(edge.exported_program(), op_map={})

    transformed = edge.transform([layout_pass])
    permutes = [
        node
        for node in transformed.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.permute_copy.default
    ]

    assert len(permutes) == 1
    assert (
        _count(
            transformed.exported_program().graph_module,
            exir_ops.edge.channels_last.permute_copy.default,
        )
        == 0
    )
    assert layout_pass.report.inserted_copy_count == 0
    assert layout_pass.report.boundary_copy_count == 0
    assert layout_pass.report.internal_copy_count == 0
