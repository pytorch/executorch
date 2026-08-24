# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from executorch.backends.transforms.convert_conv1d_to_conv2d_pass import (
    ConvertConv1dToConv2dPass,
)
from executorch.backends.transforms.utils import set_param_tensor
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.program._program import _transform


class Conv1d(torch.nn.Module):
    def __init__(
        self,
        bias: bool = True,
        groups: int = 1,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            2,
            4,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            2, 4, 3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        return self.conv(x)


class DynamicWeightConv1d(torch.nn.Module):
    def forward(self, x, weight):
        return F.conv1d(x, weight, padding=1)


class BufferConv1d(torch.nn.Module):
    def __init__(self, persistent: bool):
        super().__init__()
        self.register_buffer("weight", torch.randn(4, 2, 3), persistent=persistent)

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1)


class ConstantConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 2, 3)

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1)


class SharedConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 2, 3))

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1) + F.conv1d(x, self.weight, padding=1)


class MixedUseWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 2, 3))

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1), self.weight


class WeightUsedAsConvInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.randn(4, 2, 3))
        self.other_weight = torch.nn.Parameter(torch.randn(3, 2, 1))

    def forward(self, x):
        return F.conv1d(x, self.shared, padding=1), F.conv1d(
            self.shared, self.other_weight
        )


def _edge(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    return to_edge(torch.export.export(model.eval(), inputs)).exported_program()


def _convert_and_compare(model: torch.nn.Module, x: torch.Tensor):
    reference = model(x)
    edge = _edge(model, (x,))
    converted = _transform(edge, ConvertConv1dToConv2dPass(edge))
    torch.testing.assert_close(converted.module()(x), reference)
    return converted


@pytest.mark.parametrize("bias, groups", [(True, 1), (False, 1), (False, 2)])
def test_convert_conv1d_to_conv2d(bias: bool, groups: int):
    converted = _convert_and_compare(
        Conv1d(bias=bias, groups=groups), torch.randn(1, 2, 8)
    )
    [conv] = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    assert conv.args[1].meta["val"].shape == torch.Size([4, 2 // groups, 1, 3])
    assert conv.args[3:6] == ([1, 1], [0, 1], [1, 1])
    assert conv.args[0].target == exir_ops.edge.aten.unsqueeze_copy.default
    assert next(iter(conv.users)).target == exir_ops.edge.aten.squeeze_copy.dim


def test_convert_stride_and_dilation():
    converted = _convert_and_compare(
        Conv1d(stride=3, padding=2, dilation=2), torch.randn(1, 2, 16)
    )
    [conv] = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    assert conv.args[3:6] == ([1, 3], [0, 2], [1, 2])


def test_convert_dynamic_activation_shapes():
    model = Conv1d().eval()
    example_input = torch.randn(2, 2, 8)
    batch = torch.export.Dim("batch", min=1, max=4)
    length = torch.export.Dim("length", min=4, max=32)
    edge = to_edge(
        torch.export.export(
            model,
            (example_input,),
            dynamic_shapes={"x": {0: batch, 2: length}},
        )
    ).exported_program()

    converted = _transform(edge, ConvertConv1dToConv2dPass(edge))
    runtime_input = torch.randn(3, 2, 17)

    torch.testing.assert_close(converted.module()(runtime_input), model(runtime_input))


def test_convert_transposed_conv1d():
    converted = _convert_and_compare(ConvTranspose1d(), torch.randn(1, 2, 8))
    [conv] = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    assert conv.args[1].meta["val"].shape == torch.Size([2, 4, 1, 3])
    assert conv.args[3:8] == ([1, 2], [0, 1], [1, 1], True, [0, 1])


@pytest.mark.parametrize(
    "model", [BufferConv1d(True), BufferConv1d(False), ConstantConv1d()]
)
def test_convert_lifted_weight_storage(model: torch.nn.Module):
    _convert_and_compare(model, torch.randn(1, 2, 8))


def test_convert_shared_conv1d_weight_once():
    converted = _convert_and_compare(SharedConv1d(), torch.randn(1, 2, 8))
    convs = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    assert len(convs) == 2
    assert convs[0].args[1] is convs[1].args[1]
    assert convs[0].args[1].meta["val"].dim() == 4


def test_skip_weight_with_non_conv1d_user():
    model = MixedUseWeight().eval()
    x = torch.randn(1, 2, 8)
    reference = model(x)
    edge = _edge(model, (x,))
    result = ConvertConv1dToConv2dPass(edge).call(edge.graph_module)

    assert not result.modified
    actual = edge.module()(x)
    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1])


def test_skip_weight_used_as_another_conv1d_input():
    model = WeightUsedAsConvInput().eval()
    x = torch.randn(1, 2, 8)
    reference = model(x)
    edge = _edge(model, (x,))
    converted = _transform(edge, ConvertConv1dToConv2dPass(edge))

    actual = converted.module()(x)
    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1])
    weight_ranks = sorted(
        node.args[1].meta["val"].dim()
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    )
    assert weight_ranks == [3, 4]


def test_skip_dynamic_weight():
    model = DynamicWeightConv1d().eval()
    inputs = (torch.randn(1, 2, 8), torch.randn(4, 2, 3))
    reference = model(*inputs)
    edge = _edge(model, inputs)

    result = ConvertConv1dToConv2dPass(edge).call(edge.graph_module)

    assert not result.modified
    torch.testing.assert_close(edge.module()(*inputs), reference)


def test_skip_unfolded_qdq_weight():
    edge = _edge(Conv1d(), (torch.randn(1, 2, 8),))
    [conv] = [
        node
        for node in edge.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]
    weight = conv.args[1]
    with edge.graph.inserting_before(conv):
        dequantized_weight = edge.graph.call_function(
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            args=(
                weight,
                torch.ones(4),
                torch.zeros(4, dtype=torch.int64),
                0,
                -128,
                127,
                torch.int8,
            ),
        )
    dequantized_weight.meta["val"] = weight.meta["val"]
    args = list(conv.args)
    args[1] = dequantized_weight
    conv.args = tuple(args)

    result = ConvertConv1dToConv2dPass(edge).call(edge.graph_module)

    assert not result.modified


def test_preserve_convolution_metadata():
    edge = _edge(Conv1d(), (torch.randn(1, 2, 8),))
    [conv] = [
        node
        for node in edge.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]
    metadata = {
        "input_qparams": {0: "input"},
        "output_qparams": {0: "output"},
        "custom": {"value": 1},
        "delegation_tag": "test_tag",
    }
    conv.meta.update(metadata)

    converted = _transform(edge, ConvertConv1dToConv2dPass(edge))
    [converted_conv] = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    for key, value in metadata.items():
        assert converted_conv.meta[key] == value


def test_set_param_tensor_get_attr_fallback():
    edge = _edge(Conv1d(), (torch.randn(1, 2, 8),))
    edge.graph_module.fallback_weight = torch.nn.Parameter(
        torch.randn(4, 2, 3), requires_grad=True
    )
    graph = torch.fx.Graph()
    weight_node = graph.get_attr("fallback_weight")
    replacement = torch.randn(4, 2, 1, 3)

    set_param_tensor(edge, weight_node, replacement)

    actual = edge.graph_module.fallback_weight
    assert isinstance(actual, torch.nn.Parameter)
    assert actual.requires_grad
    torch.testing.assert_close(actual, replacement)


def test_rank4_weight_with_singleton_args_is_not_converted():
    edge = _edge(torch.nn.Conv2d(2, 4, 3, padding=1), (torch.randn(1, 2, 8, 8),))
    [conv] = [
        node
        for node in edge.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]
    args = list(conv.args)
    args[3:6] = ([1], [1], [1])
    conv.args = tuple(args)

    result = ConvertConv1dToConv2dPass(edge).call(edge.graph_module)

    assert not result.modified
    assert conv.args[3:6] == ([1], [1], [1])


def test_conversion_is_idempotent():
    converted = _convert_and_compare(Conv1d(), torch.randn(1, 2, 8))

    result = ConvertConv1dToConv2dPass(converted).call(converted.graph_module)

    assert not result.modified
