# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.operator_support.convolution_support import (
    ConvolutionSupported,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

# These tests exercise the operator-support predicate directly. Building small
# FX nodes keeps unsupported cases from reaching the lowering pipeline.
#
#     input ─┐
#            ├─ conv_transpose2d / transposed convolution ──► reject early
#     weight ┘        │
#                     ├─ dilation != 1
#                     └─ grouped + per-channel weight qparams
#
# If these cases are incorrectly marked supported, later rewrite/decomposition
# passes fail with opaque _ExportedProgramGraphPassAdapter errors instead of a
# clear partition rejection.


def _fake_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    with FakeTensorMode() as mode:
        return mode.from_tensor(torch.empty(shape))


def _make_conv_node(
    target,
    args,
    input_shape=(1, 3, 8, 8),
    weight_shape=(3, 3, 3, 3),
    per_channel_weight=False,
):
    graph = torch.fx.Graph()
    input_node = graph.placeholder("input")
    input_node.meta["val"] = _fake_tensor(input_shape)
    weight_node = graph.placeholder("weight")
    weight_node.meta["val"] = _fake_tensor(weight_shape)
    if per_channel_weight:
        scale_node = graph.placeholder("scale")
        scale_node.meta["val"] = _fake_tensor((weight_shape[0],))
        zero_point_node = graph.placeholder("zero_point")
        zero_point_node.meta["val"] = _fake_tensor((weight_shape[0],))
        weight_node = graph.call_function(
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            (weight_node, scale_node, zero_point_node, 0, -127, 127, torch.int8),
        )
        weight_node.meta["val"] = _fake_tensor(weight_shape)
    bias_node = graph.placeholder("bias")
    bias_node.meta["val"] = _fake_tensor((weight_shape[1],))
    node_args = {
        "input": input_node,
        "weight": weight_node,
        "bias": bias_node,
    }
    resolved_args = tuple(
        node_args[arg] if isinstance(arg, str) else arg for arg in args
    )
    node = graph.call_function(target, resolved_args)
    node.meta["val"] = _fake_tensor((1, weight_shape[1], 8, 8))
    return node


def _checker() -> ConvolutionSupported:
    return ConvolutionSupported(
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        WhyNoPartitionReporter(),
    )


def test_rejects_edge_transpose_convolution_with_dilation() -> None:
    # edge.aten.convolution.default represents transposed convolution via the
    # boolean transposed argument. Dilation is rejected here because
    # RewriteConvPass cannot lower dilated TRANSPOSE_CONV2D.
    node = _make_conv_node(
        exir_ops.edge.aten.convolution.default,
        (
            "input",
            "weight",
            "bias",
            [1, 1],
            [0, 0],
            [2, 1],
            True,
            [0, 0],
            1,
        ),
    )

    assert not _checker().is_node_supported({}, node)


def test_rejects_aten_conv_transpose2d_with_dilation() -> None:
    # torch.export may also preserve the explicit aten.conv_transpose2d.input
    # overload. It uses a different argument layout from edge.aten.convolution,
    # so cover it separately to keep the support-check indexing correct.
    node = _make_conv_node(
        torch.ops.aten.conv_transpose2d.input,
        (
            "input",
            "weight",
            "bias",
            [1, 1],
            [0, 0],
            [0, 0],
            1,
            [2, 1],
        ),
    )

    assert not _checker().is_node_supported({}, node)


def test_rejects_grouped_aten_conv_transpose2d_with_per_channel_weights() -> None:
    # Grouped transpose convolution is decomposed into per-group convolutions.
    # Per-channel weight qparams on aten.conv_transpose2d.input do not align with
    # that decomposition, so the node must be rejected before decomposition.
    node = _make_conv_node(
        torch.ops.aten.conv_transpose2d.input,
        (
            "input",
            "weight",
            "bias",
            [1, 1],
            [0, 0],
            [0, 0],
            3,
            [1, 1],
        ),
    )
    node.meta["input_qparams"] = {
        1: QuantArgs(
            [1.0, 1.0, 1.0], [0, 0, 0], -127, 127, torch.int8, per_channel=True
        )
    }

    assert not _checker().is_node_supported({}, node)


def test_rejects_grouped_edge_transpose_convolution_with_per_channel_dq_weight() -> (
    None
):
    # The flow-suite partitioner sees converted quantized models before
    # FoldAndAnnotateQParamsPass adds input_qparams. At that point per-channel
    # weight quantization is visible through the dequantize_per_channel weight
    # input, so reject from that graph shape too.
    node = _make_conv_node(
        exir_ops.edge.aten.convolution.default,
        (
            "input",
            "weight",
            "bias",
            [1, 1],
            [0, 0],
            [1, 1],
            True,
            [0, 0],
            3,
        ),
        per_channel_weight=True,
    )

    assert not _checker().is_node_supported({}, node)


def test_rejects_grouped_edge_transpose_convolution_with_per_channel_weights() -> None:
    # The same grouped/per-channel restriction applies to the edge dialect form
    # of transposed convolution. This covers the path where export/decomposition
    # has normalized conv_transpose2d into edge.aten.convolution.default.
    node = _make_conv_node(
        exir_ops.edge.aten.convolution.default,
        (
            "input",
            "weight",
            "bias",
            [1, 1],
            [0, 0],
            [1, 1],
            True,
            [0, 0],
            3,
        ),
    )
    node.meta["input_qparams"] = {
        1: QuantArgs(
            [1.0, 1.0, 1.0], [0, 0, 0], -127, 127, torch.int8, per_channel=True
        )
    }

    assert not _checker().is_node_supported({}, node)


def test_accepts_aten_conv_transpose2d_without_unsupported_options() -> None:
    # A plain 2D transpose convolution with groups=1 and dilation=1 is still
    # supported. This guards against the rejection checks becoming too broad.
    node = _make_conv_node(
        torch.ops.aten.conv_transpose2d.input,
        (
            "input",
            "weight",
            "bias",
            [1, 1],
            [0, 0],
            [0, 0],
            1,
            [1, 1],
        ),
    )

    assert _checker().is_node_supported({}, node)
