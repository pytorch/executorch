# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.transforms.channels_last_ops  # noqa: F401

import torch

from executorch.backends.transforms.replace_ops_with_channels_last_variants import (
    ChannelsLastOpSpec,
)
from executorch.backends.transforms.to_contiguous_channels_last_pass import (
    ToContiguousChannelsLastPass,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


def _is_rank4(node: torch.fx.Node) -> bool:
    return len(node.meta["val"].shape) == 4


def _has_input_qparams(node: torch.fx.Node) -> bool:
    return bool(node.meta.get("input_qparams"))


def _has_per_tensor_qparam(node: torch.fx.Node, key: str, index: int) -> bool:
    qparam = node.meta.get(key, {}).get(index)
    return qparam is not None and not getattr(qparam, "per_channel", False)


def _has_input_and_output_qparams(node: torch.fx.Node) -> bool:
    return _has_per_tensor_qparam(node, "input_qparams", 0) and _has_per_tensor_qparam(
        node, "output_qparams", 0
    )


def _supports_avg_pool2d(node: torch.fx.Node) -> bool:
    divisor_override = node.args[6] if len(node.args) > 6 else None
    return (
        _is_rank4(node)
        and _has_input_and_output_qparams(node)
        and divisor_override is None
    )


def _to_pair(value, default: tuple[int, int]) -> tuple[int, int]:
    if value is None or value == []:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return (int(value[0]), int(value[0]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return default


def _supports_max_pool2d(node: torch.fx.Node) -> bool:
    if not _is_rank4(node) or not _has_per_tensor_qparam(node, "input_qparams", 0):
        return False
    if (
        node.meta.get("custom", {})
        .get("cortex_m", {})
        .get("skip_quantized_max_pool2d", False)
    ):
        return False

    dilation = _to_pair(node.args[4] if len(node.args) > 4 else None, (1, 1))
    ceil_mode = bool(node.args[5]) if len(node.args) > 5 else False
    if dilation != (1, 1) or ceil_mode:
        return False

    input_qparams = node.meta["input_qparams"].get(0)
    output_qparams = node.meta.get("output_qparams", {}).get(0)
    if input_qparams is None or output_qparams is None:
        return input_qparams is not None
    return (
        not getattr(output_qparams, "per_channel", False)
        and abs(float(input_qparams.scale) - float(output_qparams.scale)) <= 1e-6
        and int(input_qparams.zp) == int(output_qparams.zp)
    )


_CORTEX_M_EXPLICIT_LAYOUT_OP_MAP = {
    exir_ops.edge.aten.convolution.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.convolution.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=lambda node: (
            _is_rank4(node) and _has_input_and_output_qparams(node)
        ),
    ),
    exir_ops.edge.aten.avg_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.avg_pool2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_supports_avg_pool2d,
    ),
    exir_ops.edge.aten.max_pool2d.default: ChannelsLastOpSpec(
        target=exir_ops.edge.channels_last.max_pool2d.default,
        input_indices=[0],
        output_indices=[0],
        filter_fn=_supports_max_pool2d,
    ),
}

_SOURCE_ANCHORS_BY_EDGE_TARGET = {
    exir_ops.edge.aten.convolution.default: frozenset(
        {
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv_transpose2d.input,
        }
    ),
    exir_ops.edge.aten.avg_pool2d.default: frozenset(
        {torch.ops.aten.avg_pool2d.default}
    ),
    exir_ops.edge.aten.max_pool2d.default: frozenset(
        {
            torch.ops.aten.max_pool2d.default,
            torch.ops.aten.max_pool2d_with_indices.default,
        }
    ),
}
assert _SOURCE_ANCHORS_BY_EDGE_TARGET.keys() == _CORTEX_M_EXPLICIT_LAYOUT_OP_MAP.keys()

CORTEX_M_EXPLICIT_LAYOUT_SOURCE_ANCHORS = frozenset(
    target for targets in _SOURCE_ANCHORS_BY_EDGE_TARGET.values() for target in targets
)

CORTEX_M_EXPLICIT_LAYOUT_TRANSPARENT_OPS = frozenset(
    {
        operator.getitem,
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp_.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardsigmoid_.default,
    }
)


def _can_propagate(node: torch.fx.Node) -> bool:
    if node.target == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default:
        return False
    if node.target not in {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    }:
        return True
    if len(node.args) < 2:
        return False
    input1, input2 = node.args[:2]
    if not isinstance(input1, torch.fx.Node) or not isinstance(input2, torch.fx.Node):
        return False
    tensor1 = input1.meta.get("val")
    tensor2 = input2.meta.get("val")
    if tensor1 is None or tensor2 is None:
        return False
    return tensor1.shape == tensor2.shape or _has_input_and_output_qparams(node)


def _is_nhwc_channel_broadcast(node: torch.fx.Node) -> bool:
    input1, input2 = node.args[:2]
    if not isinstance(input1, torch.fx.Node) or not isinstance(input2, torch.fx.Node):
        return False
    tensor1 = input1.meta.get("val")
    tensor2 = input2.meta.get("val")
    if tensor1 is None or tensor2 is None or tensor1.dim() != 4 or tensor2.dim() != 4:
        return False
    return tensor1.size(3) == tensor2.size(3) and (
        tensor1.numel() == tensor1.size(3) or tensor2.numel() == tensor2.size(3)
    )


class CortexMExplicitLayoutPass(ToContiguousChannelsLastPass):
    """Configure the common explicit-layout pipeline for Cortex-M kernels."""

    def __init__(
        self,
        exported_program: ExportedProgram,
        strict: bool = False,
    ) -> None:
        super().__init__(
            exported_program,
            op_map=dict(_CORTEX_M_EXPLICIT_LAYOUT_OP_MAP),
            can_propagate=_can_propagate,
            layout_pad_target=exir_ops.edge.channels_last.constant_pad_nd.default,
            strict=strict,
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        result = super().call(graph_module)
        for node in result.graph_module.graph.nodes:
            if node.target not in {
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.mul.Tensor,
            } or not _has_input_and_output_qparams(node):
                continue
            input1, input2 = node.args[:2]
            if not isinstance(input1, torch.fx.Node) or not isinstance(
                input2, torch.fx.Node
            ):
                continue
            if input1.meta["val"].shape == input2.meta["val"].shape:
                continue
            if not _is_nhwc_channel_broadcast(node):
                raise RuntimeError(
                    f"Quantized channel-broadcast node {node.name} did not join "
                    "an explicit NHWC layout region."
                )
        return result
