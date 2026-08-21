# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


class CortexMExplicitLayoutPass(ToContiguousChannelsLastPass):
    """Configure the common explicit-layout pipeline for Cortex-M kernels."""

    def __init__(
        self,
        exported_program: ExportedProgram,
        strict: bool = False,
    ) -> None:
        super().__init__(
            exported_program,
            op_map={
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
            },
            can_propagate=_can_propagate,
            strict=strict,
        )
