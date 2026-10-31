# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from ctypes import c_float
from dataclasses import dataclass

import executorch.backends.espressif.ops.operators  # noqa: F401
import torch
from executorch.backends.espressif.utils.quantization import quantize_multiplier
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


@dataclass(frozen=True)
class _QParams:
    scale: float
    zp: int


def _qparams(node: Node) -> _QParams | None:
    if len(node.args) < 6:
        return None
    scale, zp, qmin, qmax, dtype = node.args[1:6]
    if (
        dtype != torch.int8
        or any(type(value) is not int for value in (zp, qmin, qmax))
        or type(scale) not in (float, int)
    ):
        return None
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{node.name}: quantization scale must be finite and positive")
    if not -128 <= qmin <= zp <= qmax <= 127:
        raise ValueError(f"{node.name}: invalid int8 quantization range or zero point")
    if (qmin, qmax) != (-128, 127):
        return None
    return _QParams(float(scale), zp)


def _mul_multiplier(
    input1: _QParams, input2: _QParams, output: _QParams
) -> tuple[int, int] | None:
    rescale = input1.scale * input2.scale / output.scale
    if not math.isfinite(rescale) or rescale <= 0:
        return None
    try:
        multiplier, shift = quantize_multiplier(rescale)
    except ValueError:
        return None
    # ESP-NN forms (1 << -shift) - 1 with signed int32 arithmetic.
    if shift < -30:
        return None
    max_input1 = max(abs(-128 - input1.zp), abs(127 - input1.zp))
    max_input2 = max(abs(-128 - input2.zp), abs(127 - input2.zp))
    max_product = max_input1 * max_input2
    if max_product << max(shift, 0) > torch.iinfo(torch.int32).max - abs(output.zp):
        return None
    # Account for float32 rounding of both the scales and dequantized inputs.
    max_float1 = c_float(max_input1 * c_float(input1.scale).value).value
    max_float2 = c_float(max_input2 * c_float(input2.scale).value).value
    if (
        not math.isfinite(max_float1)
        or not math.isfinite(max_float2)
        or max_float1 * max_float2 > torch.finfo(torch.float32).max
    ):
        return None
    return multiplier, shift


def _mul_args(node: Node, quantize: Node) -> tuple | None:
    inputs = node.args[:2]
    if len(inputs) != 2 or not all(
        isinstance(arg, Node) and arg.target == _DEQUANTIZE for arg in inputs
    ):
        return None
    params = [_qparams(arg) for arg in (*inputs, quantize)]
    if any(param is None for param in params):
        return None
    values = [arg.args[0].meta.get("val") for arg in inputs] + [
        quantize.meta.get("val")
    ]
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.int8
        and all(type(dim) is int for dim in value.shape)
        and 0 < value.numel() <= torch.iinfo(torch.int32).max
        and value.is_contiguous()
        for value in values
    ) or not all(value.shape == values[0].shape for value in values[1:]):
        return None
    if not all(
        isinstance(arg.meta.get("val"), torch.Tensor)
        and arg.meta["val"].dtype == torch.float32
        for arg in (*inputs, node)
    ):
        return None
    rescale = _mul_multiplier(*params)
    if rescale is None:
        return None
    input1, input2, output = params
    return (
        inputs[0].args[0],
        inputs[1].args[0],
        -input1.zp,
        -input2.zp,
        output.zp,
        *rescale,
    )


def _lower_mul(node: Node) -> bool:
    quantizers = [
        user for user in node.users if user.target == _QUANTIZE and user.args[0] is node
    ]
    if len(quantizers) != 1:
        return False
    quantize = quantizers[0]
    args = _mul_args(node, quantize)
    if args is None:
        return False
    with node.graph.inserting_before(quantize):
        replacement = node.graph.call_function(
            exir_ops.edge.esp_nn.quantized_mul_int8.default, args
        )
    replacement.meta = quantize.meta.copy()
    # Replace the quantized branch; direct float consumers still need the ATen op.
    quantize.replace_all_uses_with(replacement)
    node.graph.erase_node(quantize)
    return True


_LOWERINGS = {exir_ops.edge.aten.mul.Tensor: _lower_mul}


class AtenToEspPass(PassBase):
    """Fold supported Q/DQ patterns directly into ESP-NN operator calls."""

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            lowering = _LOWERINGS.get(node.target)
            if lowering is not None:
                modified |= lowering(node)
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
