# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from ctypes import c_float
from dataclasses import dataclass

import executorch.backends.espressif.ops.operators  # noqa: F401

import torch
from executorch.backends.espressif.utils.quantization import quantize_multiplier
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import get_constant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.fx import GraphModule


@dataclass(frozen=True)
class _QParams:
    scale: float
    zp: int


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


def _add_multipliers(
    input1: _QParams, input2: _QParams, output: _QParams
) -> tuple[int, ...] | None:
    left_shift = 20
    twice_max_scale = 2 * max(input1.scale, input2.scale)
    rescales = (
        input1.scale / twice_max_scale,
        input2.scale / twice_max_scale,
        twice_max_scale / ((1 << left_shift) * output.scale),
    )
    try:
        multipliers = [quantize_multiplier(rescale) for rescale in rescales]
    except ValueError:
        return None
    if any(not -30 <= shift <= 0 for _, shift in multipliers):
        return None
    max_float1 = c_float(
        max(abs(-128 - input1.zp), abs(127 - input1.zp)) * c_float(input1.scale).value
    ).value
    max_float2 = c_float(
        max(abs(-128 - input2.zp), abs(127 - input2.zp)) * c_float(input2.scale).value
    ).value
    if max_float1 + max_float2 > torch.finfo(torch.float32).max:
        return None
    return *multipliers[0], *multipliers[1], left_shift, *multipliers[2]


def _prepare_binary(node, params, target, rescale_fn) -> tuple | None:
    inputs = node.args[:2]
    values = [arg.meta.get("val") for arg in inputs] + [node.meta.get("val")]
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.int8
        and all(type(dim) is int for dim in value.shape)
        and 0 < value.numel() <= torch.iinfo(torch.int32).max
        and value.is_contiguous()
        for value in values
    ):
        return None
    if node.target == exir_ops.edge.fused_quant.mul.default:
        # ESP-NN repeats one trailing block of input2 across the full input1.
        if values[0].numel() < values[1].numel():
            inputs = inputs[::-1]
            params = [params[1], params[0], params[2]]
            values[0], values[1] = values[1], values[0]
        rank = max(values[0].dim(), values[1].dim())
        full_shape = (1,) * (rank - values[0].dim()) + tuple(values[0].shape)
        block_shape = tuple(values[1].shape)
        while block_shape and block_shape[0] == 1:
            block_shape = block_shape[1:]
        if values[2].shape != full_shape or (
            block_shape and full_shape[-len(block_shape) :] != block_shape
        ):
            return None
        if values[0].shape != values[1].shape:
            target = exir_ops.edge.esp_nn.quantized_mul_broadcast_int8.default
    elif not all(value.shape == values[0].shape for value in values[1:]):
        return None
    input1, input2, output = params
    # Portable float32 quantization multiplies by the reciprocal output scale.
    if not math.isfinite(c_float(1.0 / output.scale).value):
        return None
    rescale = rescale_fn(*params)
    if rescale is None:
        return None
    return target, (
        inputs[0],
        inputs[1],
        -input1.zp,
        -input2.zp,
        output.zp,
        *rescale,
    )


def _qparams(program, node, prefix):
    params = get_qparams_from_node(node, prefix)
    dtype = torch.int8 if prefix == "out" else torch.float32
    if params is None or not params.is_per_tensor() or params.dtype != dtype:
        return None
    if prefix != "out" and get_arg(node, prefix).meta["val"].dtype != torch.int8:
        return None
    if not -128 <= params.quant_min <= params.quant_max <= 127:
        raise ValueError(
            f"{node.name}: {prefix} has an invalid int8 quantization range"
        )
    scale = get_constant(program, params.scale)
    zero_point = get_constant(program, params.zero_point)
    if scale is None or zero_point is None:
        return None
    scale, zero_point = float(scale.item()), int(zero_point.item())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{node.name}: {prefix} scale must be finite and positive")
    if not -128 <= zero_point <= 127:
        raise ValueError(f"{node.name}: {prefix} has an invalid int8 zero point")
    if (params.quant_min, params.quant_max) != (-128, 127):
        return None
    return _QParams(scale, zero_point)


_BINARY_LOWERINGS = {
    exir_ops.edge.fused_quant.mul.default: (
        exir_ops.edge.esp_nn.quantized_mul_int8.default,
        _mul_multiplier,
    ),
    exir_ops.edge.fused_quant.add.default: (
        exir_ops.edge.esp_nn.quantized_add_int8.default,
        _add_multipliers,
    ),
}


class LowerToEsp(ExportedProgramPassBase):
    """Lower supported fused-quant operators to ESP-NN."""

    def call(self, exported_program):
        # The pass manager copies programs shallowly; decomposition also retires
        # qparam constants, so isolate both graph and constant dictionaries.
        graph = copy.deepcopy(exported_program.graph)
        for original, cloned in zip(exported_program.graph.nodes, graph.nodes):
            cloned.name = original.name
        module = GraphModule(exported_program.graph_module, graph)
        module.meta = exported_program.graph_module.meta.copy()
        exported_program._graph_module = module
        exported_program._graph_signature = copy.deepcopy(
            exported_program.graph_signature
        )
        exported_program._state_dict = exported_program.state_dict.copy()
        exported_program._constants = exported_program.constants.copy()

        modified = False
        for node in list(graph.nodes):
            lowering = _BINARY_LOWERINGS.get(node.target)
            if lowering is None:
                continue
            if (
                node.target == exir_ops.edge.fused_quant.add.default
                and get_arg(node, "alpha") != 1
            ):
                continue
            target, rescale_fn = lowering
            params = [
                _qparams(exported_program, node, prefix)
                for prefix in ("inp", "other", "out")
            ]
            if any(param is None for param in params):
                continue
            prepared = _prepare_binary(node, params, target, rescale_fn)
            if prepared is None:
                continue
            target, args = prepared
            with graph.inserting_before(node):
                replacement = graph.call_function(target, args)
            replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
