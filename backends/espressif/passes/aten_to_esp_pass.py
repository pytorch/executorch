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
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.constant_prop_pass import get_constant_placeholder_dict
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
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


def _add_multipliers(
    input1: _QParams, input2: _QParams, output: _QParams
) -> tuple[int, ...] | None:
    # Portable float32 quantization multiplies by the reciprocal output scale.
    if not math.isfinite(c_float(1.0 / output.scale).value):
        return None
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


def _activation_kind(node: Node) -> str | None:
    if node.target == exir_ops.edge.aten.relu.default:
        return "relu"
    if node.target == exir_ops.edge.aten.relu6.default:
        return "relu6"
    if node.target == exir_ops.edge.aten.hardtanh.default:
        minimum = node.args[1] if len(node.args) > 1 else node.kwargs.get("min_val", -1)
        maximum = node.args[2] if len(node.args) > 2 else node.kwargs.get("max_val", 1)
        if (minimum, maximum) == (0, 6):
            return "relu6"
    return None


def _binary_args(node: Node, quantize: Node, rescale_fn) -> tuple | None:
    if node.kwargs.get("alpha", 1) != 1:
        return None
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
    ):
        return None
    if node.target == exir_ops.edge.aten.mul.Tensor:
        # ESP-NN repeats one trailing block of input2 across the full input1.
        if values[0].numel() < values[1].numel():
            inputs = inputs[::-1]
            params[0], params[1] = params[1], params[0]
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
    elif not all(value.shape == values[0].shape for value in values[1:]):
        return None
    if not all(
        isinstance(arg.meta.get("val"), torch.Tensor)
        and arg.meta["val"].dtype == torch.float32
        for arg in (*inputs, node, quantize.args[0])
    ):
        return None
    rescale = rescale_fn(*params)
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


def _quantized_output(node: Node) -> tuple[Node, str | None] | None:
    quantizers = [
        (user, None)
        for user in node.users
        if user.target == _QUANTIZE and user.args[0] is node
    ]
    for user in node.users:
        activation = _activation_kind(user)
        if activation is not None and user.args[0] is node:
            quantizers.extend(
                (quantize, activation)
                for quantize in user.users
                if quantize.target == _QUANTIZE and quantize.args[0] is user
            )
    if len(quantizers) != 1:
        return None
    return quantizers[0]


def _activation_bounds(quantize: Node, activation: str | None) -> tuple[int, int]:
    minimum = quantize.args[2] if activation is not None else -128
    maximum = (
        int(_QUANTIZE(torch.tensor(6.0, dtype=torch.float32), *quantize.args[1:]))
        if activation == "relu6"
        else 127
    )
    return minimum, maximum


def _lower_binary(node: Node, target, rescale_fn) -> bool:
    output = _quantized_output(node)
    if output is None:
        return False
    quantize, activation = output
    args = _binary_args(node, quantize, rescale_fn)
    if args is None:
        return False
    with node.graph.inserting_before(quantize):
        replacement = node.graph.call_function(
            target, (*args, *_activation_bounds(quantize, activation))
        )
    replacement.meta = quantize.meta.copy()
    # Replace the quantized branch; direct float consumers still need the ATen op.
    quantize.replace_all_uses_with(replacement)
    node.graph.erase_node(quantize)
    return True


def _linear_parameters(
    input_params: _QParams,
    output_params: _QParams,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple | None:
    weight = weight.detach().cpu()
    scales = scales.detach().cpu().to(torch.float64)
    bias = bias.detach().cpu() if bias is not None else None
    try:
        rescales = [
            quantize_multiplier(input_params.scale * scale / output_params.scale)
            for scale in scales.tolist()
        ]
    except ValueError:
        return None
    if any(shift < -30 for _, shift in rescales):
        return None
    max_input = max(abs(-128 - input_params.zp), abs(127 - input_params.zp))
    weight_sums = weight.to(torch.int64).abs().sum(dim=1)
    limits = max_input * weight_sums
    if bias is not None:
        limits += bias.to(torch.int64).abs()
    if any(
        int(limit) << max(shift, 0)
        > torch.iinfo(torch.int32).max - abs(output_params.zp)
        for limit, (_, shift) in zip(limits, rescales)
    ):
        return None
    max_float_input = c_float(max_input * c_float(input_params.scale).value).value
    float_weights = weight.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)
    max_float_outputs = max_float_input * float_weights.abs().to(torch.float64).sum(
        dim=1
    )
    if bias is not None:
        bias_scales = torch.tensor(input_params.scale, dtype=torch.float32) * scales.to(
            torch.float32
        )
        max_float_outputs += (
            (bias.to(torch.float32) * bias_scales).abs().to(torch.float64)
        )
    # Bound float32 accumulation rounding, including the bias addition.
    max_float_outputs /= 1 - (weight.shape[1] + 1) * torch.finfo(torch.float32).eps
    if (
        not math.isfinite(c_float(1.0 / output_params.scale).value)
        or not torch.all(torch.isfinite(max_float_outputs))
        or torch.any(max_float_outputs > torch.finfo(torch.float32).max)
    ):
        return None
    multipliers, shifts = zip(*rescales)
    return (
        torch.tensor(multipliers, dtype=torch.int32),
        torch.tensor(shifts, dtype=torch.int32),
    )


def _linear_bias(
    node: Node, input_scale: float, weight_scales: torch.Tensor, constants: dict
) -> tuple[Node, torch.Tensor] | None:
    if (
        node.target != exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
        or node.args[3] != 0
        or node.args[6] != torch.int32
    ):
        return None
    bias, scales, zero_points = (constants.get(arg) for arg in node.args[:3])
    if (
        bias is None
        or bias.dtype != torch.int32
        or not bias.is_contiguous()
        or bias.shape != weight_scales.shape
        or scales is None
        or not torch.equal(scales, weight_scales * input_scale)
        or zero_points is None
        or zero_points.shape != bias.shape
        or torch.any(zero_points != 0)
    ):
        return None
    return node.args[0], bias


def _linear_args(node: Node, quantize: Node, constants: dict) -> tuple | None:
    if any(node.kwargs.get(key, 1) != 1 for key in ("alpha", "beta")):
        return None
    input_node, transpose = node.args[-2:]
    if (
        input_node.target != _DEQUANTIZE
        or transpose.target != exir_ops.edge.aten.permute_copy.default
        or tuple(transpose.args[1]) != (1, 0)
    ):
        return None
    weight_dq = transpose.args[0]
    if (
        weight_dq.target
        != exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
        or weight_dq.args[3] != 0
        or weight_dq.args[6] != torch.int8
    ):
        return None
    weight_node, scale_node, zp_node = weight_dq.args[:3]
    weight, scales, zero_points = (
        constants.get(arg) for arg in (weight_node, scale_node, zp_node)
    )
    if (
        weight is None
        or weight.dtype != torch.int8
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.numel() > torch.iinfo(torch.int32).max
        or any(not 0 < dim <= 65535 for dim in weight.shape)
        or scales is None
        or scales.dtype != torch.float32
        or scales.shape != (weight.shape[0],)
        or zero_points is None
        or zero_points.shape != scales.shape
        or torch.any(zero_points != 0)
        or not torch.all(torch.isfinite(scales) & (scales > 0))
    ):
        return None
    input_params, output_params = _qparams(input_node), _qparams(quantize)
    if input_params is None or output_params is None:
        return None
    bias_node, bias = None, None
    if node.target == exir_ops.edge.aten.addmm.default:
        bias_args = _linear_bias(node.args[0], input_params.scale, scales, constants)
        if bias_args is None:
            return None
        bias_node, bias = bias_args
    input_value = input_node.args[0].meta.get("val")
    output_value = quantize.meta.get("val")
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.int8
        and value.dim() == 2
        and all(type(dim) is int for dim in value.shape)
        and 0 < value.numel() <= torch.iinfo(torch.int32).max
        and value.is_contiguous()
        for value in (input_value, output_value)
    ):
        return None
    if input_value.shape[1] != weight.shape[1] or output_value.shape != (
        input_value.shape[0],
        weight.shape[0],
    ):
        return None
    if not all(
        isinstance(arg.meta.get("val"), torch.Tensor)
        and arg.meta["val"].dtype == torch.float32
        for arg in (input_node, weight_dq, node, quantize.args[0])
    ):
        return None
    parameters = _linear_parameters(input_params, output_params, weight, scales, bias)
    if parameters is None:
        return None
    return (
        input_node.args[0],
        weight_node,
        bias_node,
        *parameters,
        -input_params.zp,
        output_params.zp,
    )


def _lower_linear(node: Node, program: ExportedProgram, constants: dict) -> bool:
    output = _quantized_output(node)
    if output is None:
        return False
    quantize, activation = output
    args = _linear_args(node, quantize, constants)
    if args is None:
        return False
    input_node, weight_node, bias_node, *parameters, input_offset, output_offset = args
    first_input = next(n for n in node.graph.nodes if n.op == "placeholder")
    with node.graph.inserting_before(first_input):
        parameter_nodes = [
            create_constant_placeholder(
                program,
                node.graph,
                f"_esp_{node.name}_{name}",
                InputKind.CONSTANT_TENSOR,
                value,
            )
            for name, value in zip(("multiplier", "shift"), parameters)
        ]
    with node.graph.inserting_before(quantize):
        replacement = node.graph.call_function(
            exir_ops.edge.esp_nn.quantized_linear_int8.default,
            (
                input_node,
                weight_node,
                bias_node,
                *parameter_nodes,
                input_offset,
                output_offset,
                *_activation_bounds(quantize, activation),
            ),
        )
    replacement.meta = quantize.meta.copy()
    quantize.replace_all_uses_with(replacement)
    node.graph.erase_node(quantize)
    return True


_LOWERINGS = {
    exir_ops.edge.aten.mul.Tensor: (
        exir_ops.edge.esp_nn.quantized_mul_int8.default,
        _mul_multiplier,
    ),
    exir_ops.edge.aten.add.Tensor: (
        exir_ops.edge.esp_nn.quantized_add_int8.default,
        _add_multipliers,
    ),
}


class AtenToEspPass(PassBase):
    """Fold supported Q/DQ patterns directly into ESP-NN operator calls."""

    def __init__(self, exported_program: ExportedProgram | None = None):
        self.exported_program = exported_program
        self.constants = (
            get_constant_placeholder_dict(exported_program)
            if exported_program is not None
            else {}
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            lowering = _LOWERINGS.get(node.target)
            if lowering is not None:
                modified |= _lower_binary(node, *lowering)
            elif self.exported_program is not None and node.target in (
                exir_ops.edge.aten.addmm.default,
                exir_ops.edge.aten.mm.default,
            ):
                modified |= _lower_linear(node, self.exported_program, self.constants)
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
