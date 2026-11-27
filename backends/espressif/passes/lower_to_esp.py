# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
from ctypes import c_float
from dataclasses import dataclass
from functools import partial

import executorch.backends.espressif.ops.operators  # noqa: F401

import torch
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.utils.quantization import quantize_multiplier
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import add_constant, get_constant
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes import make_alloc_node
from executorch.exir.passes.constant_prop_pass import get_propagated_const_tensor_dict
from torch.export.graph_signature import InputKind


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
    input1: _QParams,
    input2: _QParams,
    output: _QParams,
    *,
    subtract: bool = False,
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
    input2_multiplier, input2_shift = multipliers[1]
    return (
        *multipliers[0],
        -input2_multiplier if subtract else input2_multiplier,
        input2_shift,
        left_shift,
        *multipliers[2],
    )


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


def _qparams(program, node, prefix, *, allow_output_clamp=False):
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
    if not (prefix == "out" and allow_output_clamp) and (
        params.quant_min,
        params.quant_max,
    ) != (-128, 127):
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
    exir_ops.edge.fused_quant.sub.default: (
        exir_ops.edge.esp_nn.quantized_add_int8.default,
        partial(_add_multipliers, subtract=True),
    ),
}


_LUT_OPS = {
    exir_ops.edge.fused_quant.sigmoid.default,
    exir_ops.edge.fused_quant.tanh.default,
    exir_ops.edge.fused_quant.gelu.default,
    exir_ops.edge.fused_quant.silu.default,
    exir_ops.edge.fused_quant.hardswish.default,
    exir_ops.edge.fused_quant.hardsigmoid.default,
    exir_ops.edge.fused_quant.relu.default,
    exir_ops.edge.fused_quant.hard_tanh.default,
}


def _lower_lut(program, node) -> bool:
    inp = get_arg(node, "inp", torch.fx.Node)
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.int8
        and all(type(dim) is int for dim in value.shape)
        and 0 < value.numel() <= torch.iinfo(torch.int32).max
        and value.is_contiguous()
        for value in (inp.meta.get("val"), node.meta.get("val"))
    ):
        return False
    if any(_qparams(program, node, prefix) is None for prefix in ("inp", "out")):
        return False

    # ESP-NN indexes the table by casting int8 input to uint8.
    codes = torch.arange(256, dtype=torch.int32).to(torch.int8)
    args, kwargs = torch.fx.map_arg(
        (node.args, node.kwargs),
        lambda arg: codes if arg is inp else get_constant(program, arg),
    )
    table = node.target(*args, **kwargs)
    table_node = add_constant(
        program, f"{node.name}_table", table, node, InputKind.CONSTANT_TENSOR
    )
    with node.graph.inserting_before(node):
        replacement = node.graph.call_function(
            exir_ops.edge.esp_nn.quantized_lut_int8.default, (inp, table_node)
        )
    replacement.meta = node.meta.copy()
    node.replace_all_uses_with(replacement)
    node.graph.erase_node(node)
    return True


def _pool_kernel_args(program, node, output_value) -> tuple | None:
    inp = get_arg(node, "inp", torch.fx.Node)
    input_value = inp.meta.get("val")
    if not all(
        isinstance(value, torch.Tensor)
        and value.dim() == 4
        and value.dtype == torch.int8
        and value.is_contiguous()
        and all(type(dim) is int and 0 < dim <= 65535 for dim in value.shape)
        and value.numel() <= torch.iinfo(torch.int32).max
        for value in (input_value, output_value)
    ):
        return None
    # Select the S3 assembly kernels, which require four-channel alignment.
    if input_value.shape[3] % 4:
        return None
    input_params = _qparams(program, node, "inp")
    if input_params is None or input_params != _qparams(program, node, "out"):
        return None
    max_input = max(abs(-128 - input_params.zp), abs(127 - input_params.zp))
    if not math.isfinite(c_float(1.0 / input_params.scale).value) or not math.isfinite(
        c_float(max_input * c_float(input_params.scale).value).value
    ):
        return None
    kernel = get_arg(node, "kernel_size")
    stride = get_arg(node, "stride") or kernel
    pairs = []
    for arg in (kernel, stride, get_arg(node, "padding")):
        if not isinstance(arg, (list, tuple)) or len(arg) not in (1, 2):
            return None
        pair = list(arg) * (2 if len(arg) == 1 else 1)
        if any(type(dim) is not int or not 0 <= dim <= 65535 for dim in pair):
            return None
        pairs.append(pair)
    kernel, stride, padding = pairs
    if get_arg(node, "ceil_mode") or min(*kernel, *stride) == 0:
        return None
    if any(pad > window // 2 for pad, window in zip(padding, kernel)):
        return None
    if (
        input_value.shape[0] != output_value.shape[0]
        or input_value.shape[3] != output_value.shape[3]
    ):
        return None
    if any(
        (size + 2 * pad - window) // step + 1 != out_size
        for size, pad, window, step, out_size in zip(
            input_value.shape[1:3], padding, kernel, stride, output_value.shape[1:3]
        )
    ):
        return None
    return inp, kernel, stride, padding


def _lower_max_pool2d(program, node) -> bool:
    if get_arg(node, "dilation") not in ([1], [1, 1], (1,), (1, 1)):
        return False
    values = list(node.users)
    if not values or any(
        user.target != operator.getitem or user.args[1] != 0 for user in values
    ):
        return False
    args = _pool_kernel_args(program, node, node.meta["val"][0])
    if args is None:
        return False
    graph = node.graph
    with graph.inserting_before(node):
        replacement = graph.call_function(
            exir_ops.edge.esp_nn.quantized_max_pool2d_int8.default,
            args,
        )
    replacement.meta = values[0].meta.copy()
    for value in values:
        value.replace_all_uses_with(replacement)
        graph.erase_node(value)
    graph.erase_node(node)
    return True


def _lower_avg_pool2d(program, node) -> bool:
    if get_arg(node, "divisor_override") is not None:
        return False
    args = _pool_kernel_args(program, node, node.meta["val"])
    if args is None:
        return False
    inp, kernel, _, padding = args
    if get_arg(node, "count_include_pad") and any(padding):
        return False
    height, width = inp.meta["val"].shape[1:3]
    count = min(height, kernel[0]) * min(width, kernel[1])
    # The vendor adds a signed rounding nudge to the int32 sum.
    if 128 * count + count // 2 > torch.iinfo(torch.int32).max:
        return False
    graph = node.graph
    with graph.inserting_before(node):
        replacement = graph.call_function(
            exir_ops.edge.esp_nn.quantized_avg_pool2d_int8.default, args
        )
    replacement.meta = node.meta.copy()
    node.replace_all_uses_with(replacement)
    graph.erase_node(node)
    return True


def _weighted_multipliers(input_params, output_params, weight, scales, bias):
    rescales = [
        quantize_multiplier(input_params.scale * scale / output_params.scale)
        for scale in scales.flatten().tolist()
    ]
    if any(shift < -30 for _, shift in rescales):
        raise ValueError("requantization shift is outside the ESP-NN range")
    max_input = max(abs(-128 - input_params.zp), abs(127 - input_params.zp))
    limits = max_input * weight.to(torch.int64).abs().sum(dim=1)
    if bias is not None:
        limits += bias.to(torch.int64).abs()
    if any(
        int(limit) << max(shift, 0)
        > torch.iinfo(torch.int32).max - abs(output_params.zp)
        for limit, (_, shift) in zip(limits, rescales)
    ):
        raise ValueError("accumulation or requantization may overflow int32")
    max_float_input = c_float(max_input * c_float(input_params.scale).value).value
    float_weights = weight.to(torch.float32) * scales
    max_float_outputs = max_float_input * float_weights.abs().to(torch.float64).sum(
        dim=1
    )
    if bias is not None:
        bias_scales = input_params.scale * scales.flatten()
        max_float_outputs += (
            (bias.to(torch.float32) * bias_scales).abs().to(torch.float64)
        )
    # Bound float32 accumulation rounding, including the bias addition.
    rounding_bound = 1 - (weight.shape[1] + 1) * torch.finfo(torch.float32).eps
    if rounding_bound <= 0:
        raise ValueError("the float32 reference may overflow")
    max_float_outputs /= rounding_bound
    if (
        not math.isfinite(c_float(1.0 / output_params.scale).value)
        or not torch.all(torch.isfinite(max_float_outputs))
        or torch.any(max_float_outputs > torch.finfo(torch.float32).max)
    ):
        raise ValueError("the float32 reference may overflow")
    multipliers, shifts = zip(*rescales)
    return torch.tensor(multipliers, dtype=torch.int32), torch.tensor(
        shifts, dtype=torch.int32
    )


def _linear_kernel_args(program, node, constants):
    inp, weight_node, bias_node = (
        get_arg(node, name) for name in ("inp", "weight", "bias")
    )
    input_value, output_value = inp.meta.get("val"), node.meta.get("val")
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.int8
        and value.dim() >= 1
        and all(type(dim) is int for dim in value.shape)
        and 0 < value.numel() <= torch.iinfo(torch.int32).max
        and value.is_contiguous()
        for value in (input_value, output_value)
    ):
        raise ValueError(
            "input and output must be nonempty, static, contiguous int8 tensors"
        )
    input_params, output_params = (
        _qparams(program, node, name, allow_output_clamp=True)
        for name in ("inp", "out")
    )
    if input_params is None or output_params is None:
        raise ValueError("input and output require constant per-tensor int8 qparams")
    weight = constants.get(weight_node)
    if (
        weight is None
        or weight.dtype != torch.int8
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.numel() > torch.iinfo(torch.int32).max
        or any(not 0 < dim <= 65535 for dim in weight.shape)
    ):
        raise ValueError(
            "weight must be a constant contiguous int8 matrix with dimensions in [1, 65535]"
        )
    if input_value.shape[-1] != weight.shape[1] or output_value.shape != (
        *input_value.shape[:-1],
        weight.shape[0],
    ):
        raise ValueError("input, weight and output shapes do not match linear")
    multiplier_node, shift_node = _weighted_constants(
        program, node, constants, weight, input_params, output_params
    )
    return (
        inp,
        weight_node,
        bias_node,
        multiplier_node,
        shift_node,
        -input_params.zp,
        output_params.zp,
        get_arg(node, "out_quant_min"),
        get_arg(node, "out_quant_max"),
    )


def _weighted_constants(program, node, constants, weight, input_params, output_params):
    bias_node = get_arg(node, "bias")
    weight_params = get_qparams_from_node(node, "weight")
    if weight_params is None or weight_params.dtype != torch.float32:
        raise ValueError(
            "weight requires float32 dequantization with per-output-channel scales"
        )
    scales, zero_points = (
        constants.get(arg) for arg in (weight_params.scale, weight_params.zero_point)
    )
    if (
        scales is None
        or scales.dtype != torch.float32
        or scales.shape != (weight.shape[0],) + (1,) * (weight.dim() - 1)
        or not torch.all(torch.isfinite(scales) & (scales > 0))
        or zero_points is None
        or zero_points.shape != scales.shape
        or torch.any(zero_points != 0)
    ):
        raise ValueError(
            "weight requires constant per-output-channel scales and zero zero-points"
        )
    bias = None
    if bias_node is not None:
        bias = constants.get(bias_node)
        bias_params = get_qparams_from_node(node, "bias")
        if (
            bias is None
            or bias.dtype != torch.int32
            or not bias.is_contiguous()
            or bias.shape != (weight.shape[0],)
            or bias_params is None
            or bias_params.dtype != torch.float32
        ):
            raise ValueError(
                "bias must be a constant int32 vector with derived qparams"
            )
        bias_scales, bias_zero_points = (
            constants.get(arg) for arg in (bias_params.scale, bias_params.zero_point)
        )
        if (
            bias_scales is None
            or not torch.equal(bias_scales, input_params.scale * scales.flatten())
            or bias_zero_points is None
            or bias_zero_points.shape != bias.shape
            or torch.any(bias_zero_points != 0)
        ):
            raise ValueError(
                "bias scales must equal input scale times weight scales, with zero zero-points"
            )
    multipliers, shifts = _weighted_multipliers(
        input_params,
        output_params,
        weight.flatten(1),
        scales.flatten().reshape(-1, 1),
        bias,
    )
    multiplier_node, shift_node = (
        add_constant(
            program, f"{node.name}_{name}", tensor, node, InputKind.CONSTANT_TENSOR
        )
        for name, tensor in (("multiplier", multipliers), ("shift", shifts))
    )
    return multiplier_node, shift_node


def _conv2d_kernel_args(program, node, constants, size_query, *, depthwise=False):
    if (
        get_arg(node, "transposed")
        or (not depthwise and get_arg(node, "groups") != 1)
        or get_arg(node, "dilation") != [1, 1]
        or get_arg(node, "output_padding") != [0, 0]
    ):
        return None
    inp, weight_node, bias_node = (
        get_arg(node, name) for name in ("inp", "weight", "bias")
    )
    input_value, output_value = inp.meta.get("val"), node.meta.get("val")
    weight = constants.get(weight_node)
    if not all(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.int8
        and value.dim() == 4
        and all(type(dim) is int and 0 < dim <= 65535 for dim in value.shape)
        and value.is_contiguous()
        for value in (input_value, output_value, weight)
    ):
        return None
    batch, height, width, channels = input_value.shape
    outputs, kernel_h, kernel_w, weight_channels = weight.shape
    stride, padding = (get_arg(node, name) for name in ("stride", "padding"))
    if (
        len(stride) != 2
        or len(padding) != 2
        or any(type(v) is not int or not 0 < v <= 65535 for v in stride)
        or any(type(v) is not int or not 0 <= v <= 65535 for v in padding)
        or weight_channels != (1 if depthwise else channels)
        or (depthwise and (get_arg(node, "groups") != channels or outputs % channels))
        or output_value.shape
        != (
            batch,
            (height + 2 * padding[0] - kernel_h) // stride[0] + 1,
            (width + 2 * padding[1] - kernel_w) // stride[1] + 1,
            outputs,
        )
    ):
        return None
    # Bound aligned copies, padded input and SIMD staging before the int32 vendor query.
    aligned_window = (kernel_h * kernel_w * channels + 15) & ~15
    aligned_channels = (channels + 15) & ~15
    scratch_bound = (
        (height + 2 * padding[0] + kernel_h)
        * (width + 2 * padding[1] + kernel_w)
        * aligned_channels
        + (kernel_w * aligned_channels + 15) * kernel_h * outputs
        + 40 * aligned_window
        + 8 * outputs
        + 16 * 1024
        + 256
    )
    if depthwise:
        # Vendor C paths store each window's spatial origin in int16.
        if aligned_channels > 65535 or any(
            pad > 32768 or (out - 1) * step - pad > 32767
            for out, step, pad in zip(output_value.shape[1:3], stride, padding)
        ):
            return None
        scratch_bound = (
            2 * kernel_h * kernel_w * max(aligned_channels, outputs)
            + 2
            * (height + 2 * padding[0] + 2 * kernel_h + 2 * stride[0])
            * (width + 2 * padding[1] + 2 * kernel_w + 2 * stride[1])
            * aligned_channels
            + output_value.shape[1] * output_value.shape[2] * aligned_channels
            + 12 * aligned_channels
            + 128
        )
    if (
        max(scratch_bound, input_value.numel(), output_value.numel(), weight.numel())
        > torch.iinfo(torch.int32).max
    ):
        return None
    input_params, output_params = (
        _qparams(program, node, name) for name in ("inp", "out")
    )
    if input_params is None or output_params is None:
        return None
    try:
        scratch_bytes = size_query(
            input_dims=(width, height, channels),
            filter_dims=(kernel_w, kernel_h, outputs if depthwise else channels),
            output_dims=(output_value.shape[2], output_value.shape[1], outputs),
            stride=tuple(reversed(stride)),
            padding=tuple(reversed(padding)),
            **({"channel_multiplier": outputs // channels} if depthwise else {}),
        )
        if not 0 <= scratch_bytes <= scratch_bound:
            raise RuntimeError("ESP-NN returned an invalid convolution scratch size")
        multiplier_node, shift_node = _weighted_constants(
            program, node, constants, weight, input_params, output_params
        )
    except ValueError:
        return None
    # FQ uses OHWI; ESP-NN depthwise indexes weights as 1HWO.
    weight = weight.permute(3, 1, 2, 0).contiguous() if depthwise else weight
    weight_node = add_constant(
        program, f"{node.name}_weight", weight, node, InputKind.CONSTANT_TENSOR
    )
    with node.graph.inserting_before(node):
        scratch = make_alloc_node(
            program.graph_module,
            input_value.new_empty((scratch_bytes,), dtype=torch.uint8),
            None,
        )
    return (
        inp,
        weight_node,
        bias_node,
        stride,
        padding,
        *((outputs // channels,) if depthwise else ()),
        multiplier_node,
        shift_node,
        -input_params.zp,
        output_params.zp,
        scratch,
    )


def _lower_linear(program, node, constants):
    try:
        args = _linear_kernel_args(program, node, constants)
    except ValueError:
        return False
    with node.graph.inserting_before(node):
        replacement = node.graph.call_function(
            exir_ops.edge.esp_nn.quantized_linear_int8.default, args
        )
    replacement.meta = node.meta.copy()
    node.replace_all_uses_with(replacement)
    node.graph.erase_node(node)
    return True


def _lower_conv2d(program, node, constants, target, kernel_mode):
    queries = get_scratch_queries(target, kernel_mode)
    if queries is None:
        return False
    depthwise = get_arg(node, "groups") != 1
    try:
        size_query = (
            queries.depthwise_conv2d_scratch_size
            if depthwise
            else queries.conv2d_scratch_size
        )
    except AttributeError:
        return False
    args = _conv2d_kernel_args(
        program, node, constants, size_query, depthwise=depthwise
    )
    if args is None:
        return False
    with node.graph.inserting_before(node):
        replacement = node.graph.call_function(
            (
                exir_ops.edge.esp_nn.quantized_depthwise_conv2d_int8.default
                if depthwise
                else exir_ops.edge.esp_nn.quantized_conv2d_int8.default
            ),
            args,
        )
    replacement.meta = node.meta.copy()
    node.replace_all_uses_with(replacement)
    node.graph.erase_node(node)
    return True


class LowerToEsp(ExportedProgramPassBase):
    """Lower supported fused-quant operators to ESP-NN."""

    def __init__(
        self,
        *,
        target: EspTarget = EspTarget.ESP32S3,
        kernel_mode: EspKernelMode = EspKernelMode.OPTIMIZED,
    ):
        super().__init__()
        self.target = EspTarget(target)
        self.kernel_mode = EspKernelMode(kernel_mode)

    def call(self, exported_program):
        module = exported_program.graph_module
        graph = module.graph
        weighted_nodes = [
            *graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.linear.default
            ),
            *graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.fused_quant.convolution_channels_last.default,
            ),
        ]
        constants = (
            get_propagated_const_tensor_dict(exported_program, None)
            if weighted_nodes
            else {}
        )

        modified = False
        for node in list(graph.nodes):
            if node.target in _LUT_OPS:
                modified |= _lower_lut(exported_program, node)
                continue
            if node.target == exir_ops.edge.fused_quant.linear.default:
                modified |= _lower_linear(exported_program, node, constants)
                continue
            if (
                node.target
                == exir_ops.edge.fused_quant.convolution_channels_last.default
            ):
                modified |= _lower_conv2d(
                    exported_program, node, constants, self.target, self.kernel_mode
                )
                continue
            if (
                node.target
                == exir_ops.edge.fused_quant.avg_pool2d_channels_last.default
            ):
                modified |= _lower_avg_pool2d(exported_program, node)
                continue
            if (
                node.target
                == exir_ops.edge.fused_quant.max_pool2d_with_indices_channels_last.default
            ):
                modified |= _lower_max_pool2d(exported_program, node)
                continue
            lowering = _BINARY_LOWERINGS.get(node.target)
            if lowering is None:
                continue
            if (
                node.target
                in (
                    exir_ops.edge.fused_quant.add.default,
                    exir_ops.edge.fused_quant.sub.default,
                )
                and get_arg(node, "alpha") != 1
            ):
                continue
            target, rescale_fn = lowering
            params = [
                _qparams(exported_program, node, prefix, allow_output_clamp=True)
                for prefix in ("inp", "other", "out")
            ]
            if any(param is None for param in params):
                continue
            prepared = _prepare_binary(node, params, target, rescale_fn)
            if prepared is None:
                continue
            target, args = prepared
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    target,
                    (
                        *args,
                        get_arg(node, "out_quant_min"),
                        get_arg(node, "out_quant_max"),
                    ),
                )
            replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            module.recompile()
        return ExportedProgramPassResult(exported_program, modified)
