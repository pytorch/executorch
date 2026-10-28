# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import struct

import torch

from .xts3 import (
    _activation_range,
    _fusable_activation,
    _normalize_int_tuple,
    _per_channel_weight_scales,
    _quantize_multiplier,
    _require_output_qparams,
    _serialize_per_channel_requantization,
    _shape_from_node,
    _target_kind,
    COPY_FLAG_FILL,
    COPY_HEADER,
    COPY_MAX_RANK,
    COPY_SEGMENT,
    TensorRegistry,
    TRANSPOSE_CONV_PARAMS,
    UPSAMPLE_HEADER,
    UPSAMPLE_MODE_BILINEAR,
    UPSAMPLE_MODE_NEAREST,
    UPSAMPLE_WEIGHT_ONE,
    XTS3DType,
    XTS3OpCode,
    XTS3Operation,
    XTS3Tensor,
)


MOVEMENT_KINDS = {
    "reshape",
    "flatten",
    "unflatten",
    "permute",
    "transpose",
    "squeeze",
    "unsqueeze",
    "slice",
    "select",
    "expand",
    "split_item",
    "identity_move",
}

SPLIT_KINDS = {"split", "split_with_sizes", "unbind", "chunk"}


def is_dequantize(node) -> bool:
    return (
        hasattr(node, "op")
        and node.op == "call_function"
        and "dequantize_per_tensor" in str(node.target)
    )


def is_quantize(node) -> bool:
    return (
        hasattr(node, "op")
        and node.op == "call_function"
        and "quantize_per_tensor" in str(node.target)
        and "dequantize" not in str(node.target)
    )


def node_kind(node) -> str | None:
    if not hasattr(node, "op") or node.op != "call_function":
        return None
    kind = _target_kind(node.target)
    if (
        kind == "getitem"
        and _target_kind(getattr(node.args[0], "target", None)) in SPLIT_KINDS
    ):
        return "split_item"
    if kind in ("identity", "dropout") and node.args:
        source = node.args[0]
        while is_dequantize(source) and is_quantize(source.args[0]):
            source = source.args[0].args[0]
        if node_kind(source) in MOVEMENT_KINDS:
            return "identity_move"
    return kind


def _skip_quantization(node):
    while is_dequantize(node) and is_quantize(node.args[0]):
        node = node.args[0].args[0]
    while node_kind(node) in ("identity", "dropout"):
        node = node.args[0]
        while is_dequantize(node) and is_quantize(node.args[0]):
            node = node.args[0].args[0]
    return node


def consumers(node) -> list:
    result = []
    for user in node.users:
        if is_quantize(user):
            for dq in user.users:
                if is_dequantize(dq):
                    result.extend(consumers(dq))
                else:
                    result.append(dq)
        elif node_kind(user) in ("identity", "dropout"):
            result.extend(consumers(user))
        else:
            result.append(user)
    return result


def is_movement_terminal(node) -> bool:
    users = consumers(node)
    if not users:
        return True
    return any(node_kind(user) not in MOVEMENT_KINDS for user in users)


def _movement_chain(node):
    chain = []
    current = node
    while node_kind(current) in MOVEMENT_KINDS:
        chain.append(current)
        if node_kind(current) == "split_item":
            current = _skip_quantization(current.args[0].args[0])
        else:
            current = _skip_quantization(current.args[0])
    chain.reverse()
    return current, chain


def _logical_index_view(shape: tuple[int, ...]) -> torch.Tensor:
    count = max(1, math.prod(shape))
    base = torch.arange(count, dtype=torch.int64)
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError(f"Rank-4 int8 tensors must have batch 1, got {shape}")
        return base.view(shape[2], shape[3], shape[1]).permute(2, 0, 1).unsqueeze(0)
    return base.view(shape)


def _device_view(view: torch.Tensor) -> torch.Tensor:
    if view.dim() == 4:
        if view.shape[0] != 1:
            raise ValueError(
                f"Rank-4 int8 tensors must have batch 1, got {tuple(view.shape)}"
            )
        return view[0].permute(1, 2, 0)
    return view


def _resolved_shape(view: torch.Tensor, shape) -> list[int]:
    shape = [int(dim) for dim in shape]
    if shape.count(-1) == 1:
        known = math.prod(dim for dim in shape if dim != -1)
        shape[shape.index(-1)] = view.numel() // max(known, 1)
    return shape


def _split_bounds(split_node, index: int) -> tuple[int, int, int, bool]:
    kind = _target_kind(split_node.target)
    source_shape = _shape_from_node(split_node.args[0])
    if kind == "unbind":
        dim = int(split_node.args[1]) if len(split_node.args) > 1 else 0
        dim = dim % len(source_shape)
        return dim, index, index + 1, True
    dim = (
        int(split_node.args[2])
        if len(split_node.args) > 2
        else int(split_node.kwargs.get("dim", 0))
    )
    dim = dim % len(source_shape)
    size = source_shape[dim]
    if kind == "chunk":
        chunks = int(split_node.args[1])
        step = (size + chunks - 1) // chunks
        sizes = [min(step, size - start) for start in range(0, size, step)]
    elif kind == "split_with_sizes" or isinstance(split_node.args[1], (list, tuple)):
        sizes = [int(v) for v in split_node.args[1]]
    else:
        step = int(split_node.args[1])
        sizes = [min(step, size - start) for start in range(0, size, step)]
    start = sum(sizes[:index])
    return dim, start, start + sizes[index], False


def _apply_movement(view: torch.Tensor, node):  # noqa: C901
    kind = node_kind(node)
    args = node.args
    if kind in ("reshape", "flatten", "unflatten"):
        if kind == "reshape":
            shape = _resolved_shape(view, args[1])
        else:
            shape = list(_shape_from_node(node))
        try:
            return view.view(shape), None
        except RuntimeError:
            return None, shape
    if kind == "permute":
        return view.permute([int(d) for d in args[1]]), None
    if kind == "transpose":
        return view.transpose(int(args[1]), int(args[2])), None
    if kind == "squeeze":
        if len(args) > 1:
            dims = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
            return view.squeeze(tuple(int(d) for d in dims)), None
        return view.squeeze(), None
    if kind == "unsqueeze":
        return view.unsqueeze(int(args[1])), None
    if kind == "slice":
        dim = int(args[1]) if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else None
        end = args[3] if len(args) > 3 else None
        step = int(args[4]) if len(args) > 4 else 1
        return torch.ops.aten.slice.Tensor(view, dim, start, end, step), None
    if kind == "select":
        return view.select(int(args[1]), int(args[2])), None
    if kind == "expand":
        return view.expand([int(d) for d in args[1]]), None
    if kind == "identity_move":
        return view, None
    if kind == "split_item":
        dim, start, end, squeeze = _split_bounds(node.args[0], int(node.args[1]))
        sliced = torch.ops.aten.slice.Tensor(view, dim, start, end, 1)
        return (sliced.squeeze(dim) if squeeze else sliced), None
    raise ValueError(f"Unsupported data movement {node.name}")


def _coalesce(dims, in_strides, out_strides):
    items = [
        (int(d), int(i), int(o))
        for d, i, o in zip(dims, in_strides, out_strides)
        if int(d) != 1
    ]
    merged = []
    for dim, in_stride, out_stride in items:
        if merged:
            p_dim, p_in, p_out = merged[-1]
            if p_in == in_stride * dim and p_out == out_stride * dim:
                merged[-1] = (p_dim * dim, in_stride, out_stride)
                continue
        merged.append((dim, in_stride, out_stride))
    if not merged:
        merged = [(1, 1, 1)]
    return [m[0] for m in merged], [m[1] for m in merged], [m[2] for m in merged]


def _contiguous_strides(dims) -> list[int]:
    strides = []
    running = 1
    for dim in reversed(dims):
        strides.append(running)
        running *= int(dim)
    return list(reversed(strides))


def copy_segment(
    input_index: int,
    dims,
    in_strides,
    in_offset: int,
    out_strides,
    out_offset: int,
    source: XTS3Tensor,
    result: XTS3Tensor,
) -> bytes:
    dims, in_strides, out_strides = _coalesce(dims, in_strides, out_strides)
    if len(dims) > COPY_MAX_RANK:
        raise ValueError(
            f"Data movement needs {len(dims)} dimensions; the limit is {COPY_MAX_RANK}"
        )
    requant = (
        source.quant_scale != result.quant_scale
        or source.quant_zero_point != result.quant_zero_point
    )
    multiplier, shift = (0, 0)
    if requant:
        multiplier, shift = _quantize_multiplier(
            source.quant_scale / result.quant_scale
        )
    pad = COPY_MAX_RANK - len(dims)
    return struct.pack(
        COPY_SEGMENT,
        input_index,
        len(dims),
        int(in_offset),
        int(out_offset),
        *(list(dims) + [1] * pad),
        *(list(in_strides) + [0] * pad),
        *(list(out_strides) + [0] * pad),
        int(requant),
        multiplier,
        shift,
        source.quant_zero_point,
        result.quant_zero_point,
    )


def copy_operation(inputs, output_id, segments, fill_value=None) -> XTS3Operation:
    flags = COPY_FLAG_FILL if fill_value is not None else 0
    header = struct.pack(COPY_HEADER, len(segments), flags, int(fill_value or 0))
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.COPY,
        inputs=list(inputs),
        outputs=[output_id],
        params=header + b"".join(segments),
    )


def _gather_operation(source_id, output_id, view, registry):
    source = registry.tensors[source_id]
    result = registry.tensors[output_id]
    device = _device_view(view)
    segment = copy_segment(
        0,
        list(device.shape),
        list(device.stride()),
        device.storage_offset(),
        _contiguous_strides(device.shape),
        0,
        source,
        result,
    )
    return copy_operation([source_id], output_id, [segment])


def _new_int8_tensor(registry: TensorRegistry, shape, like: XTS3Tensor) -> int:
    tensor_id = len(registry.tensors)
    registry.tensors.append(
        XTS3Tensor(
            tensor_id=tensor_id,
            shape=tuple(int(d) for d in shape),
            dtype=XTS3DType.INT8,
            quant_scale=like.quant_scale,
            quant_zero_point=like.quant_zero_point,
        )
    )
    return tensor_id


def lower_movement(
    node, registry: TensorRegistry, extra_operations: list
) -> XTS3Operation | None:
    if not is_movement_terminal(node):
        registry.fused_nodes.add(node)
        return None
    _require_output_qparams(registry, node)
    source_node, chain = _movement_chain(node)
    source_id = registry.get_or_create(source_node)
    view = _logical_index_view(registry.tensors[source_id].shape)
    current_id = source_id
    for link in chain:
        moved, reshape_to = _apply_movement(view, link)
        if moved is None:
            temp_id = _new_int8_tensor(
                registry, (view.numel(),), registry.tensors[current_id]
            )
            temp = registry.tensors[temp_id]
            segment = copy_segment(
                0,
                list(view.shape),
                list(view.stride()),
                view.storage_offset(),
                _contiguous_strides(view.shape),
                0,
                registry.tensors[current_id],
                temp,
            )
            extra_operations.append(copy_operation([current_id], temp_id, [segment]))
            current_id = temp_id
            view = torch.arange(view.numel(), dtype=torch.int64).view(reshape_to)
        else:
            view = moved
    output_id = registry.get_or_create(node)
    result = registry.tensors[output_id]
    if tuple(view.shape) != tuple(result.shape):
        raise ValueError(
            f"Data movement {node.name} traced to {tuple(view.shape)}, expected {result.shape}"
        )
    return _gather_operation(current_id, output_id, view, registry)


def lower_cat(
    node, registry: TensorRegistry, extra_operations: list, stack: bool = False
) -> XTS3Operation:
    sources = list(node.args[0])
    dim = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs.get("dim", 0))
    _require_output_qparams(registry, node)
    output_id = registry.get_or_create(node)
    result = registry.tensors[output_id]
    rank = len(result.shape)
    dim = dim % rank
    out_view = _logical_index_view(result.shape)
    input_ids = [registry.get_or_create(source) for source in sources]
    operations = []
    offset = 0
    pending_inputs = []
    pending_segments = []
    for input_id in input_ids:
        source = registry.tensors[input_id]
        if stack:
            if len(source.shape) != rank - 1:
                raise ValueError(f"stack {node.name} mixes ranks")
            part = out_view.select(dim, offset)
            offset += 1
        else:
            if len(source.shape) != rank:
                raise ValueError(f"cat {node.name} mixes ranks")
            size = source.shape[dim]
            part = torch.ops.aten.slice.Tensor(out_view, dim, offset, offset + size, 1)
            offset += size
        if source.shape and math.prod(source.shape) == 0:
            continue
        in_device = _device_view(_logical_index_view(source.shape))
        out_device = _device_view(part)
        dims = list(in_device.shape)
        if len(pending_inputs) == 6:
            operations.append(
                copy_operation(pending_inputs, output_id, pending_segments)
            )
            pending_inputs, pending_segments = [], []
        pending_segments.append(
            copy_segment(
                len(pending_inputs),
                dims,
                list(in_device.stride()),
                in_device.storage_offset(),
                list(out_device.stride()),
                out_device.storage_offset(),
                source,
                result,
            )
        )
        pending_inputs.append(input_id)
    operations.append(copy_operation(pending_inputs, output_id, pending_segments))
    extra_operations.extend(operations[:-1])
    return operations[-1]


def lower_pad(node, registry: TensorRegistry) -> XTS3Operation:
    pads = [int(v) for v in node.args[1]]
    value = (
        float(node.args[2])
        if len(node.args) > 2
        else float(node.kwargs.get("value", 0.0))
    )
    if any(p < 0 for p in pads):
        raise ValueError(f"Negative padding (cropping) is not supported: {node.name}")
    _require_output_qparams(registry, node)
    input_id = registry.get_or_create(node.args[0])
    output_id = registry.get_or_create(node)
    source = registry.tensors[input_id]
    result = registry.tensors[output_id]
    rank = len(result.shape)
    out_view = _logical_index_view(result.shape)
    region = out_view
    for i in range(len(pads) // 2):
        dim = rank - 1 - i
        before = pads[2 * i]
        region = torch.ops.aten.slice.Tensor(
            region, dim, before, before + source.shape[dim], 1
        )
    in_device = _device_view(_logical_index_view(source.shape))
    out_device = _device_view(region)
    fill = round(value / result.quant_scale) + result.quant_zero_point
    fill = max(-128, min(127, fill))
    segment = copy_segment(
        0,
        list(in_device.shape),
        list(in_device.stride()),
        0,
        list(out_device.stride()),
        out_device.storage_offset(),
        source,
        result,
    )
    return copy_operation([input_id], output_id, [segment], fill_value=fill)


def broadcast_operand(
    arg, target_shape, registry: TensorRegistry, extra_operations: list
) -> int:
    tensor_id = registry.get_or_create(arg)
    tensor = registry.tensors[tensor_id]
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor_id
    if tensor.is_constant:
        return registry.broadcast_constant(tensor_id, tuple(target_shape))
    shape = (1,) * (len(target_shape) - len(tensor.shape)) + tuple(tensor.shape)
    view = _logical_index_view(tensor.shape).reshape(shape).expand(*target_shape)
    temp_id = _new_int8_tensor(registry, target_shape, tensor)
    extra_operations.append(_gather_operation(tensor_id, temp_id, view, registry))
    return temp_id


def upsample_tables(node, in_size: int, out_size: int, axis: int, bilinear: bool):
    args = node.args
    kind = _target_kind(node.target)
    scales = None
    align_corners = False
    if bilinear:
        align_corners = (
            bool(args[2])
            if len(args) > 2
            else bool(node.kwargs.get("align_corners", False))
        )
        if kind.endswith("_vec"):
            scales = args[3] if len(args) > 3 else node.kwargs.get("scale_factors")
        else:
            factor = args[3 + axis] if len(args) > 3 + axis else None
            scales = None if factor is None else [factor, factor]
    else:
        if kind.endswith("_vec"):
            scales = args[2] if len(args) > 2 else node.kwargs.get("scale_factors")
        else:
            factor = args[2 + axis] if len(args) > 2 + axis else None
            scales = None if factor is None else [factor, factor]
    scale_factor = None
    if scales is not None and len(scales) == 2 and scales[axis] is not None:
        scale_factor = float(scales[axis])
    table = []
    if not bilinear:
        ratio = (1.0 / scale_factor) if scale_factor else in_size / out_size
        for dst in range(out_size):
            src = min(int(math.floor(dst * ratio)), in_size - 1)
            table.extend([src, src, 0])
        return table
    if align_corners:
        ratio = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    else:
        ratio = (1.0 / scale_factor) if scale_factor else in_size / out_size
    for dst in range(out_size):
        if align_corners:
            src = dst * ratio
        else:
            src = max((dst + 0.5) * ratio - 0.5, 0.0)
        x0 = min(int(math.floor(src)), in_size - 1)
        x1 = x0 + 1 if x0 < in_size - 1 else x0
        weight = int(round((src - x0) * UPSAMPLE_WEIGHT_ONE)) if x1 != x0 else 0
        table.extend([x0, x1, weight])
    return table


def lower_upsample(node, registry: TensorRegistry) -> XTS3Operation:
    kind = _target_kind(node.target)
    bilinear = kind.startswith("upsample_bilinear2d")
    _require_output_qparams(registry, node)
    input_id = registry.get_or_create(node.args[0])
    output_id = registry.get_or_create(node)
    source = registry.tensors[input_id]
    result = registry.tensors[output_id]
    if len(source.shape) != 4 or source.shape[0] != 1:
        raise ValueError(f"Upsampling needs a batch-1 NCHW tensor: {node.name}")
    if (
        source.quant_scale != result.quant_scale
        or source.quant_zero_point != result.quant_zero_point
    ):
        raise ValueError(f"Upsampling {node.name} must keep its input quantization")
    in_h, in_w = source.shape[2], source.shape[3]
    out_h, out_w = result.shape[2], result.shape[3]
    y_table = upsample_tables(node, in_h, out_h, 0, bilinear)
    x_table = upsample_tables(node, in_w, out_w, 1, bilinear)
    params = (
        struct.pack(
            UPSAMPLE_HEADER,
            UPSAMPLE_MODE_BILINEAR if bilinear else UPSAMPLE_MODE_NEAREST,
            in_h,
            in_w,
            out_h,
            out_w,
        )
        + struct.pack(f"<{len(y_table)}i", *y_table)
        + struct.pack(f"<{len(x_table)}i", *x_table)
    )
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.UPSAMPLE,
        inputs=[input_id],
        outputs=[output_id],
        params=params,
    )


def transposed_conv_args(node) -> dict:
    kind = _target_kind(node.target)
    args = node.args
    if kind == "convolution":
        return {
            "input": args[0],
            "weight": args[1],
            "bias": args[2],
            "stride": _normalize_int_tuple(args[3], 2, "stride"),
            "padding": _normalize_int_tuple(args[4], 2, "padding"),
            "dilation": _normalize_int_tuple(args[5], 2, "dilation"),
            "output_padding": _normalize_int_tuple(args[7], 2, "output_padding"),
            "groups": int(args[8]),
        }
    return {
        "input": args[0],
        "weight": args[1],
        "bias": args[2] if len(args) > 2 else None,
        "stride": _normalize_int_tuple(args[3] if len(args) > 3 else 1, 2, "stride"),
        "padding": _normalize_int_tuple(args[4] if len(args) > 4 else 0, 2, "padding"),
        "output_padding": _normalize_int_tuple(
            args[5] if len(args) > 5 else 0, 2, "output_padding"
        ),
        "groups": int(args[6]) if len(args) > 6 else 1,
        "dilation": _normalize_int_tuple(
            args[7] if len(args) > 7 else 1, 2, "dilation"
        ),
    }


def lower_transposed_conv(node, registry: TensorRegistry) -> XTS3Operation:
    conv = transposed_conv_args(node)
    if conv["groups"] != 1:
        raise ValueError(
            f"Grouped transposed convolution is not supported: {node.name}"
        )
    activation, activation_kind = _fusable_activation(node, registry)
    output_node = activation if activation is not None else node
    _require_output_qparams(registry, output_node)
    input_id = registry.get_or_create(conv["input"])
    weight_scales = _per_channel_weight_scales(registry, conv["weight"])
    weight_id = registry.get_or_create(conv["weight"], "TRANSPOSED")
    input_tensor = registry.tensors[input_id]
    bias_id = registry.add_int32_bias(
        conv["bias"], input_tensor.quant_scale, weight_scales
    )
    output_id = registry.get_or_create(output_node)
    output_tensor = registry.tensors[output_id]
    if activation is not None:
        registry.fused_nodes.add(activation)
    weight_shape = _shape_from_node(conv["weight"])
    act_min, act_max = _activation_range(
        activation_kind, output_tensor.quant_scale, output_tensor.quant_zero_point
    )
    params = struct.pack(
        TRANSPOSE_CONV_PARAMS,
        weight_shape[2],
        weight_shape[3],
        *conv["stride"],
        *conv["padding"],
        *conv["output_padding"],
        *conv["dilation"],
        conv["groups"],
        act_min,
        act_max,
    ) + _serialize_per_channel_requantization(
        input_tensor.quant_scale, weight_scales, output_tensor.quant_scale
    )
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.TRANSPOSE_CONV2D,
        inputs=[input_id, weight_id, bias_id],
        outputs=[output_id],
        params=params,
    )


__all__ = [
    "MOVEMENT_KINDS",
    "SPLIT_KINDS",
    "broadcast_operand",
    "copy_operation",
    "copy_segment",
    "is_dequantize",
    "is_movement_terminal",
    "is_quantize",
    "lower_cat",
    "lower_movement",
    "lower_pad",
    "lower_transposed_conv",
    "lower_upsample",
    "node_kind",
    "transposed_conv_args",
    "upsample_tables",
]
