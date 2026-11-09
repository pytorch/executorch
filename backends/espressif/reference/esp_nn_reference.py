# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..serialization.xts3 import (
    COPY_FLAG_FILL,
    COPY_HEADER,
    COPY_HEADER_SIZE,
    COPY_MAX_RANK,
    COPY_SEGMENT,
    COPY_SEGMENT_SIZE,
    INT8_ADD_PARAMS,
    INT8_CONV_BASE_PARAMS,
    INT8_CONV_BASE_PARAMS_SIZE,
    INT8_MUL_PARAMS,
    INT8_PAIR_PARAMS,
    INT8_PAIR_PARAMS_SIZE,
    INT8_POOL_PARAMS,
    MUL_MODE_CHANNEL_BROADCAST,
    SOFTMAX_OUTPUT_ZERO_POINT,
    TENSOR_FLAG_GRAPH_INPUT,
    TENSOR_FLAG_GRAPH_OUTPUT,
    TRANSPOSE_CONV_PARAMS,
    TRANSPOSE_CONV_PARAMS_SIZE,
    UPSAMPLE_HEADER,
    UPSAMPLE_HEADER_SIZE,
    UPSAMPLE_WEIGHT_BITS,
    XTS3_VERSIONS,
    XTS3DType,
    XTS3Graph,
    XTS3OpCode,
    XTS3Tensor,
)


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def wrap_int32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return ((values + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def saturating_rounding_doubling_high_mul(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    overflow = (a == b) & (a == INT32_MIN)
    nudge = np.where((a < 0) ^ (b < 0), 1 - (1 << 30), 1 << 30)
    product = a * b + nudge
    high = np.where(product >= 0, product >> 31, -((-product) >> 31))
    return np.where(overflow, INT32_MAX, high)


def rounding_divide_by_power_of_two(x, exponent) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    exponent = np.asarray(exponent, dtype=np.int64)
    mask = (np.int64(1) << exponent) - 1
    remainder = x & mask
    result = x >> exponent
    threshold = (mask >> 1) + (result < 0)
    return result + (remainder > threshold)


def multiply_by_quantized_multiplier(x, multiplier, shift) -> np.ndarray:
    shift = np.asarray(shift, dtype=np.int64)
    multiplier = np.asarray(multiplier, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    shifted = wrap_int32(np.asarray(x, dtype=np.int64) << left)
    return rounding_divide_by_power_of_two(
        saturating_rounding_doubling_high_mul(shifted, multiplier), right
    )


def _exact_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = a.astype(np.float64) @ b.astype(np.float64)
    if result.size and np.max(np.abs(result)) >= float(1 << 52):
        raise OverflowError("Accumulator exceeds the exactly representable range")
    return np.rint(result).astype(np.int64)


def device_shape(tensor: XTS3Tensor) -> tuple[int, ...]:
    shape = tuple(tensor.shape)
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError(
                f"Tensor {tensor.tensor_id} has batch {shape[0]}; only batch 1 is supported"
            )
        return (shape[2], shape[3], shape[1])
    if len(shape) == 2 and shape[0] == 1:
        return (shape[1],)
    return shape


def to_device_layout(tensor: XTS3Tensor, logical: np.ndarray) -> np.ndarray:
    logical = np.asarray(logical)
    if logical.shape != tuple(tensor.shape):
        raise ValueError(f"Expected shape {tensor.shape}, got {logical.shape}")
    if len(tensor.shape) == 4:
        return np.ascontiguousarray(logical[0].transpose(1, 2, 0))
    return np.ascontiguousarray(logical.reshape(device_shape(tensor)))


def to_logical_layout(tensor: XTS3Tensor, device: np.ndarray) -> np.ndarray:
    if len(tensor.shape) == 4:
        return np.ascontiguousarray(device.transpose(2, 0, 1)[None])
    return np.ascontiguousarray(device.reshape(tensor.shape))


def quantize(tensor: XTS3Tensor, values: np.ndarray) -> np.ndarray:
    inverse = np.float32(1.0 / tensor.quant_scale)
    scaled = np.asarray(values, dtype=np.float32) * inverse
    quantized = np.rint(scaled).astype(np.int64) + tensor.quant_zero_point
    return np.clip(quantized, -128, 127).astype(np.int8)


def dequantize(tensor: XTS3Tensor, values: np.ndarray) -> np.ndarray:
    return (
        values.astype(np.float32) - np.float32(tensor.quant_zero_point)
    ) * np.float32(tensor.quant_scale)


@dataclass
class ConvParams:
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    groups: int
    act_min: int
    act_max: int
    multipliers: np.ndarray
    shifts: np.ndarray
    dilation_h: int = 1
    dilation_w: int = 1


def requantize(values, multiplier, shift, in_zero_point, out_zero_point) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.int64) - in_zero_point
    result = (
        multiply_by_quantized_multiplier(shifted, multiplier, shift) + out_zero_point
    )
    return np.clip(result, -128, 127)


def run_copy(
    params: bytes, sources: list[np.ndarray], output: np.ndarray
) -> np.ndarray:
    count, flags, fill = struct.unpack_from(COPY_HEADER, params, 0)
    flat = output.reshape(-1).copy()
    if flags & COPY_FLAG_FILL:
        flat[:] = fill
    offset = COPY_HEADER_SIZE
    for _ in range(count):
        fields = struct.unpack_from(COPY_SEGMENT, params, offset)
        offset += COPY_SEGMENT_SIZE
        index, rank, in_offset, out_offset = fields[:4]
        dims = fields[4 : 4 + COPY_MAX_RANK][:rank]
        in_strides = fields[4 + COPY_MAX_RANK : 4 + 2 * COPY_MAX_RANK][:rank]
        out_strides = fields[4 + 2 * COPY_MAX_RANK : 4 + 3 * COPY_MAX_RANK][:rank]
        requant, multiplier, shift, in_zp, out_zp = fields[4 + 3 * COPY_MAX_RANK :]
        grids = np.meshgrid(
            *[np.arange(d, dtype=np.int64) for d in dims], indexing="ij"
        )
        in_index = np.full(grids[0].shape, in_offset, dtype=np.int64)
        out_index = np.full(grids[0].shape, out_offset, dtype=np.int64)
        for grid, in_stride, out_stride in zip(grids, in_strides, out_strides):
            in_index += grid * in_stride
            out_index += grid * out_stride
        values = sources[index].reshape(-1)[in_index.reshape(-1)].astype(np.int64)
        if requant:
            values = requantize(values, multiplier, shift, in_zp, out_zp)
        flat[out_index.reshape(-1)] = values
    return flat.reshape(output.shape)


def parse_conv_params(params: bytes, channels: int) -> ConvParams:
    expected = INT8_CONV_BASE_PARAMS_SIZE + 8 * channels
    if len(params) != expected:
        raise ValueError(
            f"Convolution parameters are {len(params)} bytes, expected {expected}"
        )
    kh, kw, sh, sw, ph, pw, dh, dw, groups, act_min, act_max = struct.unpack_from(
        INT8_CONV_BASE_PARAMS, params, 0
    )
    base = INT8_CONV_BASE_PARAMS_SIZE
    multipliers = np.frombuffer(
        params, dtype="<i4", count=channels, offset=base
    ).astype(np.int64)
    shifts = np.frombuffer(
        params, dtype="<i4", count=channels, offset=base + 4 * channels
    ).astype(np.int64)
    return ConvParams(
        kh, kw, sh, sw, ph, pw, groups, act_min, act_max, multipliers, shifts, dh, dw
    )


class ReferenceInterpreter:
    """
    Bit-exact host model of the ESP-NN int8 kernels used by the firmware.

    Activations are held in the device memory layout: rank-4 tensors as
    HWC (batch 1, channels last), rank-2 batch-1 tensors as flat vectors.
    """

    def __init__(self, graph: XTS3Graph):
        if graph.version not in XTS3_VERSIONS:
            raise ValueError(
                f"Expected an ESP-NN int8 graph (versions {XTS3_VERSIONS})"
            )
        self.graph = graph
        self.inputs = list(graph.inputs) or [
            t.tensor_id for t in graph.tensors if t.flags & TENSOR_FLAG_GRAPH_INPUT
        ]
        self.outputs = list(graph.outputs) or [
            t.tensor_id for t in graph.tensors if t.flags & TENSOR_FLAG_GRAPH_OUTPUT
        ]

    def constant(self, tensor_id: int) -> np.ndarray:
        tensor = self.graph.tensors[tensor_id]
        dtype = {XTS3DType.INT8: np.int8, XTS3DType.INT32: "<i4"}[tensor.dtype]
        count = int(np.prod(tensor.shape))
        return np.frombuffer(
            self.graph.constant_data,
            dtype=dtype,
            count=count,
            offset=tensor.data_offset,
        ).astype(np.int64)

    def value(self, tensor_id: int, values) -> np.ndarray:
        if tensor_id in values:
            return values[tensor_id]
        tensor = self.graph.tensors[tensor_id]
        if tensor.is_constant:
            return (
                self.constant(tensor_id).astype(np.int8).reshape(device_shape(tensor))
            )
        raise KeyError(f"Tensor {tensor_id} has not been produced")

    def run(self, inputs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        values: dict[int, np.ndarray] = {}
        for tensor_id in self.inputs:
            tensor = self.graph.tensors[tensor_id]
            data = np.asarray(inputs[tensor_id], dtype=np.int8)
            if data.shape != device_shape(tensor):
                raise ValueError(
                    f"Input {tensor_id} must have device shape {device_shape(tensor)}"
                )
            values[tensor_id] = data
        for operation in self.graph.operations:
            values[operation.outputs[0]] = self._run_operation(operation, values)
        return values

    def run_outputs(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        values = self.run(dict(zip(self.inputs, inputs)))
        return [values[tensor_id] for tensor_id in self.outputs]

    def _run_operation(self, operation, values) -> np.ndarray:  # noqa: C901
        tensors = self.graph.tensors
        output = tensors[operation.outputs[0]]
        opcode = operation.opcode

        if opcode in (XTS3OpCode.CONV2D, XTS3OpCode.DEPTHWISE_CONV2D):
            source = tensors[operation.inputs[0]]
            x = values[source.tensor_id].astype(np.int64)
            channels = output.shape[1]
            p = parse_conv_params(operation.params, channels)
            weights = self.constant(operation.inputs[1])
            bias = self.constant(operation.inputs[2])
            height, width, in_channels = x.shape
            out_h, out_w = output.shape[2], output.shape[3]
            dh, dw = p.dilation_h, p.dilation_w
            padded = np.zeros(
                (height + 2 * p.pad_h, width + 2 * p.pad_w, in_channels), dtype=np.int64
            )
            padded[p.pad_h : p.pad_h + height, p.pad_w : p.pad_w + width] = (
                x - source.quant_zero_point
            )
            accumulator = np.zeros((out_h, out_w, channels), dtype=np.int64)
            if opcode == XTS3OpCode.CONV2D:
                groups = p.groups
                group_in = in_channels // groups
                group_out = channels // groups
                filters = weights.reshape(channels, p.kernel_h, p.kernel_w, group_in)
                for kh in range(p.kernel_h):
                    for kw in range(p.kernel_w):
                        patch = padded[
                            kh * dh : kh * dh
                            + p.stride_h * (out_h - 1)
                            + 1 : p.stride_h,
                            kw * dw : kw * dw
                            + p.stride_w * (out_w - 1)
                            + 1 : p.stride_w,
                        ]
                        for g in range(groups):
                            accumulator[
                                :, :, g * group_out : (g + 1) * group_out
                            ] += _exact_matmul(
                                patch[:, :, g * group_in : (g + 1) * group_in].reshape(
                                    -1, group_in
                                ),
                                filters[
                                    g * group_out : (g + 1) * group_out, kh, kw, :
                                ].T,
                            ).reshape(
                                out_h, out_w, group_out
                            )
            else:
                multiplier = channels // in_channels
                filters = weights.reshape(p.kernel_h, p.kernel_w, channels)
                expanded = (
                    np.repeat(padded, multiplier, axis=2) if multiplier > 1 else padded
                )
                for kh in range(p.kernel_h):
                    for kw in range(p.kernel_w):
                        patch = expanded[
                            kh * dh : kh * dh
                            + p.stride_h * (out_h - 1)
                            + 1 : p.stride_h,
                            kw * dw : kw * dw
                            + p.stride_w * (out_w - 1)
                            + 1 : p.stride_w,
                        ]
                        accumulator += patch * filters[kh, kw]
            accumulator = wrap_int32(accumulator + bias)
            result = multiply_by_quantized_multiplier(
                accumulator, p.multipliers, p.shifts
            )
            result = result + output.quant_zero_point
            return np.clip(result, p.act_min, p.act_max).astype(np.int8)

        if opcode == XTS3OpCode.RELU6:
            act_min, act_max = struct.unpack(INT8_PAIR_PARAMS, operation.params)
            return np.clip(values[operation.inputs[0]], act_min, act_max).astype(
                np.int8
            )

        if opcode == XTS3OpCode.ADD:
            (
                lhs_mult,
                lhs_shift,
                rhs_mult,
                rhs_shift,
                left_shift,
                out_mult,
                out_shift,
                act_min,
                act_max,
            ) = struct.unpack(INT8_ADD_PARAMS, operation.params)
            lhs = tensors[operation.inputs[0]]
            rhs = tensors[operation.inputs[1]]
            a = (
                self.value(lhs.tensor_id, values).astype(np.int64)
                - lhs.quant_zero_point
            ) << left_shift
            b = (
                self.value(rhs.tensor_id, values).astype(np.int64)
                - rhs.quant_zero_point
            ) << left_shift
            a = rounding_divide_by_power_of_two(
                saturating_rounding_doubling_high_mul(a, lhs_mult), -lhs_shift
            )
            b = rounding_divide_by_power_of_two(
                saturating_rounding_doubling_high_mul(b, rhs_mult), -rhs_shift
            )
            total = wrap_int32(a + b)
            total = rounding_divide_by_power_of_two(
                saturating_rounding_doubling_high_mul(total, out_mult), -out_shift
            )
            total = total + output.quant_zero_point
            return np.clip(total, act_min, act_max).astype(np.int8)

        if opcode == XTS3OpCode.AVGPOOL2D:
            multiplier, shift = struct.unpack(INT8_PAIR_PARAMS, operation.params)
            source = tensors[operation.inputs[0]]
            x = values[source.tensor_id].astype(np.int64)
            elements = x.shape[0] * x.shape[1]
            total = x.sum(axis=(0, 1)) - elements * source.quant_zero_point
            result = multiply_by_quantized_multiplier(total, multiplier, shift)
            result = result + output.quant_zero_point
            return np.clip(result, -128, 127).astype(np.int8).reshape(1, 1, -1)

        if opcode == XTS3OpCode.FLATTEN:
            source = tensors[operation.inputs[0]]
            x = values[source.tensor_id]
            if len(source.shape) == 4:
                return np.ascontiguousarray(x.transpose(2, 0, 1)).reshape(-1)
            return x.reshape(-1)

        if opcode in (XTS3OpCode.MAX_POOL2D, XTS3OpCode.AVG_POOL2D):
            kh, kw, sh, sw, ph, pw, _, _, act_min, act_max = struct.unpack(
                INT8_POOL_PARAMS, operation.params
            )
            x = values[operation.inputs[0]].astype(np.int64)
            height, width, channels = x.shape
            out_h, out_w = output.shape[2], output.shape[3]
            result = np.zeros((out_h, out_w, channels), dtype=np.int64)
            for oy in range(out_h):
                y0 = oy * sh - ph
                ys, ye = max(0, y0), min(height, y0 + kh)
                for ox in range(out_w):
                    x0 = ox * sw - pw
                    xs, xe = max(0, x0), min(width, x0 + kw)
                    window = x[ys:ye, xs:xe].reshape(-1, channels)
                    if opcode == XTS3OpCode.MAX_POOL2D:
                        result[oy, ox] = window.max(axis=0)
                    else:
                        count = window.shape[0]
                        total = window.sum(axis=0)
                        half = count // 2
                        numerator = np.where(total > 0, total + half, total - half)
                        result[oy, ox] = np.sign(numerator) * (
                            np.abs(numerator) // count
                        )
            return np.clip(result, act_min, act_max).astype(np.int8)

        if opcode == XTS3OpCode.LUT:
            table = np.frombuffer(operation.params, dtype=np.int8)
            x = values[operation.inputs[0]].astype(np.int64)
            return table[x + 128]

        if opcode == XTS3OpCode.MUL:
            multiplier, shift, act_min, act_max, mode = struct.unpack(
                INT8_MUL_PARAMS, operation.params
            )
            lhs = tensors[operation.inputs[0]]
            rhs = tensors[operation.inputs[1]]
            a = (
                self.value(lhs.tensor_id, values).astype(np.int64)
                - lhs.quant_zero_point
            )
            b = (
                self.value(rhs.tensor_id, values).astype(np.int64)
                - rhs.quant_zero_point
            )
            if mode == MUL_MODE_CHANNEL_BROADCAST:
                b = b.reshape(-1)
            product = multiply_by_quantized_multiplier(a * b, multiplier, shift)
            product = product + output.quant_zero_point
            return np.clip(product, act_min, act_max).astype(np.int8)

        if opcode == XTS3OpCode.SOFTMAX:
            table = np.frombuffer(operation.params, dtype="<u4").astype(np.int64)
            x = values[operation.inputs[0]].astype(np.int64)
            rows = x.reshape(-1, x.shape[-1])
            exps = table[rows.max(axis=1, keepdims=True) - rows]
            totals = exps.sum(axis=1, keepdims=True)
            result = (2 * exps * 256 + totals) // (
                2 * totals
            ) + SOFTMAX_OUTPUT_ZERO_POINT
            return np.clip(result, -128, 127).astype(np.int8).reshape(x.shape)

        if opcode == XTS3OpCode.CONCAT:
            parts = [values[tensor_id] for tensor_id in operation.inputs]
            return np.ascontiguousarray(np.concatenate(parts, axis=-1))

        if opcode == XTS3OpCode.LINEAR:
            source = tensors[operation.inputs[0]]
            weight = tensors[operation.inputs[1]]
            out_features, in_features = weight.shape
            act_min, act_max = struct.unpack_from(INT8_PAIR_PARAMS, operation.params, 0)
            base = INT8_PAIR_PARAMS_SIZE
            multipliers = np.frombuffer(
                operation.params, dtype="<i4", count=out_features, offset=base
            ).astype(np.int64)
            shifts = np.frombuffer(
                operation.params,
                dtype="<i4",
                count=out_features,
                offset=base + 4 * out_features,
            ).astype(np.int64)
            x = values[source.tensor_id].astype(np.int64).reshape(-1, in_features)
            x = x - source.quant_zero_point
            w = self.constant(weight.tensor_id).reshape(out_features, in_features)
            accumulator = wrap_int32(
                _exact_matmul(x, w.T) + self.constant(operation.inputs[2])
            )
            result = multiply_by_quantized_multiplier(accumulator, multipliers, shifts)
            result = result + output.quant_zero_point
            return (
                np.clip(result, act_min, act_max)
                .astype(np.int8)
                .reshape(device_shape(output))
            )

        if opcode == XTS3OpCode.COPY:
            sources = [self.value(tensor_id, values) for tensor_id in operation.inputs]
            current = values.get(output.tensor_id)
            if current is None:
                current = np.zeros(device_shape(output), dtype=np.int64)
            return run_copy(operation.params, sources, current.astype(np.int64)).astype(
                np.int8
            )

        if opcode == XTS3OpCode.UPSAMPLE:
            mode, in_h, in_w, out_h, out_w = struct.unpack_from(
                UPSAMPLE_HEADER, operation.params, 0
            )
            tables = np.frombuffer(
                operation.params, dtype="<i4", offset=UPSAMPLE_HEADER_SIZE
            ).astype(np.int64)
            y = tables[: 3 * out_h].reshape(out_h, 3)
            xs = tables[3 * out_h :].reshape(out_w, 3)
            x = values[operation.inputs[0]].astype(np.int64)
            one = 1 << UPSAMPLE_WEIGHT_BITS
            top = (
                x[y[:, 0]][:, xs[:, 0]] * (one - xs[:, 2])[None, :, None]
                + x[y[:, 0]][:, xs[:, 1]] * xs[:, 2][None, :, None]
            )
            bottom = (
                x[y[:, 1]][:, xs[:, 0]] * (one - xs[:, 2])[None, :, None]
                + x[y[:, 1]][:, xs[:, 1]] * xs[:, 2][None, :, None]
            )
            total = (
                top * (one - y[:, 2])[:, None, None] + bottom * y[:, 2][:, None, None]
            )
            result = (total + (1 << (2 * UPSAMPLE_WEIGHT_BITS - 1))) >> (
                2 * UPSAMPLE_WEIGHT_BITS
            )
            return np.clip(result, -128, 127).astype(np.int8)

        if opcode == XTS3OpCode.TRANSPOSE_CONV2D:
            source = tensors[operation.inputs[0]]
            x = values[source.tensor_id].astype(np.int64) - source.quant_zero_point
            channels = output.shape[1]
            fields = struct.unpack_from(TRANSPOSE_CONV_PARAMS, operation.params, 0)
            kh, kw, sh, sw, ph, pw, _, _, dh, dw, _, act_min, act_max = fields
            base = TRANSPOSE_CONV_PARAMS_SIZE
            multipliers = np.frombuffer(
                operation.params, dtype="<i4", count=channels, offset=base
            ).astype(np.int64)
            shifts = np.frombuffer(
                operation.params,
                dtype="<i4",
                count=channels,
                offset=base + 4 * channels,
            ).astype(np.int64)
            weights = self.constant(operation.inputs[1])
            bias = self.constant(operation.inputs[2])
            height, width, in_channels = x.shape
            out_h, out_w = output.shape[2], output.shape[3]
            filters = weights.reshape(channels, kh, kw, in_channels)
            accumulator = np.zeros((out_h, out_w, channels), dtype=np.int64)
            for ky in range(kh):
                for kx in range(kw):
                    for iy in range(height):
                        oy = iy * sh - ph + ky * dh
                        if not 0 <= oy < out_h:
                            continue
                        ox = np.arange(width) * sw - pw + kx * dw
                        valid = (ox >= 0) & (ox < out_w)
                        if not np.any(valid):
                            continue
                        contribution = _exact_matmul(
                            x[iy][valid], filters[:, ky, kx, :].T
                        )
                        accumulator[oy, ox[valid]] += contribution
            accumulator = wrap_int32(accumulator + bias)
            result = multiply_by_quantized_multiplier(accumulator, multipliers, shifts)
            result = result + output.quant_zero_point
            return np.clip(result, act_min, act_max).astype(np.int8)

        raise ValueError(f"Unsupported opcode {opcode}")


def plan_activation_arena(
    graph: XTS3Graph, alignment: int = 16
) -> tuple[int, dict[int, int]]:
    """Mirror of Int8Interpreter::plan_memory in the firmware."""
    op_count = len(graph.operations)
    big = 1 << 30
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for tensor in graph.tensors:
        if tensor.is_constant:
            continue
        first[tensor.tensor_id] = -1 if tensor.flags & TENSOR_FLAG_GRAPH_INPUT else big
        last[tensor.tensor_id] = (
            op_count if tensor.flags & TENSOR_FLAG_GRAPH_OUTPUT else -1
        )
        size = int(np.prod(tensor.shape))
        sizes[tensor.tensor_id] = (size + alignment - 1) // alignment * alignment
    for index, operation in enumerate(graph.operations):
        for tensor_id in operation.inputs:
            if tensor_id in last:
                last[tensor_id] = max(last[tensor_id], index)
        for tensor_id in operation.outputs:
            first[tensor_id] = min(first[tensor_id], index)
            last[tensor_id] = max(last[tensor_id], index)
    order = sorted(
        (tensor_id for tensor_id in sizes if first[tensor_id] != big),
        key=lambda tensor_id: (-sizes[tensor_id], tensor_id),
    )
    offsets: dict[int, int] = {}
    arena = 0
    for tensor_id in order:
        candidate = 0
        moved = True
        while moved:
            moved = False
            for other, offset in offsets.items():
                if not (
                    first[other] <= last[tensor_id] and first[tensor_id] <= last[other]
                ):
                    continue
                if (
                    candidate < offset + sizes[other]
                    and offset < candidate + sizes[tensor_id]
                ):
                    candidate = offset + sizes[other]
                    moved = True
        offsets[tensor_id] = candidate
        arena = max(arena, candidate + sizes[tensor_id])
    return arena, offsets


__all__ = [
    "ReferenceInterpreter",
    "plan_activation_arena",
    "dequantize",
    "device_shape",
    "multiply_by_quantized_multiplier",
    "quantize",
    "rounding_divide_by_power_of_two",
    "saturating_rounding_doubling_high_mul",
    "to_device_layout",
    "to_logical_layout",
]
