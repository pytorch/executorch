# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from math import frexp, prod

import torch


MAGIC = b"XTS3"
XTS3_VERSION = 6
XTS3_VERSIONS = (5, 6)
IO_TABLE_HEADER = "<HH"
IO_TABLE_HEADER_SIZE = struct.calcsize(IO_TABLE_HEADER)

INT8_CONV_BASE_PARAMS = "<8BIii"
INT8_CONV_BASE_PARAMS_SIZE = struct.calcsize(INT8_CONV_BASE_PARAMS)
INT8_ADD_PARAMS = "<9i"
INT8_ADD_PARAMS_SIZE = struct.calcsize(INT8_ADD_PARAMS)
INT8_PAIR_PARAMS = "<2i"
INT8_POOL_PARAMS = "<8Bii"
INT8_MUL_PARAMS = "<5i"
INT8_LUT_SIZE = 256
SOFTMAX_EXP_BITS = 20
SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128
MUL_MODE_ELEMENTWISE = 0
MUL_MODE_CHANNEL_BROADCAST = 1
INT8_POOL_PARAMS_SIZE = struct.calcsize(INT8_POOL_PARAMS)
INT8_PAIR_PARAMS_SIZE = struct.calcsize(INT8_PAIR_PARAMS)
ADD_LEFT_SHIFT = 20
COPY_HEADER = "<3i"
COPY_HEADER_SIZE = struct.calcsize(COPY_HEADER)
COPY_MAX_RANK = 6
COPY_SEGMENT = "<4i6i6i6i5i"
COPY_SEGMENT_SIZE = struct.calcsize(COPY_SEGMENT)
COPY_FLAG_FILL = 1
COPY_MAX_INPUTS = 6
UPSAMPLE_HEADER = "<5i"
UPSAMPLE_HEADER_SIZE = struct.calcsize(UPSAMPLE_HEADER)
UPSAMPLE_MODE_NEAREST = 0
UPSAMPLE_MODE_BILINEAR = 1
UPSAMPLE_WEIGHT_BITS = 10
UPSAMPLE_WEIGHT_ONE = 1 << UPSAMPLE_WEIGHT_BITS
TRANSPOSE_CONV_PARAMS = "<13i"
TRANSPOSE_CONV_PARAMS_SIZE = struct.calcsize(TRANSPOSE_CONV_PARAMS)

HEADER_FORMAT = "<4sHIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

TENSOR_FORMAT = "<IIBBH4ifi"
TENSOR_SIZE = struct.calcsize(TENSOR_FORMAT)

OP_FORMAT = "<BBBB6I2III"
OP_SIZE = struct.calcsize(OP_FORMAT)


class XTS3OpCode(IntEnum):
    CONV2D = 0
    DEPTHWISE_CONV2D = 1
    RELU6 = 2
    ADD = 3
    AVGPOOL2D = 4
    LINEAR = 5
    FLATTEN = 6
    MAX_POOL2D = 7
    AVG_POOL2D = 8
    CONCAT = 9
    LUT = 10
    MUL = 11
    SOFTMAX = 12
    COPY = 13
    UPSAMPLE = 14
    TRANSPOSE_CONV2D = 15


class XTS3DType(IntEnum):
    UNKNOWN = 0
    FLOAT32 = 1
    INT8 = 2
    UINT8 = 3
    INT32 = 4


def dtype_size(dtype: XTS3DType) -> int:
    if dtype == XTS3DType.FLOAT32:
        return 4
    if dtype == XTS3DType.INT8:
        return 1
    if dtype == XTS3DType.UINT8:
        return 1
    if dtype == XTS3DType.INT32:
        return 4
    raise ValueError(f"Unsupported ESP32-S3 dtype: {dtype}")


TENSOR_FLAG_CONSTANT = 1 << 0
TENSOR_FLAG_QUANTIZED = 1 << 1
TENSOR_FLAG_GRAPH_INPUT = 1 << 2
TENSOR_FLAG_GRAPH_OUTPUT = 1 << 3


@dataclass
class XTS3Tensor:
    tensor_id: int
    shape: tuple[int, ...]
    dtype: XTS3DType
    data_offset: int = 0
    data_size: int = 0
    flags: int = 0
    quant_scale: float = 0.0
    quant_zero_point: int = 0

    @property
    def is_constant(self) -> bool:
        return bool(self.flags & TENSOR_FLAG_CONSTANT)


@dataclass
class XTS3Operation:
    op_id: int
    opcode: XTS3OpCode
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    params: bytes = b""


@dataclass
class XTS3Graph:
    tensors: list[XTS3Tensor] = field(default_factory=list)
    operations: list[XTS3Operation] = field(default_factory=list)
    constant_data: bytes = b""
    version: int = XTS3_VERSION
    tensor_names: dict[int, str] = field(default_factory=dict)
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)


class ExportedTensorResolver:
    def __init__(self, exported_program):
        self.exported_program = exported_program
        self.graph_signature = getattr(
            exported_program,
            "graph_signature",
            None,
        )
        self.state_dict = (
            getattr(
                exported_program,
                "state_dict",
                {},
            )
            or {}
        )

    def _lookup_mapping(self, mapping, name):
        if mapping is None:
            return None

        if name in mapping:
            return mapping[name]

        for key, value in mapping.items():
            if str(key) == str(name):
                return value

        return None

    def resolve(self, node):  # noqa: C901
        if node is not None and node.op == "call_function":
            target = str(node.target)
            if "dequantize_per_channel" in target:
                quantized = self.resolve(node.args[0])
                scales = self.resolve(node.args[1])
                zero_points = self.resolve(node.args[2])
                axis = int(node.args[3])
                if quantized is None or scales is None or zero_points is None:
                    return None
                view = [1] * quantized.dim()
                view[axis] = -1
                scale_values = scales.to(torch.float32).reshape(view)
                zero_values = zero_points.to(torch.float32).reshape(view)
                return (quantized.to(torch.float32) - zero_values) * scale_values
        if node.op != "placeholder":
            return None

        name = node.name
        target = node.target

        parameter_mapping = getattr(
            self.graph_signature,
            "inputs_to_parameters",
            {},
        )

        buffer_mapping = getattr(
            self.graph_signature,
            "inputs_to_buffers",
            {},
        )

        state_name = self._lookup_mapping(
            parameter_mapping,
            name,
        )

        if state_name is None:
            state_name = self._lookup_mapping(
                parameter_mapping,
                target,
            )

        if state_name is None:
            state_name = self._lookup_mapping(
                buffer_mapping,
                name,
            )

        if state_name is None:
            state_name = self._lookup_mapping(
                buffer_mapping,
                target,
            )

        if state_name is not None and state_name in self.state_dict:
            return self.state_dict[state_name]

        constants = getattr(self.exported_program, "constants", None) or {}
        constant_mapping = getattr(
            self.graph_signature,
            "inputs_to_lifted_tensor_constants",
            {},
        )
        constant_name = self._lookup_mapping(constant_mapping, name)
        if constant_name is None:
            constant_name = self._lookup_mapping(constant_mapping, target)
        if constant_name is not None and constant_name in constants:
            return constants[constant_name]

        for candidate in (target, name):
            if candidate in self.state_dict:
                return self.state_dict[candidate]
            if candidate in constants:
                return constants[candidate]

        return None


def _shape_from_node(node) -> tuple[int, ...]:
    value = node.meta.get("val")

    if value is None:
        return ()

    shape = getattr(
        value,
        "shape",
        None,
    )

    if shape is None:
        return ()

    return tuple(int(dim) for dim in shape)


def _dtype_from_node(node) -> torch.dtype:
    value = node.meta.get("val")

    dtype = getattr(
        value,
        "dtype",
        None,
    )

    if dtype is None:
        raise ValueError(f"Node {node.name} has no dtype metadata")

    return dtype


def _normalize_int_tuple(
    value,
    length: int,
    name: str,
) -> tuple[int, ...]:

    if isinstance(value, int):
        return (int(value),) * length

    result = tuple(int(x) for x in value)

    if len(result) != length:
        raise ValueError(f"Expected {length} values for " f"{name}, got {result}")

    return result


def _target_text(target) -> str:
    return str(target)


LUT_KINDS = (
    "hardswish",
    "hardsigmoid",
    "sigmoid",
    "silu",
    "tanh",
    "gelu",
    "elu",
    "leaky_relu",
)


def _target_kind(target) -> str | None:  # noqa: C901
    text = _target_text(target)

    if "aten.convolution.default" in text:
        return "convolution"

    if "aten.conv_transpose2d" in text:
        return "conv_transpose2d"

    if "aten.upsample_nearest2d.vec" in text:
        return "upsample_nearest2d_vec"

    if "aten.upsample_nearest2d.default" in text:
        return "upsample_nearest2d"

    if "aten.upsample_bilinear2d.vec" in text:
        return "upsample_bilinear2d_vec"

    if "aten.upsample_bilinear2d.default" in text:
        return "upsample_bilinear2d"

    if "aten.constant_pad_nd.default" in text or "aten.pad.default" in text:
        return "pad"

    for name, kind in (
        ("permute", "permute"),
        ("transpose", "transpose"),
        ("unsqueeze", "unsqueeze"),
        ("squeeze", "squeeze"),
        ("expand", "expand"),
        ("unflatten", "unflatten"),
        ("split_with_sizes", "split_with_sizes"),
        ("unbind", "unbind"),
        ("chunk", "chunk"),
    ):
        if f"aten.{name}." in text or f"aten.{name}_copy." in text:
            return kind

    if "aten.split." in text or "aten.split_copy." in text:
        return "split"

    if "aten.slice.Tensor" in text or "aten.slice_copy.Tensor" in text:
        return "slice"

    if "aten.select.int" in text or "aten.select_copy.int" in text:
        return "select"

    if "aten.conv2d.default" in text:
        return "conv2d"

    if "aten.hardtanh_.default" in text:
        return "hardtanh_"

    if "aten.hardtanh.default" in text:
        return "hardtanh"

    if "aten.relu_.default" in text or "aten.relu.default" in text:
        return "relu"

    if "aten.relu6.default" in text:
        return "relu6"

    if "aten.add.Tensor" in text or "aten.add_.Tensor" in text:
        return "add"

    if "aten.mul.Tensor" in text or "aten.mul_.Tensor" in text:
        return "mul"

    if "aten._softmax.default" in text or "aten.softmax.int" in text:
        return "softmax"

    for name in LUT_KINDS:
        if f"aten.{name}.default" in text or f"aten.{name}_.default" in text:
            return name

    if "aten.adaptive_avg_pool2d.default" in text:
        return "adaptive_avg_pool2d"

    if "aten.max_pool2d_with_indices.default" in text:
        return "max_pool2d_with_indices"

    if "aten.max_pool2d.default" in text:
        return "max_pool2d"

    if "aten.avg_pool2d.default" in text:
        return "avg_pool2d"

    if "aten.cat.default" in text:
        return "cat"

    if "aten.stack.default" in text:
        return "stack"

    if any(
        marker in text
        for marker in (
            "aten.clone.default",
            "_clone_dim_order.default",
            "aten.alias.default",
            "aten.alias_copy.default",
            "aten.detach.default",
            "aten.detach_copy.default",
        )
    ):
        return "identity"

    if "aten.dropout.default" in text or "aten.dropout_.default" in text:
        return "dropout"

    if text == "<built-in function getitem>" or text.endswith("operator.getitem"):
        return "getitem"

    if any(
        marker in text
        for marker in (
            "aten.view.default",
            "aten.view_copy.default",
            "aten.reshape.default",
            "aten._unsafe_view.default",
        )
    ):
        return "reshape"

    if "aten.mean.dim" in text:
        return "mean_dim"

    if "aten.flatten.using_ints" in text:
        return "flatten"

    if "aten.linear.default" in text:
        return "linear"

    return None


def _int8_device_bytes(values: torch.Tensor, shape: tuple[int, ...]) -> bytes:
    values = values.to(torch.int8).reshape(shape)
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError(f"Rank-4 int8 constants must have batch 1, got {shape}")
        values = values[0].permute(1, 2, 0)
    return values.contiguous().numpy().tobytes()


def _int8_from_device(values: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 4:
        return (
            values.reshape(shape[2], shape[3], shape[1]).permute(2, 0, 1).unsqueeze(0)
        )
    return values.reshape(shape)


class TensorRegistry:
    def __init__(self, resolver: ExportedTensorResolver, graph_module):
        self.resolver = resolver
        self.tensors: list[XTS3Tensor] = []
        self.node_to_id: dict[object, int] = {}
        self.constant_data = bytearray()
        self.fused_nodes: set = set()
        self.extra_operations: list = []
        self.node_to_qparams: dict[object, tuple[float, int]] = {}
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or "quantize_per_tensor" not in str(
                node.target
            ):
                continue
            if len(node.args) < 3 or not hasattr(node.args[0], "meta"):
                continue
            self.node_to_qparams[node.args[0]] = (
                float(node.args[1]),
                int(node.args[2]),
            )

    def _align_constant_data(self, alignment: int = 16) -> None:
        self.constant_data.extend(b"\x00" * ((-len(self.constant_data)) % alignment))

    def get_or_create(self, node, filter_layout: str | None = None) -> int:
        return self._get_or_create_int8(node, filter_layout)

    def _append_constant(self, tensor: XTS3Tensor, raw: bytes) -> int:
        self._align_constant_data()
        tensor.data_offset = len(self.constant_data)
        tensor.data_size = len(raw)
        tensor.flags |= TENSOR_FLAG_CONSTANT
        self.constant_data.extend(raw)
        self.tensors.append(tensor)
        return tensor.tensor_id

    def _new_int8_activation(self, node, scale: float, zero_point: int) -> int:
        if node in self.node_to_id:
            tensor_id = self.node_to_id[node]
            tensor = self.tensors[tensor_id]
            if tensor.quant_scale != scale or tensor.quant_zero_point != zero_point:
                raise ValueError(
                    f"Tensor {node.name} has inconsistent activation qparams"
                )
            return tensor_id
        shape = _shape_from_node(node)
        if not shape:
            raise ValueError(f"Quantized activation {node.name} has no shape metadata")
        tensor_id = len(self.tensors)
        tensor = XTS3Tensor(
            tensor_id=tensor_id,
            shape=shape,
            dtype=XTS3DType.INT8,
            quant_scale=float(scale),
            quant_zero_point=int(zero_point),
        )
        self.tensors.append(tensor)
        self.node_to_id[node] = tensor_id
        return tensor_id

    def _constant_int8(self, node) -> int:
        quantized_node = node.args[0]
        if quantized_node in self.node_to_id:
            return self.node_to_id[quantized_node]
        quantized = self.resolver.resolve(quantized_node)
        if quantized is None or quantized.dtype != torch.int8:
            raise ValueError(
                f"Expected quantize_per_tensor or an int8 constant before {node.name}"
            )
        shape = tuple(int(dim) for dim in quantized.shape)
        if len(shape) > 4:
            raise ValueError(
                f"Constant {node.name} has rank {len(shape)}; the limit is 4"
            )
        tensor_id = len(self.tensors)
        tensor = XTS3Tensor(
            tensor_id=tensor_id,
            shape=shape,
            dtype=XTS3DType.INT8,
            quant_scale=float(node.args[1]),
            quant_zero_point=int(node.args[2]),
        )
        self._append_constant(
            tensor, _int8_device_bytes(quantized.detach().cpu(), shape)
        )
        self.node_to_id[quantized_node] = tensor_id
        return tensor_id

    def broadcast_constant(self, tensor_id: int, shape: tuple[int, ...]) -> int:
        source = self.tensors[tensor_id]
        values = torch.frombuffer(
            bytearray(
                self.constant_data[
                    source.data_offset : source.data_offset + source.data_size
                ]
            ),
            dtype=torch.int8,
        )
        logical = _int8_from_device(values, source.shape)
        expanded = (
            logical.reshape(
                (1,) * (len(shape) - len(source.shape)) + tuple(source.shape)
            )
            .expand(shape)
            .contiguous()
        )
        result_id = len(self.tensors)
        result = XTS3Tensor(
            tensor_id=result_id,
            shape=tuple(shape),
            dtype=XTS3DType.INT8,
            quant_scale=source.quant_scale,
            quant_zero_point=source.quant_zero_point,
        )
        self._append_constant(result, _int8_device_bytes(expanded, tuple(shape)))
        return result_id

    def _get_or_create_int8(  # noqa: C901
        self, node, filter_layout: str | None = None
    ) -> int:
        if node.op == "call_function" and "dequantize_per_tensor" in str(node.target):
            quantized_node = node.args[0]
            if not (
                getattr(quantized_node, "op", None) == "call_function"
                and "quantize_per_tensor" in str(quantized_node.target)
            ):
                if (
                    quantized_node.op == "placeholder"
                    and self.resolver.resolve(quantized_node) is None
                ):
                    return self._new_int8_activation(
                        quantized_node, float(node.args[1]), int(node.args[2])
                    )
                return self._constant_int8(node)
            return self._new_int8_activation(
                quantized_node.args[0],
                float(node.args[1]),
                int(node.args[2]),
            )

        if node.op == "call_function" and "dequantize_per_channel" in str(node.target):
            quantized_node = node.args[0]
            if quantized_node in self.node_to_id:
                return self.node_to_id[quantized_node]
            quantized = self.resolver.resolve(quantized_node)
            if quantized is None or quantized.dtype != torch.int8:
                raise ValueError(
                    f"Quantized weight {node.name} is not backed by int8 data"
                )
            expected_axis = 1 if filter_layout == "TRANSPOSED" else 0
            if int(node.args[3]) != expected_axis:
                raise ValueError(
                    f"Only output-channel weight quantization is supported: {node.name}"
                )
            if not torch.all(self.resolver.resolve(node.args[2]) == 0):
                raise ValueError(
                    f"Only symmetric int8 weights are supported: {node.name}"
                )
            shape = tuple(int(dim) for dim in quantized.shape)
            tensor_id = len(self.tensors)
            tensor = XTS3Tensor(
                tensor_id=tensor_id,
                shape=shape,
                dtype=XTS3DType.INT8,
            )
            raw = quantized.detach().cpu().contiguous()
            if len(shape) == 4 and filter_layout == "OHWI":
                raw = raw.permute(0, 2, 3, 1).contiguous()
            elif len(shape) == 4 and filter_layout == "TRANSPOSED":
                raw = raw.permute(1, 2, 3, 0).contiguous()
                tensor.shape = (shape[1], shape[2], shape[3], shape[0])
            elif len(shape) == 4 and filter_layout == "HWOC":
                raw = (
                    raw.permute(2, 3, 0, 1)
                    .reshape(shape[2], shape[3], shape[0])
                    .contiguous()
                )
            elif len(shape) == 4:
                raise ValueError(
                    f"A convolution filter layout is required for {node.name}"
                )
            raw_bytes = raw.numpy().tobytes()
            self._append_constant(tensor, raw_bytes)
            self.node_to_id[quantized_node] = tensor_id
            return tensor_id

        if node in self.node_to_id:
            return self.node_to_id[node]

        quantization = self.node_to_qparams.get(node)
        if quantization is None:
            raise ValueError(
                f"Activation {node.name} has no PT2E quantization parameters"
            )
        return self._new_int8_activation(node, *quantization)

    def add_int32_bias(
        self, node, input_scale: float, weight_scales: torch.Tensor
    ) -> int:
        scales = weight_scales.detach().cpu().to(torch.float64).reshape(-1)
        if node is None:
            bias = torch.zeros(scales.numel(), dtype=torch.float64)
        else:
            bias = self.resolver.resolve(node)
            if bias is None:
                raise ValueError(f"Unable to resolve bias tensor {node.name}")
            bias = bias.detach().cpu().to(torch.float64).reshape(-1)
        if bias.numel() != scales.numel():
            raise ValueError("Bias and per-channel weight scale counts differ")
        bias_q = torch.round(bias / (float(input_scale) * scales))
        if torch.any(bias_q < -(1 << 31)) or torch.any(bias_q > (1 << 31) - 1):
            raise ValueError("Quantized bias overflows int32")
        tensor_id = len(self.tensors)
        tensor = XTS3Tensor(
            tensor_id=tensor_id,
            shape=(int(bias_q.numel()),),
            dtype=XTS3DType.INT32,
        )
        raw_bytes = bias_q.to(torch.int32).contiguous().numpy().tobytes()
        self._append_constant(tensor, raw_bytes)
        return tensor_id


def _get_conv_args(node):
    args = node.args

    if len(args) < 2:
        raise ValueError(
            f"Convolution node {node.name} " f"has fewer than two arguments"
        )

    kind = _target_kind(node.target)

    if kind == "convolution":
        if len(args) < 9:
            raise ValueError(
                f"Edge convolution node "
                f"{node.name} has unexpected "
                f"argument count: {len(args)}"
            )

        transposed = bool(args[6])

        output_padding = _normalize_int_tuple(
            args[7],
            2,
            "output_padding",
        )

        if transposed:
            raise ValueError(
                f"Transposed convolution must use the transposed lowering: {node.name}"
            )

        if any(output_padding):
            raise ValueError(
                f"Non-zero output_padding is " f"not supported: {node.name}"
            )

        return {
            "input": args[0],
            "weight": args[1],
            "bias": args[2],
            "stride": _normalize_int_tuple(
                args[3],
                2,
                "stride",
            ),
            "padding": _normalize_int_tuple(
                args[4],
                2,
                "padding",
            ),
            "dilation": _normalize_int_tuple(
                args[5],
                2,
                "dilation",
            ),
            "groups": int(args[8]),
        }

    return {
        "input": args[0],
        "weight": args[1],
        "bias": (args[2] if len(args) > 2 else None),
        "stride": _normalize_int_tuple(
            args[3] if len(args) > 3 else (1, 1),
            2,
            "stride",
        ),
        "padding": _normalize_int_tuple(
            args[4] if len(args) > 4 else (0, 0),
            2,
            "padding",
        ),
        "dilation": _normalize_int_tuple(
            args[5] if len(args) > 5 else (1, 1),
            2,
            "dilation",
        ),
        "groups": (int(args[6]) if len(args) > 6 else 1),
    }


def _is_depthwise_conv(
    node,
    conv_args,
) -> bool:

    groups = conv_args["groups"]

    weight_shape = _shape_from_node(conv_args["weight"])

    input_shape = _shape_from_node(conv_args["input"])

    if len(weight_shape) != 4 or len(input_shape) < 2:
        return False

    return groups > 1 and groups == input_shape[1] and weight_shape[1] == 1


def _is_relu6(node) -> bool:
    if len(node.args) < 3:
        return False

    try:
        minimum = float(node.args[1])
        maximum = float(node.args[2])
    except (
        TypeError,
        ValueError,
    ):
        return False

    return minimum == 0.0 and maximum == 6.0


def _quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    if not real_multiplier > 0.0:
        raise ValueError(
            f"Quantized multiplier must be positive, got {real_multiplier}"
        )
    significand, exponent = frexp(real_multiplier)
    multiplier = round(significand * (1 << 31))
    if multiplier == (1 << 31):
        multiplier >>= 1
        exponent += 1
    if multiplier > 0x7FFFFFFF:
        multiplier = 0x7FFFFFFF
    if exponent < -31 or exponent > 30:
        raise ValueError(
            f"Requantization multiplier {real_multiplier} is outside the ESP-NN shift range"
        )
    return int(multiplier), int(exponent)


def _quantize_multiplier_smaller_than_one(real_multiplier: float) -> tuple[int, int]:
    if not 0.0 < real_multiplier < 1.0:
        raise ValueError(f"Expected a multiplier in (0, 1), got {real_multiplier}")
    multiplier, shift = _quantize_multiplier(real_multiplier)
    if shift > 0:
        raise ValueError(f"Unexpected positive shift for {real_multiplier}")
    return multiplier, shift


def _per_channel_weight_scales(registry: TensorRegistry, weight_node) -> torch.Tensor:
    if weight_node.op != "call_function" or "dequantize_per_channel" not in str(
        weight_node.target
    ):
        raise ValueError(
            f"Expected per-channel quantized weights at {weight_node.name}"
        )
    scales = registry.resolver.resolve(weight_node.args[1])
    zero_points = registry.resolver.resolve(weight_node.args[2])
    if scales is None or zero_points is None:
        raise ValueError(
            f"Unable to resolve per-channel weight qparams for {weight_node.name}"
        )
    scales = scales.detach().cpu().to(torch.float64).reshape(-1)
    if not torch.all(torch.isfinite(scales)) or torch.any(scales <= 0):
        raise ValueError(f"Invalid per-channel weight scales for {weight_node.name}")
    if not torch.all(zero_points == 0):
        raise ValueError(
            f"Only symmetric per-channel weights are supported for {weight_node.name}"
        )
    return scales


def _serialize_per_channel_requantization(
    input_scale: float,
    weight_scales: torch.Tensor,
    output_scale: float,
) -> bytes:
    multipliers = []
    shifts = []
    for weight_scale in weight_scales.tolist():
        multiplier, shift = _quantize_multiplier(
            float(input_scale) * float(weight_scale) / float(output_scale)
        )
        multipliers.append(multiplier)
        shifts.append(shift)
    return struct.pack(f"<{len(multipliers)}i", *multipliers) + struct.pack(
        f"<{len(shifts)}i", *shifts
    )


def _round_half_away(value: float) -> int:
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


def _activation_range(
    kind: str | None, scale: float, zero_point: int
) -> tuple[int, int]:
    if kind is None:
        return -128, 127
    low = max(-128, zero_point)
    if kind == "relu":
        return low, 127
    if kind == "relu6":
        return low, min(127, zero_point + _round_half_away(6.0 / scale))
    raise ValueError(f"Unsupported fused activation {kind}")


def _int8_activation_kind(node) -> str | None:
    kind = _target_kind(node.target) if node.op == "call_function" else None
    if kind == "relu":
        return "relu"
    if kind == "relu6":
        return "relu6"
    if kind in ("hardtanh", "hardtanh_") and _is_relu6(node):
        return "relu6"
    return None


def _fusable_activation(node, registry: TensorRegistry):
    users = list(node.users)
    if len(users) != 1:
        return None, None
    activation = users[0]
    kind = _int8_activation_kind(activation)
    if kind is None or activation.args[0] is not node:
        return None, None
    if activation not in registry.node_to_qparams:
        return None, None
    return activation, kind


def _require_output_qparams(registry: TensorRegistry, node) -> None:
    if node not in registry.node_to_qparams:
        raise ValueError(
            f"Int8 operation {node.name} has no output quantization; "
            "quantize the model with EspNNQuantizer before lowering"
        )


def lower_int8_node(  # noqa: C901
    node, registry: TensorRegistry
) -> XTS3Operation | None:
    if node.op != "call_function" or node in registry.fused_nodes:
        return None
    kind = _target_kind(node.target)

    if kind == "conv_transpose2d" or (
        kind == "convolution" and len(node.args) > 6 and bool(node.args[6])
    ):
        from .int8_ops import lower_transposed_conv

        return lower_transposed_conv(node, registry)

    if kind in (
        "upsample_nearest2d",
        "upsample_nearest2d_vec",
        "upsample_bilinear2d",
        "upsample_bilinear2d_vec",
    ):
        from .int8_ops import lower_upsample

        return lower_upsample(node, registry)

    if kind == "pad":
        from .int8_ops import lower_pad

        return lower_pad(node, registry)

    if kind in ("split", "split_with_sizes", "unbind", "chunk"):
        registry.fused_nodes.add(node)
        return None

    if kind in (
        "permute",
        "transpose",
        "squeeze",
        "unsqueeze",
        "slice",
        "select",
        "expand",
        "unflatten",
    ) or (
        kind == "getitem"
        and _target_kind(getattr(node.args[0], "target", None))
        in ("split", "split_with_sizes", "unbind", "chunk")
    ):
        from .int8_ops import lower_movement

        return lower_movement(node, registry, registry.extra_operations)

    if kind in ("conv2d", "convolution"):
        conv = _get_conv_args(node)
        if (
            max(conv["dilation"]) > 255
            or max(conv["padding"]) > 255
            or max(conv["stride"]) > 255
        ):
            raise ValueError(
                f"Convolution geometry exceeds the 8-bit parameter fields: {node.name}"
            )
        depthwise = _is_depthwise_conv(node, conv)
        activation, activation_kind = _fusable_activation(node, registry)
        output_node = activation if activation is not None else node
        _require_output_qparams(registry, output_node)
        input_id = registry.get_or_create(conv["input"])
        weight_scales = _per_channel_weight_scales(registry, conv["weight"])
        weight_id = registry.get_or_create(
            conv["weight"], "HWOC" if depthwise else "OHWI"
        )
        input_tensor = registry.tensors[input_id]
        bias_id = registry.add_int32_bias(
            conv["bias"], input_tensor.quant_scale, weight_scales
        )
        output_id = registry.get_or_create(output_node)
        output_tensor = registry.tensors[output_id]
        if activation is not None:
            registry.fused_nodes.add(activation)
        weight_shape = _shape_from_node(conv["weight"])
        if len(weight_shape) != 4:
            raise ValueError(f"Convolution weight must be rank 4: {node.name}")
        act_min, act_max = _activation_range(
            activation_kind, output_tensor.quant_scale, output_tensor.quant_zero_point
        )
        params = struct.pack(
            INT8_CONV_BASE_PARAMS,
            weight_shape[2],
            weight_shape[3],
            conv["stride"][0],
            conv["stride"][1],
            conv["padding"][0],
            conv["padding"][1],
            conv["dilation"][0],
            conv["dilation"][1],
            conv["groups"],
            act_min,
            act_max,
        ) + _serialize_per_channel_requantization(
            input_tensor.quant_scale, weight_scales, output_tensor.quant_scale
        )
        return XTS3Operation(
            op_id=-1,
            opcode=(XTS3OpCode.DEPTHWISE_CONV2D if depthwise else XTS3OpCode.CONV2D),
            inputs=[input_id, weight_id, bias_id],
            outputs=[output_id],
            params=params,
        )

    if kind in ("hardtanh_", "hardtanh", "relu", "relu6"):
        activation_kind = _int8_activation_kind(node)
        _require_output_qparams(registry, node)
        input_id = registry.get_or_create(node.args[0])
        output_id = registry.get_or_create(node)
        source = registry.tensors[input_id]
        result = registry.tensors[output_id]
        if activation_kind is None or (
            source.quant_scale != result.quant_scale
            or source.quant_zero_point != result.quant_zero_point
        ):
            return _lut_operation(node, input_id, output_id, registry)
        act_min, act_max = _activation_range(
            activation_kind, result.quant_scale, result.quant_zero_point
        )
        return XTS3Operation(
            op_id=-1,
            opcode=XTS3OpCode.RELU6,
            inputs=[input_id],
            outputs=[output_id],
            params=struct.pack(INT8_PAIR_PARAMS, act_min, act_max),
        )

    if kind in LUT_KINDS:
        _require_output_qparams(registry, node)
        input_id = registry.get_or_create(node.args[0])
        output_id = registry.get_or_create(node)
        return _lut_operation(node, input_id, output_id, registry)

    if kind in ("add", "mul"):
        scalar_operation = _lower_scalar_binary(node, kind, registry)
        if scalar_operation is not None:
            return scalar_operation

    if kind == "mul":
        return _lower_int8_mul(node, registry)

    if kind == "softmax":
        return _lower_int8_softmax(node, registry)

    if kind == "add":
        activation, activation_kind = _fusable_activation(node, registry)
        output_node = activation if activation is not None else node
        _require_output_qparams(registry, output_node)
        from .int8_ops import broadcast_operand

        output_id = registry.get_or_create(output_node)
        result = registry.tensors[output_id]
        lhs_id = broadcast_operand(
            node.args[0], result.shape, registry, registry.extra_operations
        )
        rhs_id = broadcast_operand(
            node.args[1], result.shape, registry, registry.extra_operations
        )
        if activation is not None:
            registry.fused_nodes.add(activation)
        lhs = registry.tensors[lhs_id]
        rhs = registry.tensors[rhs_id]
        if lhs.shape != rhs.shape or lhs.shape != result.shape:
            raise ValueError(f"Broadcasting add could not be resolved: {node.name}")
        twice_max_input_scale = 2.0 * max(lhs.quant_scale, rhs.quant_scale)
        lhs_mult, lhs_shift = _quantize_multiplier_smaller_than_one(
            lhs.quant_scale / twice_max_input_scale
        )
        rhs_mult, rhs_shift = _quantize_multiplier_smaller_than_one(
            rhs.quant_scale / twice_max_input_scale
        )
        out_mult, out_shift = _quantize_multiplier_smaller_than_one(
            twice_max_input_scale / ((1 << ADD_LEFT_SHIFT) * result.quant_scale)
        )
        act_min, act_max = _activation_range(
            activation_kind, result.quant_scale, result.quant_zero_point
        )
        return XTS3Operation(
            op_id=-1,
            opcode=XTS3OpCode.ADD,
            inputs=[lhs_id, rhs_id],
            outputs=[output_id],
            params=struct.pack(
                INT8_ADD_PARAMS,
                lhs_mult,
                lhs_shift,
                rhs_mult,
                rhs_shift,
                ADD_LEFT_SHIFT,
                out_mult,
                out_shift,
                act_min,
                act_max,
            ),
        )

    if kind in ("adaptive_avg_pool2d", "mean_dim"):
        if kind == "adaptive_avg_pool2d":
            output_size = (
                tuple(int(v) for v in node.args[1]) if len(node.args) > 1 else ()
            )
            if output_size != (1, 1):
                return _lower_adaptive_avg_pool(node, output_size, registry)
        else:
            if (
                len(node.args) < 3
                or tuple(int(d) for d in node.args[1]) not in ((2, 3), (-2, -1))
                or not bool(node.args[2])
            ):
                raise ValueError(f"Only spatial keepdim mean is supported: {node.name}")
        _require_output_qparams(registry, node)
        input_id = registry.get_or_create(node.args[0])
        output_id = registry.get_or_create(node)
        source = registry.tensors[input_id]
        result = registry.tensors[output_id]
        if len(source.shape) != 4:
            raise ValueError(f"Average pooling input must be rank 4: {node.name}")
        elements = source.shape[2] * source.shape[3]
        mult, shift = _quantize_multiplier(
            source.quant_scale / (result.quant_scale * elements)
        )
        return XTS3Operation(
            op_id=-1,
            opcode=XTS3OpCode.AVGPOOL2D,
            inputs=[input_id],
            outputs=[output_id],
            params=struct.pack(INT8_PAIR_PARAMS, mult, shift),
        )

    if kind in ("max_pool2d", "avg_pool2d"):
        return _lower_int8_pool(node, kind, registry)

    if kind == "max_pool2d_with_indices":
        users = list(node.users)
        if any(
            _target_kind(user.target) != "getitem" or user.args[1] != 0
            for user in users
        ):
            raise ValueError(
                f"max_pool2d indices are used by {node.name}; only values are supported"
            )
        if len(users) != 1:
            raise ValueError(
                f"max_pool2d_with_indices {node.name} must have one getitem user"
            )
        operation = _lower_int8_pool(users[0], "max_pool2d", registry, source_node=node)
        registry.fused_nodes.add(users[0])
        return operation

    if kind in ("identity", "dropout"):
        from .int8_ops import lower_movement, node_kind

        if node_kind(node) == "identity_move":
            return lower_movement(node, registry, registry.extra_operations)

    if kind in ("identity", "dropout", "getitem"):
        if kind == "dropout" and len(node.args) > 2 and bool(node.args[2]):
            raise ValueError(f"Training-mode dropout cannot be lowered: {node.name}")
        if kind == "getitem":
            raise ValueError(f"Unsupported tuple access {node.name}")
        source_id = registry.get_or_create(node.args[0])
        quantization = registry.node_to_qparams.get(node)
        tensor = registry.tensors[source_id]
        if quantization is not None and quantization != (
            tensor.quant_scale,
            tensor.quant_zero_point,
        ):
            raise ValueError(f"Identity op {node.name} changes quantization parameters")
        registry.node_to_id[node] = source_id
        registry.fused_nodes.add(node)
        return None

    if kind == "stack":
        from .int8_ops import lower_cat

        return lower_cat(node, registry, registry.extra_operations, stack=True)

    if kind == "cat":
        try:
            return _lower_int8_concat(node, registry)
        except ValueError:
            from .int8_ops import lower_cat

            return lower_cat(node, registry, registry.extra_operations)

    if kind in ("flatten", "reshape"):
        from .int8_ops import (
            _skip_quantization,
            is_movement_terminal,
            lower_movement,
            MOVEMENT_KINDS,
            node_kind,
        )

        if not is_movement_terminal(node):
            registry.fused_nodes.add(node)
            return None
        _require_output_qparams(registry, node)
        producer = _skip_quantization(node.args[0])
        result_shape = _shape_from_node(node)
        if (
            node_kind(producer) in MOVEMENT_KINDS
            or len(result_shape) != 2
            or result_shape[0] != 1
        ):
            return lower_movement(node, registry, registry.extra_operations)
        input_id = registry.get_or_create(node.args[0])
        output_id = registry.get_or_create(node)
        source = registry.tensors[input_id]
        result = registry.tensors[output_id]
        if (
            source.quant_scale != result.quant_scale
            or source.quant_zero_point != result.quant_zero_point
        ):
            return lower_movement(node, registry, registry.extra_operations)
        return XTS3Operation(
            op_id=-1,
            opcode=XTS3OpCode.FLATTEN,
            inputs=[input_id],
            outputs=[output_id],
        )

    if kind == "linear":
        activation, activation_kind = _fusable_activation(node, registry)
        output_node = activation if activation is not None else node
        _require_output_qparams(registry, output_node)
        input_shape = _shape_from_node(node.args[0])
        if len(input_shape) == 4 or len(input_shape) < 1:
            raise ValueError(
                f"linear {node.name} needs a rank 1 to 3 input, got {input_shape}"
            )
        input_id = registry.get_or_create(node.args[0])
        weight_scales = _per_channel_weight_scales(registry, node.args[1])
        weight_id = registry.get_or_create(node.args[1])
        input_tensor = registry.tensors[input_id]
        bias_id = registry.add_int32_bias(
            node.args[2] if len(node.args) > 2 else None,
            input_tensor.quant_scale,
            weight_scales,
        )
        output_id = registry.get_or_create(output_node)
        if activation is not None:
            registry.fused_nodes.add(activation)
        output_tensor = registry.tensors[output_id]
        act_min, act_max = _activation_range(
            activation_kind, output_tensor.quant_scale, output_tensor.quant_zero_point
        )
        params = struct.pack(INT8_PAIR_PARAMS, act_min, act_max) + (
            _serialize_per_channel_requantization(
                input_tensor.quant_scale, weight_scales, output_tensor.quant_scale
            )
        )
        return XTS3Operation(
            op_id=-1,
            opcode=XTS3OpCode.LINEAR,
            inputs=[input_id, weight_id, bias_id],
            outputs=[output_id],
            params=params,
        )

    return None


def _unary_function(node, kind: str):
    import torch.nn.functional as F

    args = list(node.args)
    kwargs = dict(node.kwargs)
    if kind == "hardswish":
        return F.hardswish
    if kind == "hardsigmoid":
        return F.hardsigmoid
    if kind == "sigmoid":
        return torch.sigmoid
    if kind == "silu":
        return F.silu
    if kind == "tanh":
        return torch.tanh
    if kind == "gelu":
        approximate = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        return lambda x: F.gelu(x, approximate=approximate)
    if kind == "elu":
        alpha = float(args[1]) if len(args) > 1 else float(kwargs.get("alpha", 1.0))
        scale = float(args[2]) if len(args) > 2 else float(kwargs.get("scale", 1.0))
        input_scale = (
            float(args[3]) if len(args) > 3 else float(kwargs.get("input_scale", 1.0))
        )
        return lambda x: torch.ops.aten.elu(x, alpha, scale, input_scale)
    if kind == "leaky_relu":
        slope = (
            float(args[1])
            if len(args) > 1
            else float(kwargs.get("negative_slope", 0.01))
        )
        return lambda x: F.leaky_relu(x, slope)
    if kind == "relu":
        return torch.relu
    if kind == "relu6":
        return lambda x: torch.clamp(x, 0.0, 6.0)
    if kind in ("hardtanh", "hardtanh_"):
        low = float(args[1]) if len(args) > 1 else -1.0
        high = float(args[2]) if len(args) > 2 else 1.0
        return lambda x: torch.clamp(x, low, high)
    raise ValueError(f"No lookup-table function for {node.name}")


def build_lut(function, source: XTS3Tensor, result: XTS3Tensor) -> bytes:
    codes = torch.arange(-128, 128, dtype=torch.int32).to(torch.int8)
    values = torch.ops.quantized_decomposed.dequantize_per_tensor(
        codes, source.quant_scale, source.quant_zero_point, -128, 127, torch.int8
    )
    outputs = function(values.to(torch.float32)).to(torch.float32)
    if not torch.all(torch.isfinite(outputs)):
        raise ValueError("Lookup-table function produced non-finite values")
    quantized = torch.ops.quantized_decomposed.quantize_per_tensor(
        outputs, result.quant_scale, result.quant_zero_point, -128, 127, torch.int8
    )
    return quantized.contiguous().numpy().tobytes()


def _lut_operation(
    node, input_id: int, output_id: int, registry: TensorRegistry, function=None
):
    kind = _target_kind(node.target)
    function = function or _unary_function(node, kind)
    table = build_lut(function, registry.tensors[input_id], registry.tensors[output_id])
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.LUT,
        inputs=[input_id],
        outputs=[output_id],
        params=table,
    )


def _constant_scalar(node, registry: TensorRegistry):
    if isinstance(node, (int, float)):
        return float(node)
    if not hasattr(node, "op"):
        return None
    value = None
    if node.op == "call_function" and "dequantize_per_tensor" in str(node.target):
        quantized = registry.resolver.resolve(node.args[0])
        if quantized is not None:
            value = torch.ops.quantized_decomposed.dequantize_per_tensor(
                quantized,
                float(node.args[1]),
                int(node.args[2]),
                int(node.args[3]),
                int(node.args[4]),
                quantized.dtype,
            )
    elif node.op in ("placeholder", "get_attr"):
        value = registry.resolver.resolve(node)
    if value is None or value.numel() != 1:
        return None
    return float(value.reshape(-1)[0])


def _reject_constant_operands(node, registry: TensorRegistry) -> None:
    for arg in node.args[:2]:
        if not hasattr(arg, "op"):
            continue
        source = (
            arg.args[0]
            if (arg.op == "call_function" and "dequantize_per" in str(arg.target))
            else arg
        )
        if hasattr(source, "op") and registry.resolver.resolve(source) is not None:
            raise ValueError(
                f"{node.name} uses a constant tensor operand; only scalar constants are supported"
            )


def _lower_scalar_binary(node, kind: str, registry: TensorRegistry):
    lhs, rhs = node.args[0], node.args[1]
    constant = _constant_scalar(rhs, registry)
    tensor_arg = lhs
    if constant is None:
        constant = _constant_scalar(lhs, registry)
        tensor_arg = rhs
    if constant is None:
        return None
    alpha = float(node.kwargs.get("alpha", 1.0)) if kind == "add" else 1.0
    _require_output_qparams(registry, node)
    input_id = registry.get_or_create(tensor_arg)
    output_id = registry.get_or_create(node)
    tensor_on_left = tensor_arg is lhs

    def function(x):
        if kind != "add":
            return x * constant
        return x + constant * alpha if tensor_on_left else constant + alpha * x

    return _lut_operation(node, input_id, output_id, registry, function)


def _lower_int8_mul(node, registry: TensorRegistry) -> XTS3Operation:
    activation, activation_kind = _fusable_activation(node, registry)
    output_node = activation if activation is not None else node
    _require_output_qparams(registry, output_node)
    from .int8_ops import broadcast_operand

    lhs_id = registry.get_or_create(node.args[0])
    rhs_id = registry.get_or_create(node.args[1])
    output_id = registry.get_or_create(output_node)
    if activation is not None:
        registry.fused_nodes.add(activation)
    lhs = registry.tensors[lhs_id]
    rhs = registry.tensors[rhs_id]
    result = registry.tensors[output_id]

    def per_channel_pattern(full, small):
        return (
            len(full.shape) == 4
            and len(small.shape) == 4
            and full.shape == result.shape
            and small.shape == (1, full.shape[1], 1, 1)
            and full.shape[0] == 1
            and not small.is_constant
        )

    if not (lhs.shape == rhs.shape == result.shape) and not (
        per_channel_pattern(lhs, rhs) or per_channel_pattern(rhs, lhs)
    ):
        lhs_id = broadcast_operand(
            node.args[0], result.shape, registry, registry.extra_operations
        )
        rhs_id = broadcast_operand(
            node.args[1], result.shape, registry, registry.extra_operations
        )
        lhs = registry.tensors[lhs_id]
        rhs = registry.tensors[rhs_id]
    if lhs.shape == rhs.shape == result.shape:
        mode = MUL_MODE_ELEMENTWISE
        inputs = [lhs_id, rhs_id]
    else:

        def per_channel(full, small):
            return (
                len(full.shape) == 4
                and len(small.shape) == 4
                and full.shape == result.shape
                and small.shape == (1, full.shape[1], 1, 1)
                and full.shape[0] == 1
            )

        if per_channel(lhs, rhs):
            inputs = [lhs_id, rhs_id]
        elif per_channel(rhs, lhs):
            inputs = [rhs_id, lhs_id]
        else:
            raise ValueError(
                f"mul {node.name} broadcasts {lhs.shape} x {rhs.shape}; only identical shapes "
                "or (1,C,H,W) x (1,C,1,1) are supported"
            )
        mode = MUL_MODE_CHANNEL_BROADCAST
    multiplier, shift = _quantize_multiplier(
        lhs.quant_scale * rhs.quant_scale / result.quant_scale
    )
    act_min, act_max = _activation_range(
        activation_kind, result.quant_scale, result.quant_zero_point
    )
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.MUL,
        inputs=inputs,
        outputs=[output_id],
        params=struct.pack(INT8_MUL_PARAMS, multiplier, shift, act_min, act_max, mode),
    )


def build_softmax_table(input_scale: float) -> bytes:
    import math

    one = 1 << SOFTMAX_EXP_BITS
    table = [
        int(math.floor(math.exp(-d * input_scale) * one + 0.5)) for d in range(256)
    ]
    return struct.pack("<256I", *table)


def _lower_int8_softmax(node, registry: TensorRegistry) -> XTS3Operation:
    _require_output_qparams(registry, node)
    input_id = registry.get_or_create(node.args[0])
    output_id = registry.get_or_create(node)
    source = registry.tensors[input_id]
    result = registry.tensors[output_id]
    rank = len(source.shape)
    dim = int(node.args[1]) if len(node.args) > 1 else -1
    if dim < 0:
        dim += rank
    if source.shape[0] != 1 or not (
        (rank == 2 and dim == 1) or (rank == 4 and dim == 1)
    ):
        raise ValueError(
            f"Only channel softmax on batch-1 rank 2 or 4 tensors is supported: {node.name}"
        )
    if (
        abs(result.quant_scale - SOFTMAX_OUTPUT_SCALE) > 1e-9
        or result.quant_zero_point != SOFTMAX_OUTPUT_ZERO_POINT
    ):
        raise ValueError(
            f"softmax {node.name} output must use scale 1/256 and zero point -128 "
            "(quantize with EspNNQuantizer)"
        )
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.SOFTMAX,
        inputs=[input_id],
        outputs=[output_id],
        params=build_softmax_table(source.quant_scale),
    )


def _pool_window(value, name: str) -> tuple[int, int]:
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return None
    if isinstance(value, int):
        return (int(value), int(value))
    values = tuple(int(v) for v in value)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) != 2:
        raise ValueError(f"Expected a 2D {name}, got {values}")
    return values


def _lower_int8_pool(
    node, kind: str, registry: TensorRegistry, source_node=None
) -> XTS3Operation:
    source_node = source_node if source_node is not None else node
    args = list(source_node.args)
    kernel = _pool_window(args[1], "kernel_size")
    stride = _pool_window(args[2] if len(args) > 2 else None, "stride") or kernel
    padding = _pool_window(args[3] if len(args) > 3 else 0, "padding") or (0, 0)
    if kind == "max_pool2d":
        dilation = _pool_window(args[4] if len(args) > 4 else 1, "dilation") or (1, 1)
        ceil_mode = bool(args[5]) if len(args) > 5 else False
        if dilation != (1, 1):
            raise ValueError(f"Dilated max pooling is not supported: {node.name}")
        opcode = XTS3OpCode.MAX_POOL2D
    else:
        ceil_mode = bool(args[4]) if len(args) > 4 else False
        count_include_pad = bool(args[5]) if len(args) > 5 else True
        divisor_override = args[6] if len(args) > 6 else None
        if divisor_override is not None:
            raise ValueError(
                f"avg_pool2d divisor_override is not supported: {node.name}"
            )
        if count_include_pad and padding != (0, 0):
            raise ValueError(
                f"avg_pool2d with padding must use count_include_pad=False for ESP-NN: {node.name}"
            )
        opcode = XTS3OpCode.AVG_POOL2D
    if ceil_mode:
        raise ValueError(f"ceil_mode pooling is not supported: {node.name}")
    if padding[0] * 2 > kernel[0] or padding[1] * 2 > kernel[1]:
        raise ValueError(f"Pooling padding exceeds half the window: {node.name}")
    _require_output_qparams(registry, node)
    input_id = registry.get_or_create(source_node.args[0])
    output_id = registry.get_or_create(node)
    source = registry.tensors[input_id]
    result = registry.tensors[output_id]
    if (
        source.quant_scale != result.quant_scale
        or source.quant_zero_point != result.quant_zero_point
    ):
        raise ValueError(f"Pooling {node.name} changes quantization parameters")
    if len(source.shape) != 4 or source.shape[0] != 1:
        raise ValueError(f"Pooling input must be batch-1 rank 4: {node.name}")
    return XTS3Operation(
        op_id=-1,
        opcode=opcode,
        inputs=[input_id],
        outputs=[output_id],
        params=struct.pack(
            INT8_POOL_PARAMS,
            kernel[0],
            kernel[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            1,
            1,
            -128,
            127,
        ),
    )


def _lower_adaptive_avg_pool(
    node, output_size, registry: TensorRegistry
) -> XTS3Operation:
    input_shape = _shape_from_node(node.args[0])
    if len(input_shape) != 4 or len(output_size) != 2:
        raise ValueError(f"Adaptive pooling needs a rank-4 input: {node.name}")
    height, width = input_shape[2], input_shape[3]
    out_h, out_w = output_size
    if height % out_h or width % out_w:
        raise ValueError(
            f"Adaptive pooling {node.name} maps {height}x{width} to {out_h}x{out_w}; "
            "only evenly divisible sizes are supported"
        )
    kernel = (height // out_h, width // out_w)
    if max(kernel) > 255:
        raise ValueError(f"Adaptive pooling window exceeds 255: {node.name}")
    _require_output_qparams(registry, node)
    input_id = registry.get_or_create(node.args[0])
    output_id = registry.get_or_create(node)
    source = registry.tensors[input_id]
    result = registry.tensors[output_id]
    if (
        source.quant_scale != result.quant_scale
        or source.quant_zero_point != result.quant_zero_point
    ):
        raise ValueError(
            f"Adaptive pooling {node.name} changes quantization parameters"
        )
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.AVG_POOL2D,
        inputs=[input_id],
        outputs=[output_id],
        params=struct.pack(
            INT8_POOL_PARAMS,
            kernel[0],
            kernel[1],
            kernel[0],
            kernel[1],
            0,
            0,
            1,
            1,
            -128,
            127,
        ),
    )


def _lower_int8_concat(node, registry: TensorRegistry) -> XTS3Operation:
    sources = list(node.args[0])
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    if not 2 <= len(sources) <= 6:
        raise ValueError(f"Concatenation needs 2 to 6 inputs: {node.name}")
    _require_output_qparams(registry, node)
    input_ids = [registry.get_or_create(source) for source in sources]
    output_id = registry.get_or_create(node)
    result = registry.tensors[output_id]
    rank = len(result.shape)
    if dim < 0:
        dim += rank
    if dim != 1 or rank not in (2, 4) or result.shape[0] != 1:
        raise ValueError(
            f"Only batch-1 channel concatenation is supported: {node.name}"
        )
    for input_id in input_ids:
        tensor = registry.tensors[input_id]
        if (
            tensor.quant_scale != result.quant_scale
            or tensor.quant_zero_point != result.quant_zero_point
        ):
            raise ValueError(
                f"Concatenation inputs of {node.name} use different qparams"
            )
        if len(tensor.shape) != rank or tensor.shape[2:] != result.shape[2:]:
            raise ValueError(
                f"Concatenation inputs of {node.name} have mismatched shapes"
            )
    return XTS3Operation(
        op_id=-1,
        opcode=XTS3OpCode.CONCAT,
        inputs=input_ids,
        outputs=[output_id],
    )


INT8_PASSTHROUGH_MARKERS = (
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "quantize_per_channel",
)


def _is_int8_passthrough(node) -> bool:
    text = str(node.target)
    return any(marker in text for marker in INT8_PASSTHROUGH_MARKERS)


def _quantized_source(node):
    if node.op == "call_function" and "dequantize_per_tensor" in str(node.target):
        quantized = node.args[0]
        if getattr(
            quantized, "op", None
        ) == "call_function" and "quantize_per_tensor" in str(quantized.target):
            return quantized.args[0]
    if (
        node.op == "call_function"
        and "quantize_per_tensor" in str(node.target)
        and "dequantize" not in str(node.target)
    ):
        return node.args[0]
    return None


def _user_input_placeholders(exported_program):
    graph_module = exported_program.graph_module
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    signature = getattr(exported_program, "graph_signature", None)
    if signature is None:
        return placeholders
    user_inputs = set(signature.user_inputs)
    return [node for node in placeholders if node.name in user_inputs]


def _mark_int8_graph_io(
    exported_program, registry: TensorRegistry
) -> tuple[list[int], list[int]]:
    graph_module = exported_program.graph_module
    inputs: list[int] = []
    for node in _user_input_placeholders(exported_program):
        if node not in registry.node_to_id:
            if registry.resolver.resolve(node) is not None:
                continue
            raise ValueError(
                f"Graph input {node.name} is not consumed by an int8 operation"
            )
        tensor = registry.tensors[registry.node_to_id[node]]
        if tensor.is_constant:
            continue
        tensor.flags |= TENSOR_FLAG_GRAPH_INPUT
        inputs.append(tensor.tensor_id)
    if not inputs:
        raise ValueError("Int8 graph has no quantized runtime input")

    outputs: list[int] = []
    for node in graph_module.graph.nodes:
        if node.op != "output":
            continue
        values = node.args[0]
        if not isinstance(values, (list, tuple)):
            values = (values,)
        for value in values:
            source = _quantized_source(value)
            if source is None or source not in registry.node_to_id:
                raise ValueError(
                    f"Graph output {getattr(value, 'name', value)} is not a dequantized int8 tensor"
                )
            registry.tensors[
                registry.node_to_id[source]
            ].flags |= TENSOR_FLAG_GRAPH_OUTPUT
            outputs.append(registry.node_to_id[source])
    if not outputs:
        raise ValueError("Int8 graph has no outputs")
    return inputs, outputs


def build_graph(exported_program) -> XTS3Graph:
    """Lower a PT2E-quantized ExportedProgram (or delegate partition) to an XTS3 int8 graph."""
    graph_module = exported_program.graph_module
    registry = TensorRegistry(ExportedTensorResolver(exported_program), graph_module)
    operations = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        lowered = lower_int8_node(node, registry)
        for extra in registry.extra_operations:
            extra.op_id = len(operations)
            operations.append(extra)
        registry.extra_operations.clear()
        if lowered is None:
            if node not in registry.fused_nodes and not _is_int8_passthrough(node):
                raise ValueError(
                    f"Unsupported operation for the ESP-NN int8 backend: {node.name} ({node.target})"
                )
            continue
        lowered.op_id = len(operations)
        operations.append(lowered)
    graph_inputs, graph_outputs = _mark_int8_graph_io(exported_program, registry)
    return XTS3Graph(
        tensors=registry.tensors,
        operations=operations,
        constant_data=bytes(registry.constant_data),
        version=XTS3_VERSION,
        tensor_names={
            tensor_id: node.name for node, tensor_id in registry.node_to_id.items()
        },
        inputs=graph_inputs,
        outputs=graph_outputs,
    )


def _serialize_tensor(tensor: XTS3Tensor) -> bytes:
    if len(tensor.shape) > 4:
        raise ValueError(f"Tensor rank exceeds the XTS3 limit of 4: {tensor.shape}")
    dims = list(tensor.shape) + [0] * (4 - len(tensor.shape))
    flags = tensor.flags | (TENSOR_FLAG_QUANTIZED if tensor.quant_scale > 0 else 0)
    return struct.pack(
        TENSOR_FORMAT,
        tensor.tensor_id,
        tensor.data_offset,
        int(tensor.dtype),
        flags,
        len(tensor.shape),
        *dims,
        tensor.quant_scale,
        tensor.quant_zero_point,
    )


def _serialize_operation(operation: XTS3Operation, params_offset: int) -> bytes:
    if len(operation.inputs) > 6:
        raise ValueError(f"Operation {operation.op_id} has more than 6 inputs")
    if len(operation.outputs) > 2:
        raise ValueError(f"Operation {operation.op_id} has more than 2 outputs")
    inputs = operation.inputs + [0] * (6 - len(operation.inputs))
    outputs = operation.outputs + [0] * (2 - len(operation.outputs))
    return struct.pack(
        OP_FORMAT,
        operation.op_id & 0xFF,
        int(operation.opcode),
        len(operation.inputs),
        len(operation.outputs),
        *inputs,
        *outputs,
        params_offset,
        len(operation.params),
    )


def serialize_graph(graph: XTS3Graph) -> bytes:
    """Serialize to the XTS3 wire format; constants start 16-byte aligned."""
    if graph.version not in XTS3_VERSIONS:
        raise ValueError(f"Cannot serialize XTS3 version {graph.version}")
    tensor_data = b"".join(_serialize_tensor(tensor) for tensor in graph.tensors)
    parameter_data = bytearray()
    operation_data = bytearray()
    for operation in graph.operations:
        operation_data += _serialize_operation(operation, len(parameter_data))
        parameter_data += operation.params
    io_table = b""
    if graph.version >= 6:
        if len(graph.inputs) > 0xFFFF or len(graph.outputs) > 0xFFFF:
            raise ValueError("Too many graph inputs or outputs")
        io_table = struct.pack(
            IO_TABLE_HEADER, len(graph.inputs), len(graph.outputs)
        ) + struct.pack(
            f"<{len(graph.inputs) + len(graph.outputs)}I", *graph.inputs, *graph.outputs
        )
    prefix = (
        HEADER_SIZE
        + len(io_table)
        + len(tensor_data)
        + len(operation_data)
        + len(parameter_data)
    )
    parameter_data += b"\x00" * ((-prefix) % 16)
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        graph.version,
        len(graph.tensors),
        len(graph.operations),
        len(parameter_data),
        len(graph.constant_data),
    )
    return (
        header
        + io_table
        + tensor_data
        + bytes(operation_data)
        + bytes(parameter_data)
        + bytes(graph.constant_data)
    )


def parse_graph(data: bytes) -> XTS3Graph:  # noqa: C901
    """Parse and bounds-check an XTS3 v5 or v6 int8 graph."""
    if len(data) < HEADER_SIZE:
        raise ValueError("Graph blob is smaller than the header")
    magic, version, tensor_count, operation_count, parameter_size, constant_size = (
        struct.unpack_from(HEADER_FORMAT, data, 0)
    )
    if magic != MAGIC:
        raise ValueError(f"Invalid graph magic: {magic!r}")
    if version not in XTS3_VERSIONS:
        raise ValueError(
            f"Unsupported XTS3 version {version}; expected one of {XTS3_VERSIONS}"
        )
    offset = HEADER_SIZE
    graph_inputs: list[int] = []
    graph_outputs: list[int] = []
    if version >= 6:
        if offset + IO_TABLE_HEADER_SIZE > len(data):
            raise ValueError("Graph IO table exceeds blob")
        input_count, output_count = struct.unpack_from(IO_TABLE_HEADER, data, offset)
        offset += IO_TABLE_HEADER_SIZE
        if offset + 4 * (input_count + output_count) > len(data):
            raise ValueError("Graph IO table exceeds blob")
        ids = struct.unpack_from(f"<{input_count + output_count}I", data, offset)
        graph_inputs, graph_outputs = list(ids[:input_count]), list(ids[input_count:])
        offset += 4 * (input_count + output_count)

    if offset + tensor_count * TENSOR_SIZE > len(data):
        raise ValueError("Graph tensor region exceeds blob")
    tensors = []
    for _ in range(tensor_count):
        fields = struct.unpack_from(TENSOR_FORMAT, data, offset)
        offset += TENSOR_SIZE
        tensor_id, data_offset, dtype_id, flags, rank = fields[:5]
        quant_scale, quant_zero_point = fields[9:11]
        if rank > 4:
            raise ValueError(f"Tensor {tensor_id} has invalid rank {rank}")
        try:
            dtype = XTS3DType(dtype_id)
        except ValueError as exc:
            raise ValueError(
                f"Tensor {tensor_id} has invalid dtype id {dtype_id}"
            ) from exc
        shape = tuple(fields[5 : 5 + rank])
        tensors.append(
            XTS3Tensor(
                tensor_id=tensor_id,
                shape=shape,
                dtype=dtype,
                data_offset=data_offset,
                data_size=(prod(shape) if shape else 1) * dtype_size(dtype),
                flags=flags,
                quant_scale=quant_scale,
                quant_zero_point=quant_zero_point,
            )
        )

    if offset + operation_count * OP_SIZE > len(data):
        raise ValueError("Graph operation region exceeds blob")
    raw_operations = []
    for _ in range(operation_count):
        fields = struct.unpack_from(OP_FORMAT, data, offset)
        offset += OP_SIZE
        op_id, opcode, num_inputs, num_outputs = fields[:4]
        params_offset, params_size = fields[12:14]
        if num_inputs > 6 or num_outputs > 2:
            raise ValueError(f"Operation {op_id} has invalid input or output counts")
        if params_offset + params_size > parameter_size:
            raise ValueError(
                f"Operation {op_id} parameters exceed the parameter region"
            )
        raw_operations.append(
            (
                op_id,
                XTS3OpCode(opcode),
                list(fields[4:10])[:num_inputs],
                list(fields[10:12])[:num_outputs],
                params_offset,
                params_size,
            )
        )
    parameter_base = offset
    constant_base = parameter_base + parameter_size
    if constant_base + constant_size > len(data):
        raise ValueError("Graph constant region exceeds blob")
    operations = [
        XTS3Operation(
            op_id=op_id,
            opcode=opcode,
            inputs=inputs,
            outputs=outputs,
            params=data[
                parameter_base
                + params_offset : parameter_base
                + params_offset
                + params_size
            ],
        )
        for op_id, opcode, inputs, outputs, params_offset, params_size in raw_operations
    ]
    if version == 5:
        graph_inputs = [
            t.tensor_id for t in tensors if t.flags & TENSOR_FLAG_GRAPH_INPUT
        ]
        graph_outputs = [
            t.tensor_id for t in tensors if t.flags & TENSOR_FLAG_GRAPH_OUTPUT
        ]
    for tensor_id in graph_inputs + graph_outputs:
        if tensor_id >= len(tensors):
            raise ValueError(f"Graph IO table references missing tensor {tensor_id}")
    return XTS3Graph(
        tensors=tensors,
        operations=operations,
        constant_data=data[constant_base : constant_base + constant_size],
        version=version,
        inputs=graph_inputs,
        outputs=graph_outputs,
    )


__all__ = [
    "MAGIC",
    "XTS3_VERSION",
    "XTS3_VERSIONS",
    "XTS3DType",
    "XTS3Graph",
    "XTS3OpCode",
    "XTS3Operation",
    "XTS3Tensor",
    "build_graph",
    "lower_int8_node",
    "parse_graph",
    "serialize_graph",
]
