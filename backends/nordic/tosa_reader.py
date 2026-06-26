# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Based on analysis of Vela's tosa_reader.py (ARM, Apache 2.0).
"""TOSA flatbuffer reader for the AXON backend.

Parses a TOSA flatbuffer (as produced by ExecuTorch's TOSABackend._preprocess)
and extracts operators, tensors, weights, and quantization parameters into
a simple graph representation that can be converted to AXON layer descriptors.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)
from typing import Any

from tosa import TosaGraph as TG, Op
from tosa import Attribute as TosaAttribute
from tosa import (
    ClampAttribute,
    ConcatAttribute,
    Conv2dAttribute,
    DepthwiseConv2dAttribute,
    AvgPool2dAttribute,
    MaxPool2dAttribute,
    MulAttribute,
    ReduceSumAttribute,
    RescaleAttribute,
)


# TOSA DType enum → numpy dtype
TOSA_DTYPE_TO_NUMPY = {
    2: np.uint8,   # UINT8
    3: np.int8,    # INT8
    4: np.int8,    # INT8 (alternate)
    5: np.int16,   # INT16
    6: np.int32,   # INT32
    7: np.int64,   # INT48 (stored as int64)
    12: np.float32,  # FP32
}

TOSA_DTYPE_NAMES = {
    2: "uint8", 3: "int8", 4: "int8", 5: "int16", 6: "int32",
    7: "int48", 10: "fp16", 12: "fp32", 14: "bool",
}

# Build reverse map: Op enum value → name
TOSA_OP_NAMES = {}
for _attr in dir(Op.Op):
    _val = getattr(Op.Op, _attr)
    if isinstance(_val, int) and not _attr.startswith("_"):
        TOSA_OP_NAMES[_val] = _attr


@dataclass
class TosaTensor:
    """A tensor in the TOSA graph."""
    index: int
    name: str
    shape: list[int]
    dtype: int  # TOSA DType enum value
    data: np.ndarray | None = None  # Constant data (interpreted per dtype)
    raw_bytes: bytes = b""  # Raw constant bytes (for multi-byte reinterpretation)

    @property
    def dtype_name(self) -> str:
        return TOSA_DTYPE_NAMES.get(self.dtype, f"unknown({self.dtype})")

    @property
    def has_data(self) -> bool:
        return self.data is not None

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    def __repr__(self):
        data_str = f", data={self.data.shape}" if self.has_data else ""
        return f"TosaTensor({self.name}, shape={self.shape}, dtype={self.dtype_name}{data_str})"


@dataclass
class TosaOperator:
    """An operator in the TOSA graph."""
    index: int
    op_type: int  # TOSA Op enum value
    input_tensors: list[TosaTensor]
    output_tensors: list[TosaTensor]
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def op_name(self) -> str:
        return TOSA_OP_NAMES.get(self.op_type, f"Unknown({self.op_type})")

    def __repr__(self):
        ins = [t.name.split("/")[-1][:25] for t in self.input_tensors]
        outs = [t.name.split("/")[-1][:25] for t in self.output_tensors]
        return f"TosaOperator({self.op_name}, in={ins}, out={outs})"


@dataclass
class TosaGraph:
    """Parsed TOSA graph."""
    tensors: list[TosaTensor]
    operators: list[TosaOperator]
    input_tensor_indices: list[int]   # Indices into tensors[] for graph inputs
    output_tensor_indices: list[int]  # Indices into tensors[] for graph outputs

    def get_non_const_operators(self) -> list[TosaOperator]:
        """Return operators that aren't CONST or CONST_SHAPE."""
        return [
            op for op in self.operators
            if op.op_name not in ("CONST", "CONST_SHAPE")
        ]

    def print_summary(self):
        """Log a human-readable summary of the graph."""
        logger.info("TOSA Graph: %d tensors, %d operators",
                    len(self.tensors), len(self.operators))
        logger.info("  Inputs: %s",
                    [self.tensors[i].name for i in self.input_tensor_indices])
        logger.info("  Outputs: %s",
                    [self.tensors[i].name for i in self.output_tensor_indices])
        for op in self.operators:
            if op.op_name in ("CONST", "CONST_SHAPE"):
                continue
            logger.info("  %s:", op.op_name)
            for t in op.input_tensors:
                prefix = "  [const]" if t.has_data else "        "
                logger.info("    in:  %s %-40s %-15s %s",
                           prefix, t.name.split("/")[-1][:40],
                           str(t.shape), t.dtype_name)
            for t in op.output_tensors:
                logger.info("    out:          %-40s %-15s %s",
                           t.name.split("/")[-1][:40],
                           str(t.shape), t.dtype_name)


def _parse_conv2d_attrs(fb_op) -> dict[str, Any]:
    """Extract pad, stride, dilation from TOSA Conv2dAttribute."""
    attr = Conv2dAttribute.Conv2dAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    pad = [attr.Pad(i) for i in range(attr.PadLength())] if attr.PadLength() else []
    stride = [attr.Stride(i) for i in range(attr.StrideLength())] if attr.StrideLength() else [1, 1]
    dilation = [attr.Dilation(i) for i in range(attr.DilationLength())] if attr.DilationLength() else [1, 1]
    return {"pad": pad, "stride": stride, "dilation": dilation}


def _parse_depthwise_conv2d_attrs(fb_op) -> dict[str, Any]:
    """Extract pad, stride, dilation from TOSA DepthwiseConv2dAttribute."""
    attr = DepthwiseConv2dAttribute.DepthwiseConv2dAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    pad = [attr.Pad(i) for i in range(attr.PadLength())] if attr.PadLength() else []
    stride = [attr.Stride(i) for i in range(attr.StrideLength())] if attr.StrideLength() else [1, 1]
    dilation = [attr.Dilation(i) for i in range(attr.DilationLength())] if attr.DilationLength() else [1, 1]
    return {"pad": pad, "stride": stride, "dilation": dilation}


def _parse_avg_pool2d_attrs(fb_op) -> dict[str, Any]:
    """Extract kernel, pad, stride from TOSA AvgPool2dAttribute."""
    attr = AvgPool2dAttribute.AvgPool2dAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    kernel = [attr.Kernel(i) for i in range(attr.KernelLength())] if attr.KernelLength() else []
    pad = [attr.Pad(i) for i in range(attr.PadLength())] if attr.PadLength() else []
    stride = [attr.Stride(i) for i in range(attr.StrideLength())] if attr.StrideLength() else [1, 1]
    return {"kernel": kernel, "pad": pad, "stride": stride}


def _parse_max_pool2d_attrs(fb_op) -> dict[str, Any]:
    """Extract kernel, pad, stride from TOSA MaxPool2dAttribute."""
    attr = MaxPool2dAttribute.MaxPool2dAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    kernel = [attr.Kernel(i) for i in range(attr.KernelLength())] if attr.KernelLength() else []
    pad = [attr.Pad(i) for i in range(attr.PadLength())] if attr.PadLength() else []
    stride = [attr.Stride(i) for i in range(attr.StrideLength())] if attr.StrideLength() else [1, 1]
    return {"kernel": kernel, "pad": pad, "stride": stride}


def _parse_reduce_sum_attrs(fb_op) -> dict[str, Any]:
    """Extract axis from TOSA ReduceSumAttribute."""
    attr = ReduceSumAttribute.ReduceSumAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    return {"axis": attr.Axis()}


def _parse_concat_attrs(fb_op) -> dict[str, Any]:
    """Extract axis from TOSA ConcatAttribute."""
    attr = ConcatAttribute.ConcatAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    return {"axis": attr.Axis()}


def _parse_clamp_attrs(fb_op) -> dict[str, Any]:
    """Extract min/max values from TOSA ClampAttribute.

    MinVal/MaxVal are stored as raw bytes; for INT8 quantized models
    they represent the integer clamp bounds.
    """
    attr = ClampAttribute.ClampAttribute()
    attr.Init(fb_op.Attribute().Bytes, fb_op.Attribute().Pos)
    min_val = list(attr.MinValAsNumpy()) if not attr.MinValIsNone() and attr.MinValLength() > 0 else []
    max_val = list(attr.MaxValAsNumpy()) if not attr.MaxValIsNone() and attr.MaxValLength() > 0 else []
    # Interpret as int8 for quantized models
    min_int = int(np.frombuffer(bytes(min_val), dtype=np.int8)[0]) if len(min_val) >= 1 else -128
    max_int = int(np.frombuffer(bytes(max_val), dtype=np.int8)[0]) if len(max_val) >= 1 else 127
    return {"min_int": min_int, "max_int": max_int}


# Map TOSA attribute type enum → parser function
_ATTR_PARSERS: dict[int, Any] = {
    TosaAttribute.Attribute.Conv2dAttribute: _parse_conv2d_attrs,
    TosaAttribute.Attribute.DepthwiseConv2dAttribute: _parse_depthwise_conv2d_attrs,
    TosaAttribute.Attribute.AvgPool2dAttribute: _parse_avg_pool2d_attrs,
    TosaAttribute.Attribute.MaxPool2dAttribute: _parse_max_pool2d_attrs,
    TosaAttribute.Attribute.ReduceSumAttribute: _parse_reduce_sum_attrs,
    TosaAttribute.Attribute.ClampAttribute: _parse_clamp_attrs,
    TosaAttribute.Attribute.ConcatAttribute: _parse_concat_attrs,
}


def _parse_operator_attributes(fb_op) -> dict[str, Any]:
    """Parse operator-specific attributes from the TOSA flatbuffer."""
    attr_type = fb_op.AttributeType()
    if attr_type == TosaAttribute.Attribute.NONE or fb_op.Attribute() is None:
        return {}
    parser = _ATTR_PARSERS.get(attr_type)
    if parser is not None:
        return parser(fb_op)
    return {}


def parse_tosa_flatbuffer(tosa_bytes: bytes) -> TosaGraph:
    """Parse a TOSA flatbuffer into a TosaGraph.

    Args:
        tosa_bytes: Raw TOSA flatbuffer bytes (from TOSABackend._preprocess).

    Returns:
        TosaGraph with tensors and operators.
    """
    graph = TG.TosaGraph.GetRootAs(tosa_bytes, 0)

    if graph.RegionsLength() == 0:
        raise ValueError("TOSA graph has no regions")

    region = graph.Regions(0)
    if region.BlocksLength() == 0:
        raise ValueError("TOSA region has no blocks")

    block = region.Blocks(0)

    # Parse tensors
    tensors = []
    for t in range(block.TensorsLength()):
        fb_tensor = block.Tensors(t)
        name = fb_tensor.Name().decode() if fb_tensor.Name() else f"tensor_{t}"
        shape = [fb_tensor.Shape(i) for i in range(fb_tensor.ShapeLength())]
        dtype = fb_tensor.Type()

        # Extract constant data
        data = None
        raw_bytes = b""
        data_len = fb_tensor.DataLength()
        if data_len > 0:
            raw = fb_tensor.DataAsNumpy()
            raw_bytes = bytes(raw)
            np_dtype = TOSA_DTYPE_TO_NUMPY.get(dtype)
            if np_dtype is not None:
                values = np.frombuffer(raw_bytes, dtype=np_dtype)
                numel = 1
                for s in shape:
                    numel *= s
                if len(values) >= numel:
                    data = values[:numel].reshape(shape)
                else:
                    data = values  # Can't reshape, store flat

        tensors.append(TosaTensor(
            index=t,
            name=name,
            shape=shape,
            dtype=dtype,
            data=data,
            raw_bytes=raw_bytes,
        ))

    # Build tensor lookup by name (for resolving operator inputs/outputs)
    tensor_by_name = {t.name: t for t in tensors}

    # Parse operators
    operators = []
    for o in range(block.OperatorsLength()):
        fb_op = block.Operators(o)
        op_type = fb_op.Op()

        # Resolve input/output tensors
        # TOSA flatbuffer stores tensor references as names (bytes)
        input_tensors = []
        for i in range(fb_op.InputsLength()):
            tensor_name = fb_op.Inputs(i)
            if isinstance(tensor_name, bytes):
                tensor_name = tensor_name.decode()
            if tensor_name in tensor_by_name:
                input_tensors.append(tensor_by_name[tensor_name])

        output_tensors = []
        for i in range(fb_op.OutputsLength()):
            tensor_name = fb_op.Outputs(i)
            if isinstance(tensor_name, bytes):
                tensor_name = tensor_name.decode()
            if tensor_name in tensor_by_name:
                output_tensors.append(tensor_by_name[tensor_name])

        # Deserialize operator attributes (padding, stride, dilation, kernel, etc.)
        attributes = _parse_operator_attributes(fb_op)

        operators.append(TosaOperator(
            index=o,
            op_type=op_type,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attributes=attributes,
        ))

    # Identify graph inputs (tensors with no data and no producing CONST op)
    const_output_names = set()
    for op in operators:
        if TOSA_OP_NAMES.get(op.op_type) in ("CONST", "CONST_SHAPE"):
            for t in op.output_tensors:
                const_output_names.add(t.name)

    input_indices = []
    output_indices = []
    for t in tensors:
        if not t.has_data and t.name not in const_output_names:
            # Check if this tensor is an input to any operator but not an output
            is_op_output = any(
                t in op.output_tensors
                for op in operators
                if TOSA_OP_NAMES.get(op.op_type) not in ("CONST", "CONST_SHAPE")
            )
            if not is_op_output:
                input_indices.append(t.index)

    # Graph outputs are the outputs of the last non-CONST operator
    if operators:
        last_op = operators[-1]
        output_indices = [t.index for t in last_op.output_tensors]

    return TosaGraph(
        tensors=tensors,
        operators=operators,
        input_tensor_indices=input_indices,
        output_tensor_indices=output_indices,
    )
