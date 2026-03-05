# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for element-wise division operations."""

from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    broadcast_tensors,
    cast_trt_tensor,
    get_node_dtype,
    get_trt_tensor,
    promote_and_cast_tensors,
    set_layer_name,
)


def _get_elementwise_input(
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    arg: Any,
    name: str,
    dtype: Optional[torch.dtype],
) -> trt.ITensor:
    """Get TensorRT tensor for an elementwise operation input."""
    if isinstance(arg, torch.fx.Node):
        if arg not in input_map:
            raise ValueError(
                f"Input node '{arg.name}' not found in input_map. "
                f"Available nodes: {list(input_map.keys())}"
            )
        return input_map[arg]
    return get_trt_tensor(network, arg, name, dtype)


def _get_input_ndim(arg: Any, input_map: Dict[torch.fx.Node, Any]) -> int:
    """Get the number of dimensions for an elementwise input argument."""
    if isinstance(arg, torch.fx.Node):
        if "val" in arg.meta and hasattr(arg.meta["val"], "shape"):
            return len(arg.meta["val"].shape)
        if arg in input_map:
            trt_tensor = input_map[arg]
            shape = trt_tensor.shape
            if shape is not None:
                return len(shape)
    return 0


@converter("aten.div.Tensor", "aten.div_.Tensor", "aten.div.Tensor_mode")
def convert_div(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.div.Tensor and aten.div.Tensor_mode to TensorRT ElementWise DIV.

    Handles tensor / tensor, tensor / scalar, and scalar / tensor cases.
    Includes type promotion for mixed-type operands.

    For div.Tensor_mode, supports rounding_mode kwargs:
    - None: standard float division
    - "floor": floor division (same as floor_divide)
    - "trunc": truncated division (round towards zero)

    Note: For integer division, TensorRT requires float types, so we cast
    integer operands to float32 before division.
    """
    if len(node.args) < 2:
        raise ValueError(
            f"aten.div requires at least 2 arguments, got {len(node.args)}"
        )

    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    dtype = get_node_dtype(node)

    lhs = _get_elementwise_input(network, input_map, lhs_arg, f"div_lhs_{node.name}", dtype)
    rhs = _get_elementwise_input(network, input_map, rhs_arg, f"div_rhs_{node.name}", dtype)

    # Type promotion: ensure both operands have compatible types
    lhs, rhs = promote_and_cast_tensors(network, lhs, rhs, f"div_{node.name}")

    # For integer division, TensorRT requires float types
    # Cast to float32 if both operands are integer types
    if lhs.dtype in (trt.int8, trt.int32) and rhs.dtype in (trt.int8, trt.int32):
        lhs = cast_trt_tensor(network, lhs, trt.float32, f"div_lhs_float_{node.name}")
        rhs = cast_trt_tensor(network, rhs, trt.float32, f"div_rhs_float_{node.name}")

    # Get target ndim for broadcasting
    lhs_ndim = _get_input_ndim(lhs_arg, input_map)
    rhs_ndim = _get_input_ndim(rhs_arg, input_map)
    target_ndim = max(lhs_ndim, rhs_ndim)

    if target_ndim == 0 and "val" in node.meta and hasattr(node.meta["val"], "shape"):
        target_ndim = len(node.meta["val"].shape)
    if target_ndim == 0:
        target_ndim = 1

    lhs, rhs = broadcast_tensors(network, [lhs, rhs], target_ndim, f"div_{node.name}")

    layer = network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.DIV)
    if layer is None:
        raise RuntimeError(f"Failed to create elementwise DIV layer for {node.name}")
    set_layer_name(layer, node, "div")
    result = layer.get_output(0)

    # Handle rounding modes for div.Tensor_mode
    rounding_mode = node.kwargs.get("rounding_mode", None)
    if rounding_mode == "trunc":
        # Truncated division: div then round towards zero
        # trunc(x) = sign(x) * floor(abs(x))
        # Ensure float type for unary ops (FLOOR/ABS/SIGN require float)
        original_dtype = result.dtype
        if result.dtype in (trt.int8, trt.int32, trt.int64):
            result = cast_trt_tensor(network, result, trt.float32, f"div_trunc_tofloat_{node.name}")
        abs_layer = network.add_unary(result, trt.UnaryOperation.ABS)
        abs_layer.name = f"div_trunc_abs_{node.name}"
        floor_layer = network.add_unary(abs_layer.get_output(0), trt.UnaryOperation.FLOOR)
        floor_layer.name = f"div_trunc_floor_{node.name}"
        sign_layer = network.add_unary(result, trt.UnaryOperation.SIGN)
        sign_layer.name = f"div_trunc_sign_{node.name}"
        mul_layer = network.add_elementwise(
            floor_layer.get_output(0), sign_layer.get_output(0),
            trt.ElementWiseOperation.PROD
        )
        mul_layer.name = f"div_trunc_mul_{node.name}"
        result = mul_layer.get_output(0)
        # Cast back to the original integer type to match PyTorch semantics.
        # Without this, integer trunc_div returns float32, causing output
        # mismatches for models that use integer division (e.g., emformer).
        if original_dtype in (trt.int8, trt.int32, trt.int64):
            result = cast_trt_tensor(network, result, original_dtype, f"div_trunc_toint_{node.name}")
    elif rounding_mode == "floor":
        original_dtype = result.dtype
        if result.dtype in (trt.int8, trt.int32, trt.int64):
            result = cast_trt_tensor(network, result, trt.float32, f"div_floor_tofloat_{node.name}")
        floor_layer = network.add_unary(result, trt.UnaryOperation.FLOOR)
        floor_layer.name = f"div_floor_{node.name}"
        result = floor_layer.get_output(0)
        # Cast back to the original integer type to match PyTorch semantics.
        if original_dtype in (trt.int8, trt.int32, trt.int64):
            result = cast_trt_tensor(network, result, original_dtype, f"div_floor_toint_{node.name}")

    return result


@converter("aten.floor_divide.default")
def convert_floor_divide(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.floor_divide to TensorRT ElementWise FLOOR_DIV.

    Handles tensor // tensor and tensor // scalar cases.
    """
    if len(node.args) < 2:
        raise ValueError(
            f"aten.floor_divide requires at least 2 arguments, got {len(node.args)}"
        )

    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    dtype = get_node_dtype(node)

    lhs = _get_elementwise_input(network, input_map, lhs_arg, f"floordiv_lhs_{node.name}", dtype)
    rhs = _get_elementwise_input(network, input_map, rhs_arg, f"floordiv_rhs_{node.name}", dtype)

    # Type promotion
    lhs, rhs = promote_and_cast_tensors(network, lhs, rhs, f"floordiv_{node.name}")

    # Get target ndim for broadcasting
    lhs_ndim = _get_input_ndim(lhs_arg, input_map)
    rhs_ndim = _get_input_ndim(rhs_arg, input_map)
    target_ndim = max(lhs_ndim, rhs_ndim)

    if target_ndim == 0 and "val" in node.meta and hasattr(node.meta["val"], "shape"):
        target_ndim = len(node.meta["val"].shape)
    if target_ndim == 0:
        target_ndim = 1

    lhs, rhs = broadcast_tensors(network, [lhs, rhs], target_ndim, f"floordiv_{node.name}")

    layer = network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.FLOOR_DIV)
    if layer is None:
        raise RuntimeError(f"Failed to create elementwise FLOOR_DIV layer for {node.name}")
    set_layer_name(layer, node, "floor_divide")

    return layer.get_output(0)
