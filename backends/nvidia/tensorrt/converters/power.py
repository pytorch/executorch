# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Power Operations.

Supported operations:
- aten.pow.Tensor_Scalar: Tensor raised to scalar power
- aten.pow.Tensor_Tensor: Tensor raised to tensor power

These operations are commonly used in transformer models for layer normalization
decomposition and other mathematical computations.

Design patterns follow TensorRT best practices including:
- Type promotion to ensure consistent dtypes for POW operation
- Proper type casting for integer inputs (POW supports float32/int8 only)
- Proper broadcasting using broadcast_tensors utility
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    broadcast_tensors,
    create_constant,
    get_node_shape,
)

logger: logging.Logger = logging.getLogger(__name__)


def validate_pow(node: torch.fx.Node) -> bool:
    """Validate that a pow node can be converted to TensorRT."""
    if node.op != "call_function":
        return False
    if len(node.args) < 2:
        return False
    if not isinstance(node.args[0], torch.fx.Node):
        return False
    return True


def _ensure_float_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    node_name: str,
    suffix: str,
) -> trt.ITensor:
    """Ensure tensor is float type for POW operation.

    TensorRT POW operation supports only float32, float16, and int8.
    This follows TensorRT pattern of promoting types.

    Args:
        network: TensorRT network definition.
        tensor: Input tensor to potentially cast.
        node_name: Node name for layer naming.
        suffix: Suffix for layer naming.

    Returns:
        Tensor with float dtype.
    """
    if tensor.dtype in (trt.int32, trt.int64):
        cast_layer = network.add_cast(tensor, trt.float32)
        if cast_layer is None:
            raise RuntimeError(f"Failed to create cast layer for {node_name}")
        cast_layer.name = f"pow_cast_{suffix}_{node_name}"
        return cast_layer.get_output(0)
    return tensor


def _get_dtype_for_scalar(base_tensor: trt.ITensor) -> np.dtype:
    """Get appropriate numpy dtype for scalar exponent based on base tensor type.

    Following TensorRT pattern where scalar inherits dtype from tensor.
    """
    if base_tensor.dtype == trt.float16:
        return np.float16
    elif base_tensor.dtype == trt.float32:
        return np.float32
    else:
        return np.float32


@converter("aten.pow.Tensor_Scalar", validator_fn=validate_pow)
def convert_pow_tensor_scalar(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch pow.Tensor_Scalar to TensorRT.

    PyTorch signature: aten.pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
    Computes self ** exponent element-wise.

    Following TensorRT patterns:
    - Type promotion for consistent dtypes
    - Proper scalar to tensor conversion
    - Broadcasting to match input dimensions
    """
    logger.debug(f"[TensorRT] Converting pow.Tensor_Scalar node: {node.name}")

    input_node = node.args[0]
    exponent = node.args[1]

    if input_node not in input_map:
        raise ValueError(f"Input node '{input_node.name}' not found in input_map")

    input_trt = input_map[input_node]
    input_trt = _ensure_float_dtype(network, input_trt, node.name, "input")

    exponent_value = float(exponent)
    exponent_dtype = _get_dtype_for_scalar(input_trt)
    exponent_np = np.array([exponent_value], dtype=exponent_dtype)
    exponent_trt = create_constant(network, exponent_np, f"pow_exp_{node.name}")

    # Get target ndim for broadcasting from input node metadata or tensor
    input_shape = get_node_shape(input_node)
    if input_shape is not None:
        target_ndim = len(input_shape)
    else:
        target_ndim = len(input_trt.shape)
    target_ndim = max(target_ndim, 1)

    # Broadcast exponent to match input dimensions for TensorRT elementwise op
    [exponent_trt] = broadcast_tensors(
        network, [exponent_trt], target_ndim, f"pow_exp_{node.name}"
    )

    layer = network.add_elementwise(
        input_trt, exponent_trt, trt.ElementWiseOperation.POW
    )
    if layer is None:
        raise RuntimeError(f"Failed to create pow layer for {node.name}")
    layer.name = f"pow_{node.name}"

    return layer.get_output(0)


@converter("aten.pow.Tensor_Tensor", validator_fn=validate_pow)
def convert_pow_tensor_tensor(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """Convert PyTorch pow.Tensor_Tensor to TensorRT.

    PyTorch signature: aten.pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
    Computes self ** exponent element-wise.

    Following TensorRT patterns:
    - Type promotion for both operands
    - Proper handling of constant exponents
    - Broadcasting to match dimensions
    """
    logger.debug(f"[TensorRT] Converting pow.Tensor_Tensor node: {node.name}")

    input_node = node.args[0]
    exponent_node = node.args[1]

    if input_node not in input_map:
        raise ValueError(f"Input node '{input_node.name}' not found in input_map")

    input_trt = input_map[input_node]
    input_trt = _ensure_float_dtype(network, input_trt, node.name, "input")

    if isinstance(exponent_node, torch.fx.Node):
        if exponent_node not in input_map:
            raise ValueError(
                f"Exponent node '{exponent_node.name}' not found in input_map"
            )
        exponent_trt = input_map[exponent_node]
        exponent_trt = _ensure_float_dtype(network, exponent_trt, node.name, "exp")
    else:
        exponent_value = float(exponent_node)
        exponent_dtype = _get_dtype_for_scalar(input_trt)
        exponent_np = np.array([exponent_value], dtype=exponent_dtype)
        exponent_trt = create_constant(network, exponent_np, f"pow_exp_{node.name}")

    # Determine target ndim for broadcasting
    input_shape = get_node_shape(input_node)
    if input_shape is not None:
        input_ndim = len(input_shape)
    else:
        input_ndim = len(input_trt.shape)

    exp_ndim = len(exponent_trt.shape)
    target_ndim = max(input_ndim, exp_ndim, 1)

    # Broadcast both tensors to target dimensions
    [input_trt, exponent_trt] = broadcast_tensors(
        network, [input_trt, exponent_trt], target_ndim, f"pow_{node.name}"
    )

    layer = network.add_elementwise(
        input_trt, exponent_trt, trt.ElementWiseOperation.POW
    )
    if layer is None:
        raise RuntimeError(f"Failed to create pow layer for {node.name}")
    layer.name = f"pow_{node.name}"

    return layer.get_output(0)


__all__ = [
    "convert_pow_tensor_scalar",
    "convert_pow_tensor_tensor",
    "validate_pow",
]
