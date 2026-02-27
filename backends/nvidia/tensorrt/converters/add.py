# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for element-wise addition operations."""

import logging
from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    broadcast_tensors,
    get_node_dtype,
    get_trt_tensor,
    promote_and_cast_tensors,
    set_layer_name,
)


logger: logging.Logger = logging.getLogger(__name__)


def _get_input_ndim(arg: Any, input_map: Dict[torch.fx.Node, Any]) -> int:
    """Get the number of dimensions for an elementwise input argument.

    Uses node metadata when available for reliability during network building.

    Args:
        arg: Input argument (either torch.fx.Node or scalar).
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        Number of dimensions (0 for scalars).
    """
    if isinstance(arg, torch.fx.Node):
        # Try to get ndim from node metadata first (most reliable)
        if "val" in arg.meta and hasattr(arg.meta["val"], "shape"):
            return len(arg.meta["val"].shape)
        # Fall back to TRT tensor shape - handle dynamic shapes carefully
        if arg in input_map:
            trt_tensor = input_map[arg]
            try:
                shape = trt_tensor.shape
                if shape is not None:
                    ndim = len(shape)
                    if ndim >= 0:  # Valid shape
                        return ndim
            except (ValueError, TypeError):
                pass  # Dynamic shape, fall through
    # For scalars or unknown, return 0 (will be broadcast)
    return 0


def _get_elementwise_input(
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    arg: Any,
    name: str,
    dtype: Optional[torch.dtype],
) -> trt.ITensor:
    """Get TensorRT tensor for an elementwise operation input.

    Handles:
    - FX nodes already in input_map
    - FX nodes that are lifted buffers/parameters (placeholder nodes with b_ or p_ prefix)
    - Scalar values

    Args:
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        arg: Input argument (either torch.fx.Node or scalar value).
        name: Name for the constant tensor if created.
        dtype: Data type for scalar conversion.

    Returns:
        TensorRT tensor for the input.

    Raises:
        ValueError: If arg is a Node but not found in input_map and cannot be created as constant.
    """
    if isinstance(arg, torch.fx.Node):
        if arg in input_map:
            return input_map[arg]
        
        # Handle lifted buffers and parameters that aren't in input_map
        # These are placeholder nodes with names starting with b_ (buffers) or p_ (parameters)
        # or get_attr nodes. We need to create constants from their metadata values.
        if arg.op == "placeholder" or arg.op == "get_attr":
            if "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                logger.debug(f"[TensorRT] Creating constant for lifted buffer/parameter: {arg.name}")
                trt_tensor = get_trt_tensor(network, arg.meta["val"], f"const_{arg.name}", dtype)
                input_map[arg] = trt_tensor  # Cache for future use
                return trt_tensor
        
        raise ValueError(
            f"Input node '{arg.name}' not found in input_map and could not be created as constant. "
            f"Node op: {arg.op}, target: {arg.target}. "
            f"Available nodes: {list(n.name for n in input_map.keys())}"
        )
    return get_trt_tensor(network, arg, name, dtype)


@converter("aten.add.Tensor", "aten.add_.Tensor")
def convert_add(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.add.Tensor and aten.add_.Tensor to TensorRT ElementWise SUM.

    Handles tensor + tensor, tensor + scalar, and scalar + tensor cases.
    The alpha parameter (x + alpha * y) is validated to be 1.
    Note: In-place variant (add_) is handled identically since TensorRT doesn't
    have in-place operations.

    Args:
        node: FX node representing the add operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        ctx: Optional conversion context.

    Returns:
        TensorRT tensor representing the sum.

    Raises:
        ValueError: If alpha != 1 or if required inputs are missing.
    """
    # Validate args
    if len(node.args) < 2:
        raise ValueError(
            f"aten.add requires at least 2 arguments, got {len(node.args)}"
        )

    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    alpha = node.args[2] if len(node.args) > 2 else node.kwargs.get("alpha", 1)
    if alpha != 1:
        raise ValueError(
            f"aten.add.Tensor with alpha != 1 is not supported, got alpha={alpha}"
        )

    dtype = get_node_dtype(node)

    lhs = _get_elementwise_input(network, input_map, lhs_arg, "lhs", dtype)
    rhs = _get_elementwise_input(network, input_map, rhs_arg, "rhs", dtype)

    # Type promotion: ensure both operands have compatible types
    lhs, rhs = promote_and_cast_tensors(network, lhs, rhs, f"add_{node.name}")

    # Get target ndim from node metadata for reliability
    lhs_ndim = _get_input_ndim(lhs_arg, input_map)
    rhs_ndim = _get_input_ndim(rhs_arg, input_map)
    target_ndim = max(lhs_ndim, rhs_ndim)

    # Fall back to output shape from node metadata if we couldn't get input shapes
    if target_ndim == 0 and "val" in node.meta and hasattr(node.meta["val"], "shape"):
        target_ndim = len(node.meta["val"].shape)

    # If still 0, both inputs are scalars - result is scalar (0-dim tensor in TRT is 1-dim)
    if target_ndim == 0:
        target_ndim = 1

    lhs, rhs = broadcast_tensors(network, [lhs, rhs], target_ndim, f"add_{node.name}")

    layer = network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError(f"Failed to create elementwise SUM layer for {node.name}")
    set_layer_name(layer, node, "add")

    return layer.get_output(0)
