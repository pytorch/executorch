# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for ExecuTorch Dimension Order Operations.

This module provides converters for ExecuTorch edge-specific dimension order
operations. These operations handle memory layout conversions (e.g., contiguous
to channels-last) and are inserted during edge transforms.

For TensorRT, these are treated as identity/pass-through operations since
TensorRT handles memory layout internally.

Supported operations:
- dim_order_ops._to_dim_order_copy.default: Memory layout conversion
- dim_order_ops._clone_dim_order.default: Clone with dimension order
- aten.clone.default: Clone operation

Notes:
- These operations don't change tensor values, only memory layout
- TensorRT manages memory layout internally, so we pass through the input
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import get_trt_tensor_from_node

logger: logging.Logger = logging.getLogger(__name__)


def validate_dim_order_copy(node: torch.fx.Node) -> bool:
    """
    Validate that a dim_order_copy node can be converted.

    Args:
        node: FX node representing the operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_dim_order_copy: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_clone(node: torch.fx.Node) -> bool:
    """
    Validate that a clone node can be converted.

    Args:
        node: FX node representing the clone operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_clone: node {node.name} has insufficient args"
        )
        return False

    return True


@converter(
    "dim_order_ops._to_dim_order_copy.default",
    "exir_ops.edge.dim_order_ops._to_dim_order_copy.default",
    validator_fn=validate_dim_order_copy,
)
def convert_to_dim_order_copy(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert dim_order_ops._to_dim_order_copy to TensorRT identity layer.

    This operation converts tensor memory layout (e.g., contiguous to channels-last).
    For TensorRT, we treat this as a pass-through since TensorRT manages layout
    internally.

    Args:
        node: FX node representing the dim_order_copy operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor (same as input).
    """
    logger.debug(f"[TensorRT] Converting dim_order_copy node: {node.name}")

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(
            f"Input to dim_order_copy must be a node, got {type(input_node)}"
        )

    input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)

    # For TensorRT, memory layout is handled internally.
    # We use an identity layer to pass through the tensor.
    layer = network.add_identity(input_trt)
    if layer is None:
        raise RuntimeError(
            f"Failed to create identity layer for dim_order_copy {node.name}"
        )

    layer.name = f"dim_order_copy_{node.name}"
    logger.debug(f"[TensorRT] Created identity layer for dim_order_copy: {layer.name}")

    return layer.get_output(0)


@converter(
    "dim_order_ops._clone_dim_order.default",
    "exir_ops.edge.dim_order_ops._clone_dim_order.default",
    validator_fn=validate_dim_order_copy,
)
def convert_clone_dim_order(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert dim_order_ops._clone_dim_order to TensorRT identity layer.

    This operation clones a tensor with a specific dimension order.
    For TensorRT, we treat this as a pass-through.

    Args:
        node: FX node representing the clone_dim_order operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor (same as input).
    """
    logger.debug(f"[TensorRT] Converting clone_dim_order node: {node.name}")

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(
            f"Input to clone_dim_order must be a node, got {type(input_node)}"
        )

    input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)

    layer = network.add_identity(input_trt)
    if layer is None:
        raise RuntimeError(
            f"Failed to create identity layer for clone_dim_order {node.name}"
        )

    layer.name = f"clone_dim_order_{node.name}"
    logger.debug(f"[TensorRT] Created identity layer for clone_dim_order: {layer.name}")

    return layer.get_output(0)


@converter("aten.clone.default", validator_fn=validate_clone)
def convert_clone(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert aten.clone to TensorRT identity layer.

    Clone creates a copy of a tensor. For TensorRT, we use an identity layer
    since TensorRT manages memory internally.

    Args:
        node: FX node representing the clone operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    logger.debug(f"[TensorRT] Converting clone node: {node.name}")

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to clone must be a node, got {type(input_node)}")

    input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)

    layer = network.add_identity(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create identity layer for clone {node.name}")

    layer.name = f"clone_{node.name}"
    logger.debug(f"[TensorRT] Created identity layer for clone: {layer.name}")

    return layer.get_output(0)


__all__ = [
    "convert_to_dim_order_copy",
    "convert_clone_dim_order",
    "convert_clone",
    "validate_dim_order_copy",
    "validate_clone",
]
