# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Reduction Operations.

This module provides converters for PyTorch reduction operations to TensorRT
reduction layers.

Supported operations:
- aten.mean.dim: Reduce mean along specified dimensions
- aten.sum.dim_IntList: Reduce sum along specified dimensions

Notes:
- TensorRT uses axes as a bitmask for specifying dimensions
- keepdim parameter controls whether reduced dimensions are kept
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import get_node_shape

logger: logging.Logger = logging.getLogger(__name__)


def validate_mean(node: torch.fx.Node) -> bool:
    """
    Validate that a mean node can be converted to TensorRT.

    Args:
        node: FX node representing the mean operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_mean: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Minimum args: input, dim
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_mean: node {node.name} has insufficient args"
        )
        return False

    return True


def _get_reduce_axes(
    dims: Union[int, List[int]],
    ndim: int,
) -> int:
    """
    Convert dimension indices to TensorRT axes bitmask.

    Args:
        dims: Dimension(s) to reduce.
        ndim: Total number of dimensions.

    Returns:
        TensorRT axes bitmask.
    """
    if isinstance(dims, int):
        dims = [dims]

    axes = 0
    for dim in dims:
        # Handle negative dimensions
        if dim < 0:
            dim = ndim + dim
        axes |= 1 << dim

    return axes


@converter(
    "aten.mean.dim",
    "aten.mean.default",
    validator_fn=validate_mean,
)
def convert_mean(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch mean reduction to TensorRT reduce layer.

    Args:
        node: FX node representing the mean operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: ExportedProgram for extracting parameters.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_mean") from e

    logger.debug(f"[TensorRT] Converting mean node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to mean must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get dimensions - can be a single int or a list
    dims = args[1] if len(args) > 1 else kwargs.get("dim", None)

    # Get keepdim parameter
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

    # Get input dimensions from node metadata for reliability
    input_shape = get_node_shape(input_node) or tuple(input_trt.shape)
    ndim = len(input_shape)

    if dims is None:
        # Reduce over all dimensions
        axes = (1 << ndim) - 1  # All bits set
    else:
        axes = _get_reduce_axes(dims, ndim)

    logger.debug(
        f"[TensorRT] mean reduction: dims={dims}, keepdim={keepdim}, axes={axes:b}"
    )

    # Create reduce layer with AVERAGE operation
    layer = network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, keepdim)

    if layer is None:
        raise RuntimeError(f"Failed to create reduce layer for mean {node.name}")

    layer.name = f"mean_{node.name}"
    logger.debug(f"[TensorRT] Created mean reduce layer: {layer.name}")

    return layer.get_output(0)


def validate_sum(node: torch.fx.Node) -> bool:
    """
    Validate that a sum node can be converted to TensorRT.

    Args:
        node: FX node representing the sum operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_sum: node {node.name} is not call_function"
        )
        return False

    return True


@converter(
    "aten.sum.dim_IntList",
    "aten.sum.default",
    validator_fn=validate_sum,
)
def convert_sum(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch sum reduction to TensorRT reduce layer.

    Args:
        node: FX node representing the sum operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: ExportedProgram for extracting parameters.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_sum") from e

    logger.debug(f"[TensorRT] Converting sum node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to sum must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get dimensions - can be a single int or a list
    dims = args[1] if len(args) > 1 else kwargs.get("dim", None)

    # Get keepdim parameter
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

    # Get input dimensions from node metadata for reliability
    input_shape = get_node_shape(input_node) or tuple(input_trt.shape)
    ndim = len(input_shape)

    if dims is None:
        # Reduce over all dimensions
        axes = (1 << ndim) - 1  # All bits set
    else:
        axes = _get_reduce_axes(dims, ndim)

    logger.debug(
        f"[TensorRT] sum reduction: dims={dims}, keepdim={keepdim}, axes={axes:b}"
    )

    # Create reduce layer with SUM operation
    layer = network.add_reduce(input_trt, trt.ReduceOperation.SUM, axes, keepdim)

    if layer is None:
        raise RuntimeError(f"Failed to create reduce layer for sum {node.name}")

    layer.name = f"sum_{node.name}"
    logger.debug(f"[TensorRT] Created sum reduce layer: {layer.name}")

    return layer.get_output(0)


__all__ = [
    "convert_mean",
    "convert_sum",
    "validate_mean",
    "validate_sum",
]
