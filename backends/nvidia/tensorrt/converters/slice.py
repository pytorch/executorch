# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Slice Operations.

This module provides converters for PyTorch tensor slicing operations to TensorRT
slice layers.

Supported operations:
- aten.slice.Tensor: Slice along a dimension with start, end, step
- aten.slice_copy.Tensor: Copy variant of slice
- aten.index.Tensor: Index selection with tensor indices

Notes:
- Slice uses network.add_slice()
- Start, shape, stride computed from slice parameters
- Handles negative indices and None values
"""

import logging
import sys
from typing import Any, Dict, List, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    get_node_shape,
)

logger: logging.Logger = logging.getLogger(__name__)


def _get_positive_dim(dim: int, ndim: int) -> int:
    """Convert a potentially negative dimension index to positive."""
    if dim < 0:
        dim = ndim + dim
    return dim


@converter("aten.slice.Tensor", "aten.slice_copy.Tensor")
def convert_slice(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch slice to TensorRT slice layer.

    slice.Tensor(Tensor self, int dim=0, int? start=None, int? end=None, int step=1)

    Args:
        node: FX node representing the slice operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_slice") from e

    logger.debug(f"[TensorRT] Converting slice node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to slice must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get shape from node metadata for reliability (TRT shapes can be invalid during network building)
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Parse arguments with defaults
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    start = args[2] if len(args) > 2 else kwargs.get("start", None)
    end = args[3] if len(args) > 3 else kwargs.get("end", None)
    step = args[4] if len(args) > 4 else kwargs.get("step", 1)

    # Handle None values
    if start is None:
        start = 0
    if end is None or end == sys.maxsize:
        end = input_shape[dim]

    # Handle negative dimension
    dim = _get_positive_dim(dim, ndim)

    # Handle negative indices
    dim_size = input_shape[dim]
    if isinstance(start, int) and start < 0:
        start = max(0, dim_size + start)
    if isinstance(end, int) and end < 0:
        end = max(0, dim_size + end)

    # Clamp to valid range
    if isinstance(start, int):
        start = max(0, min(start, dim_size))
    if isinstance(end, int):
        end = max(0, min(end, dim_size))

    # Build start, shape, stride for slice
    start_slice = [0] * ndim
    start_slice[dim] = start

    # Compute output shape
    output_shape = input_shape.copy()
    if isinstance(end, int) and isinstance(start, int) and isinstance(step, int):
        slice_len = max(0, (end - start + step - 1) // step) if step > 0 else 0
        output_shape[dim] = slice_len
    else:
        output_shape[dim] = 0  # Dynamic

    stride_slice = [1] * ndim
    stride_slice[dim] = step

    layer = network.add_slice(
        input_trt,
        start=trt.Dims(start_slice),
        shape=trt.Dims(output_shape),
        stride=trt.Dims(stride_slice),
    )

    if layer is None:
        raise RuntimeError(f"Failed to create slice layer for {node.name}")

    layer.name = f"slice_{node.name}"

    logger.debug(
        f"[TensorRT] Created slice layer: {layer.name}, "
        f"dim={dim}, start={start}, end={end}, step={step}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.index.Tensor")
def convert_index(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch index to TensorRT gather layer.

    index.Tensor(Tensor self, Tensor?[] indices)

    This operation indexes into a tensor using tensor indices.
    For simple 1D indexing along first dimension, we use gather.

    Args:
        node: FX node representing the index operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_index") from e

    logger.debug(f"[TensorRT] Converting index node: {node.name}")

    args = node.args

    input_node = args[0]
    indices = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to index must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # indices is a list of optional tensors, one per dimension
    # Find the first non-None index
    gather_dim = None
    index_tensor = None

    for i, idx in enumerate(indices):
        if idx is not None and isinstance(idx, torch.fx.Node):
            if gather_dim is not None:
                raise NotImplementedError(
                    "Multiple index tensors not supported, only single dimension indexing"
                )
            gather_dim = i
            index_tensor = input_map.get(idx)

    if gather_dim is None or index_tensor is None:
        raise ValueError("No valid index tensor found in indices")

    # Create gather layer
    layer = network.add_gather(input_trt, index_tensor, axis=gather_dim)

    if layer is None:
        raise RuntimeError(f"Failed to create gather layer for index {node.name}")

    layer.name = f"index_gather_{node.name}"

    logger.debug(
        f"[TensorRT] Created index/gather layer: {layer.name}, gather_dim={gather_dim}"
    )

    return layer.get_output(0)


@converter("aten.contiguous.default")
def convert_contiguous(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch contiguous to TensorRT.

    contiguous is a no-op in TensorRT as all tensors are contiguous.

    Args:
        node: FX node representing the contiguous operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT input tensor (passthrough).
    """
    logger.debug(f"[TensorRT] Converting contiguous node (no-op): {node.name}")

    args = node.args
    input_node = args[0]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to contiguous must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    # contiguous is a no-op in TensorRT
    return input_map[input_node]


@converter("aten.unflatten.int")
def convert_unflatten(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch unflatten to TensorRT shuffle layer.

    unflatten.int(Tensor self, int dim, int[] sizes)

    Unflatten a dimension of the input tensor into multiple dimensions.

    Args:
        node: FX node representing the unflatten operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_unflatten") from e

    logger.debug(f"[TensorRT] Converting unflatten node: {node.name}")

    args = node.args

    input_node = args[0]
    dim = args[1]
    sizes = args[2]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to unflatten must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get shape from node metadata for reliability (TRT shapes can be invalid during network building)
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Handle negative dimension
    dim = _get_positive_dim(dim, ndim)

    # Build output shape by replacing dim with sizes
    output_shape = input_shape[:dim] + list(sizes) + input_shape[dim + 1:]

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for unflatten {node.name}")

    layer.reshape_dims = trt.Dims(output_shape)
    layer.name = f"unflatten_{node.name}"

    logger.debug(
        f"[TensorRT] Created unflatten layer: {layer.name}, "
        f"dim={dim}, sizes={sizes}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.rsub.Scalar")
def convert_rsub_scalar(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch rsub (reverse subtract) with scalar to TensorRT.

    rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
    Computes: other - self * alpha

    Args:
        node: FX node representing the rsub operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT and numpy are required for convert_rsub_scalar") from e

    logger.debug(f"[TensorRT] Converting rsub.Scalar node: {node.name}")

    args = node.args

    input_node = args[0]
    other = args[1]  # Scalar to subtract from
    alpha = args[2] if len(args) > 2 else 1.0

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to rsub must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # If alpha != 1, first multiply input by alpha
    if alpha != 1.0:
        alpha_weights = trt.Weights(np.array([alpha], dtype=np.float32))
        alpha_const = network.add_constant([1], alpha_weights)
        alpha_const.name = f"rsub_alpha_const_{node.name}"

        mul_layer = network.add_elementwise(
            input_trt, alpha_const.get_output(0), trt.ElementWiseOperation.PROD
        )
        mul_layer.name = f"rsub_mul_alpha_{node.name}"
        input_trt = mul_layer.get_output(0)

    # Create constant for 'other' scalar
    other_weights = trt.Weights(np.array([other], dtype=np.float32))
    other_const = network.add_constant([1], other_weights)
    other_const.name = f"rsub_other_const_{node.name}"

    # Compute: other - input (reverse subtract)
    layer = network.add_elementwise(
        other_const.get_output(0), input_trt, trt.ElementWiseOperation.SUB
    )

    if layer is None:
        raise RuntimeError(f"Failed to create elementwise layer for rsub {node.name}")

    layer.name = f"rsub_scalar_{node.name}"

    logger.debug(
        f"[TensorRT] Created rsub.Scalar layer: {layer.name}, other={other}, alpha={alpha}"
    )

    return layer.get_output(0)


__all__ = [
    "convert_slice",
    "convert_index",
    "convert_contiguous",
    "convert_unflatten",
    "convert_rsub_scalar",
]
