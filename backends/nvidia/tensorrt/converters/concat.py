# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Concatenation and Split Operations.

This module provides converters for PyTorch tensor concatenation and splitting
operations to TensorRT layers.

Supported operations:
- aten.cat.default: Concatenate tensors along an axis
- aten.stack.default: Stack tensors along a new axis
- aten.split.Tensor: Split tensor into chunks of given size
- aten.split_with_sizes.default: Split tensor into chunks with given sizes
- aten.chunk.default: Split tensor into specified number of chunks

Notes:
- Concatenation uses network.add_concatenation()
- Split/chunk uses network.add_slice() for each output chunk
- Stack is implemented as unsqueeze + concatenation
"""

import logging
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import get_node_shape

logger: logging.Logger = logging.getLogger(__name__)


def _get_positive_dim(dim: int, ndim: int) -> int:
    """
    Convert a potentially negative dimension index to positive.

    Args:
        dim: Dimension index (can be negative).
        ndim: Number of dimensions.

    Returns:
        Positive dimension index.
    """
    if dim < 0:
        dim = ndim + dim
    return dim


def validate_cat(node: torch.fx.Node) -> bool:
    """
    Validate that a cat node can be converted to TensorRT.

    Args:
        node: FX node representing the cat operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(f"[TensorRT] validate_cat: node {node.name} is not call_function")
        return False

    args = node.args
    # Args: tensors (list), dim (optional, default 0)
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_cat: node {node.name} has insufficient args"
        )
        return False

    tensors = args[0]
    if not isinstance(tensors, (list, tuple)) or len(tensors) < 1:
        logger.debug(
            f"[TensorRT] validate_cat: node {node.name} has invalid tensors arg"
        )
        return False

    return True


def validate_stack(node: torch.fx.Node) -> bool:
    """
    Validate that a stack node can be converted to TensorRT.

    Args:
        node: FX node representing the stack operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_stack: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Args: tensors (list), dim (optional, default 0)
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_stack: node {node.name} has insufficient args"
        )
        return False

    tensors = args[0]
    if not isinstance(tensors, (list, tuple)) or len(tensors) < 1:
        logger.debug(
            f"[TensorRT] validate_stack: node {node.name} has invalid tensors arg"
        )
        return False

    return True


def validate_split(node: torch.fx.Node) -> bool:
    """
    Validate that a split node can be converted to TensorRT.

    Args:
        node: FX node representing the split operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_split: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Args: input, split_size_or_sections, dim (optional, default 0)
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_split: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_chunk(node: torch.fx.Node) -> bool:
    """
    Validate that a chunk node can be converted to TensorRT.

    Args:
        node: FX node representing the chunk operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_chunk: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Args: input, chunks, dim (optional, default 0)
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_chunk: node {node.name} has insufficient args"
        )
        return False

    return True


@converter("aten.cat.default", validator_fn=validate_cat)
def convert_cat(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch cat to TensorRT concatenation layer.

    Args:
        node: FX node representing the cat operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_cat") from e

    logger.debug(f"[TensorRT] Converting cat node: {node.name}")

    args = node.args

    tensors = args[0]
    cat_dim = args[1] if len(args) > 1 else 0

    if not isinstance(tensors, (list, tuple)):
        raise ValueError(f"tensors must be list or tuple, got {type(tensors)}")

    # Convert all input nodes to TensorRT tensors
    trt_tensors = []
    input_nodes = []
    for tensor_node in tensors:
        if not isinstance(tensor_node, torch.fx.Node):
            raise ValueError(f"Input must be node, got {type(tensor_node)}")
        if tensor_node not in input_map:
            raise ValueError(f"Input node {tensor_node.name} not found in input_map")
        trt_tensors.append(input_map[tensor_node])
        input_nodes.append(tensor_node)

    if len(trt_tensors) == 0:
        raise ValueError("cat requires at least one input tensor")

    # Get number of dimensions from node metadata (more reliable for dynamic shapes)
    # TRT tensor.shape can have invalid length for dynamic shapes during network building
    ndim = None
    for input_node in input_nodes:
        shape = get_node_shape(input_node)
        if shape is not None:
            ndim = len(shape)
            break

    # Fallback to TRT shape if no metadata available
    if ndim is None:
        try:
            trt_shape = trt_tensors[0].shape
            if trt_shape is not None:
                ndim = len(trt_shape)
                if ndim < 0:
                    ndim = None
        except (ValueError, TypeError):
            pass

    if ndim is None:
        raise ValueError("Cannot determine number of dimensions for cat operation")

    cat_dim = _get_positive_dim(cat_dim, ndim)

    # Create concatenation layer
    layer = network.add_concatenation(trt_tensors)
    if layer is None:
        raise RuntimeError(f"Failed to create concatenation layer for cat {node.name}")

    layer.axis = cat_dim
    layer.name = f"cat_{node.name}"

    output = layer.get_output(0)

    # Safe output shape logging
    try:
        out_shape = list(output.shape) if output.shape is not None else "dynamic"
    except (ValueError, TypeError):
        out_shape = "dynamic"

    logger.debug(
        f"[TensorRT] Created cat layer: {layer.name}, "
        f"axis={cat_dim}, num_inputs={len(trt_tensors)}, output_shape={out_shape}"
    )

    return output


@converter("aten.stack.default", validator_fn=validate_stack)
def convert_stack(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch stack to TensorRT.

    Stack is implemented as unsqueeze on each tensor followed by concatenation.

    Args:
        node: FX node representing the stack operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_stack") from e

    logger.debug(f"[TensorRT] Converting stack node: {node.name}")

    args = node.args

    tensors = args[0]
    stack_dim = args[1] if len(args) > 1 else 0

    if not isinstance(tensors, (list, tuple)):
        raise ValueError(f"tensors must be list or tuple, got {type(tensors)}")

    # Convert all input nodes to TensorRT tensors
    trt_tensors = []
    input_nodes = []
    for tensor_node in tensors:
        if not isinstance(tensor_node, torch.fx.Node):
            raise ValueError(f"Input must be node, got {type(tensor_node)}")
        if tensor_node not in input_map:
            raise ValueError(f"Input node {tensor_node.name} not found in input_map")
        trt_tensors.append(input_map[tensor_node])
        input_nodes.append(tensor_node)

    if len(trt_tensors) == 0:
        raise ValueError("stack requires at least one input tensor")

    # Get number of dimensions from node metadata (output will have ndim + 1)
    ndim = None
    input_shape = None
    for input_node in input_nodes:
        shape = get_node_shape(input_node)
        if shape is not None:
            ndim = len(shape)
            input_shape = list(shape)
            break

    # Fallback to TRT shape if no metadata available
    if ndim is None:
        try:
            trt_shape = trt_tensors[0].shape
            if trt_shape is not None:
                ndim = len(trt_shape)
                if ndim >= 0:
                    input_shape = list(trt_shape)
                else:
                    ndim = None
        except (ValueError, TypeError):
            pass

    if ndim is None or input_shape is None:
        raise ValueError("Cannot determine number of dimensions for stack operation")

    stack_dim = _get_positive_dim(stack_dim, ndim + 1)

    # Unsqueeze each tensor at the stack dimension
    unsqueezed_tensors = []
    for i, trt_tensor in enumerate(trt_tensors):
        # Build output shape with new dimension of size 1
        output_shape = input_shape[:stack_dim] + [1] + input_shape[stack_dim:]

        shuffle_layer = network.add_shuffle(trt_tensor)
        if shuffle_layer is None:
            raise RuntimeError(
                f"Failed to create shuffle layer for stack unsqueeze {node.name}"
            )
        shuffle_layer.reshape_dims = trt.Dims(output_shape)
        shuffle_layer.name = f"stack_unsqueeze_{node.name}_{i}"
        unsqueezed_tensors.append(shuffle_layer.get_output(0))

    # Create concatenation layer on the new dimension
    layer = network.add_concatenation(unsqueezed_tensors)
    if layer is None:
        raise RuntimeError(
            f"Failed to create concatenation layer for stack {node.name}"
        )

    layer.axis = stack_dim
    layer.name = f"stack_{node.name}"

    output = layer.get_output(0)

    # Safe output shape logging
    try:
        out_shape = list(output.shape) if output.shape is not None else "dynamic"
    except (ValueError, TypeError):
        out_shape = "dynamic"

    logger.debug(
        f"[TensorRT] Created stack layer: {layer.name}, "
        f"dim={stack_dim}, num_inputs={len(trt_tensors)}, output_shape={out_shape}"
    )

    return output


@converter("aten.split.Tensor", "aten.split_with_sizes.default", "aten.split_with_sizes_copy.default", validator_fn=validate_split)
def convert_split(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> List[Any]:  # List[trt.ITensor]
    """
    Convert PyTorch split to TensorRT slice layers.

    Returns a list of output tensors (one for each split chunk).

    Args:
        node: FX node representing the split operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        List of TensorRT output tensors.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_split") from e

    logger.debug(f"[TensorRT] Converting split node: {node.name}")

    args = node.args

    input_node = args[0]
    split_size_or_sections = args[1]
    split_dim = args[2] if len(args) > 2 else 0

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input must be node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get shape from node metadata for reliability (TRT shapes can be invalid during network building)
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Normalize split dimension
    split_dim = _get_positive_dim(split_dim, ndim)

    # Get dimension size
    dim_size = input_shape[split_dim]
    if dim_size <= 0:
        raise ValueError(
            f"Cannot split dynamic dimension {split_dim} with size {dim_size}"
        )

    # Determine split sizes
    if isinstance(split_size_or_sections, int):
        # split_size: split into chunks of this size
        split_sizes = []
        remaining = dim_size
        while remaining > 0:
            chunk_size = min(split_size_or_sections, remaining)
            split_sizes.append(chunk_size)
            remaining -= chunk_size
    elif isinstance(split_size_or_sections, (list, tuple)):
        # split_with_sizes: list of sizes
        split_sizes = list(split_size_or_sections)
        if sum(split_sizes) != dim_size:
            raise ValueError(
                f"split_with_sizes: sum of sizes {sum(split_sizes)} != dim_size {dim_size}"
            )
    else:
        raise ValueError(
            f"split_size_or_sections must be int or list, got {type(split_size_or_sections)}"
        )

    # Create slice layers for each chunk
    outputs = []
    start_idx = 0

    for i, chunk_size in enumerate(split_sizes):
        # Build start, shape, stride tuples for slice
        start = [0] * ndim
        shape = list(input_shape)
        stride = [1] * ndim

        start[split_dim] = start_idx
        shape[split_dim] = chunk_size

        # Create slice layer
        layer = network.add_slice(
            input_trt, trt.Dims(start), trt.Dims(shape), trt.Dims(stride)
        )

        if layer is None:
            raise RuntimeError(
                f"Failed to create slice layer for split {node.name} chunk {i}"
            )

        layer.name = f"split_{node.name}_{i}"
        outputs.append(layer.get_output(0))

        start_idx += chunk_size

    logger.debug(
        f"[TensorRT] Created {len(outputs)} slice layers for split, "
        f"dim={split_dim}, sizes={split_sizes}"
    )

    return outputs


@converter("aten.chunk.default", validator_fn=validate_chunk)
def convert_chunk(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
) -> List[Any]:  # List[trt.ITensor]
    """
    Convert PyTorch chunk to TensorRT slice layers.

    Returns a list of output tensors (one for each chunk).

    Args:
        node: FX node representing the chunk operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        List of TensorRT output tensors.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_chunk") from e

    logger.debug(f"[TensorRT] Converting chunk node: {node.name}")

    args = node.args

    input_node = args[0]
    num_chunks = args[1]
    chunk_dim = args[2] if len(args) > 2 else 0

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input must be node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get shape from node metadata for reliability (TRT shapes can be invalid during network building)
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Normalize chunk dimension
    chunk_dim = _get_positive_dim(chunk_dim, ndim)

    # Get dimension size
    dim_size = input_shape[chunk_dim]
    if dim_size <= 0:
        raise ValueError(
            f"Cannot chunk dynamic dimension {chunk_dim} with size {dim_size}"
        )

    # Calculate chunk sizes (last chunk may be smaller)
    base_chunk_size = dim_size // num_chunks
    remainder = dim_size % num_chunks

    # Create slice layers for each chunk
    outputs = []
    start_idx = 0

    for i in range(num_chunks):
        # Chunks are as equal as possible, with earlier chunks getting +1 if there's a remainder
        if i < remainder:
            chunk_size = base_chunk_size + 1
        else:
            chunk_size = base_chunk_size

        if chunk_size == 0:
            # No more chunks (num_chunks > dim_size)
            break

        # Build start, shape, stride tuples for slice
        start = [0] * ndim
        shape = list(input_shape)
        stride = [1] * ndim

        start[chunk_dim] = start_idx
        shape[chunk_dim] = chunk_size

        # Create slice layer
        layer = network.add_slice(
            input_trt, trt.Dims(start), trt.Dims(shape), trt.Dims(stride)
        )

        if layer is None:
            raise RuntimeError(
                f"Failed to create slice layer for chunk {node.name} chunk {i}"
            )

        layer.name = f"chunk_{node.name}_{i}"
        outputs.append(layer.get_output(0))

        start_idx += chunk_size

    logger.debug(
        f"[TensorRT] Created {len(outputs)} slice layers for chunk, "
        f"dim={chunk_dim}, num_chunks={num_chunks}"
    )

    return outputs


__all__ = [
    "convert_cat",
    "convert_stack",
    "convert_split",
    "convert_chunk",
    "validate_cat",
    "validate_stack",
    "validate_split",
    "validate_chunk",
]
