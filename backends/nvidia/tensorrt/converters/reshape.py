# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Reshape and View Operations.

This module provides converters for PyTorch tensor reshaping operations to TensorRT
shuffle layers.

Supported operations:
- aten.view.default: View tensor with new shape
- aten.reshape.default: Reshape tensor
- aten.flatten.using_ints: Flatten specified dimensions
- aten.squeeze.dim: Squeeze a specific dimension (remove size-1 dim)
- aten.unsqueeze.default: Unsqueeze (add dimension)
- aten.permute.default: Permute dimensions
- aten.transpose.int: Transpose two dimensions

Notes:
- All reshaping uses network.add_shuffle()
- shuffle.reshape_dims for shape changes
- shuffle.first_transpose for permutations
- Dynamic shapes require runtime shape computation
"""

import logging
from typing import Any, Dict, List, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    get_node_shape,
    get_trt_tensor_from_node,
)

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


def validate_view_reshape(node: torch.fx.Node) -> bool:
    """
    Validate that a view/reshape node can be converted to TensorRT.

    Args:
        node: FX node representing the operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_view_reshape: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Args: input, size (list of dims)
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_view_reshape: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_flatten(node: torch.fx.Node) -> bool:
    """
    Validate that a flatten node can be converted to TensorRT.

    Args:
        node: FX node representing the flatten operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # Args: input, start_dim, end_dim
    if len(args) < 3:
        logger.debug(
            f"[TensorRT] validate_flatten: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_squeeze_unsqueeze(node: torch.fx.Node) -> bool:
    """
    Validate that a squeeze/unsqueeze node can be converted to TensorRT.

    Args:
        node: FX node representing the squeeze/unsqueeze operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # Args: input, dim
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_squeeze_unsqueeze: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_permute(node: torch.fx.Node) -> bool:
    """
    Validate that a permute node can be converted to TensorRT.

    Args:
        node: FX node representing the permute operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # Args: input, dims (list of permutation)
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_permute: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_transpose(node: torch.fx.Node) -> bool:
    """
    Validate that a transpose node can be converted to TensorRT.

    Args:
        node: FX node representing the transpose operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # Args: input, dim0, dim1
    if len(args) < 3:
        logger.debug(
            f"[TensorRT] validate_transpose: node {node.name} has insufficient args"
        )
        return False

    return True


def validate_select(node: torch.fx.Node) -> bool:
    """
    Validate that a select node can be converted to TensorRT.

    Args:
        node: FX node representing the select operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # Args: input, dim, index
    if len(args) < 3:
        logger.debug(
            f"[TensorRT] validate_select: node {node.name} has insufficient args"
        )
        return False

    return True


def _compute_view_output_shape(
    node: torch.fx.Node,
    input_node: torch.fx.Node,
    input_trt: trt.ITensor,
    target_shape: List[int],
) -> List[int]:
    """Compute the output shape for view/reshape operations.

    Handles -1 dimension computation by calculating from input volume.

    Args:
        node: FX node representing the operation (used for metadata).
        input_node: FX node for the input tensor.
        input_trt: TensorRT tensor for the input.
        target_shape: Target shape specification (may contain -1).

    Returns:
        Computed output shape with -1 resolved.

    Raises:
        ValueError: If more than one -1 dimension is specified.
    """
    from executorch.backends.nvidia.tensorrt.converter_utils import resolve_shape

    # Prefer output shape from node metadata (most reliable).
    # resolve_shape maps SymInt values to -1 for TRT dynamic dims.
    if "val" in node.meta and hasattr(node.meta["val"], "shape"):
        return resolve_shape(node.meta["val"].shape)

    # Fall back to computing shape from target_shape, handling -1
    input_shape = list(get_node_shape(input_node) or input_trt.shape)

    # Calculate total input volume
    input_volume = 1
    for d in input_shape:
        if d > 0:
            input_volume *= d

    # Process target_shape, computing -1 dimensions
    output_shape = []
    neg_one_idx = -1
    known_volume = 1

    for i, dim in enumerate(target_shape):
        if isinstance(dim, int):
            if dim == -1:
                if neg_one_idx >= 0:
                    raise ValueError(
                        f"Only one -1 dimension allowed in view/reshape, "
                        f"found multiple at indices {neg_one_idx} and {i}"
                    )
                neg_one_idx = i
                output_shape.append(-1)  # Placeholder
            else:
                output_shape.append(dim)
                if dim > 0:
                    known_volume *= dim
        else:
            # Non-integer dimension (e.g., FX Node / symbolic) — TRT dynamic
            output_shape.append(-1)

    # Calculate the -1 dimension if present
    if neg_one_idx >= 0:
        if known_volume > 0:
            output_shape[neg_one_idx] = input_volume // known_volume
        else:
            # Cannot compute -1 dimension without knowing other dimensions
            output_shape[neg_one_idx] = 0  # Use 0 for TRT dynamic inference

    return output_shape


@converter("aten.view.default", "aten._unsafe_view.default", "aten.view_copy.default", validator_fn=validate_view_reshape)
def convert_view(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch view/unsafe_view to TensorRT shuffle layer.

    Args:
        node: FX node representing the view operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid or not found in input_map.
        RuntimeError: If TensorRT layer creation fails.
    """
    args = node.args

    input_node = args[0]
    target_shape = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to view must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(
            f"Input node '{input_node.name}' not found in input_map for "
            f"view node '{node.name}'"
        )

    input_trt = input_map[input_node]

    import tensorrt as trt
    import numpy as np

    # Get the actual output shape from node metadata if available
    output_shape = _compute_view_output_shape(node, input_node, input_trt, target_shape)
    logger.debug(f"[TensorRT] view {node.name}: output_shape = {output_shape}")

    num_dynamic = sum(1 for d in output_shape if d < 0)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for view {node.name}")

    if num_dynamic <= 1:
        # TRT handles at most one -1 in reshape natively.
        layer.reshape_dims = trt.Dims(output_shape)
    else:
        # Multiple dynamic dims: build shape tensor from the target_shape arg.
        # target_shape may contain FX Nodes (sym_size placeholders) which are
        # in input_map as TRT tensors.
        components = []
        for i, d in enumerate(target_shape):
            if isinstance(d, int):
                c = network.add_constant(
                    [1], trt.Weights(np.array([d], dtype=np.int32))
                )
                c.name = f"view_c{i}_{node.name}"
                components.append(c.get_output(0))
            elif isinstance(d, torch.fx.Node) and d in input_map:
                t = input_map[d]
                shuf = network.add_shuffle(t)
                shuf.reshape_dims = trt.Dims([1])
                shuf.name = f"view_sym{i}_{node.name}"
                cast = network.add_cast(shuf.get_output(0), trt.int32)
                cast.name = f"view_sym_i32_{i}_{node.name}"
                components.append(cast.get_output(0))
            else:
                c = network.add_constant(
                    [1], trt.Weights(np.array([-1], dtype=np.int32))
                )
                c.name = f"view_unk{i}_{node.name}"
                components.append(c.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"view_outshape_{node.name}"
        layer.set_input(1, shape_cat.get_output(0))

    layer.name = f"view_{node.name}"

    return layer.get_output(0)


@converter("aten.reshape.default", validator_fn=validate_view_reshape)
def convert_reshape(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch reshape to TensorRT shuffle layer.

    Args:
        node: FX node representing the reshape operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid or target_shape is malformed.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting reshape node: {node.name}")

    args = node.args

    input_node = args[0]
    target_shape = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to reshape must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(
            f"Input node '{input_node.name}' not found in input_map for "
            f"reshape node '{node.name}'"
        )

    if not isinstance(target_shape, (list, tuple)):
        raise ValueError(f"target_shape must be list or tuple, got {type(target_shape)}")

    input_trt = input_map[input_node]

    # Use the same shape computation logic as convert_view for consistency
    output_shape = _compute_view_output_shape(node, input_node, input_trt, list(target_shape))

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for reshape {node.name}")

    layer.reshape_dims = trt.Dims(output_shape)
    layer.name = f"reshape_{node.name}"
    logger.debug(f"[TensorRT] Created reshape layer: {layer.name}, shape={output_shape}")

    return layer.get_output(0)


@converter("aten.flatten.using_ints", validator_fn=validate_flatten)
def convert_flatten(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch flatten to TensorRT shuffle layer.

    Flatten merges dimensions from start_dim to end_dim (inclusive).

    Args:
        node: FX node representing the flatten operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid or dimensions are inconsistent.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting flatten node: {node.name}")

    args = node.args

    input_node = args[0]
    start_dim = args[1]
    end_dim = args[2]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to flatten must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(
            f"Input node '{input_node.name}' not found in input_map for "
            f"flatten node '{node.name}'"
        )

    input_trt = input_map[input_node]

    # Get shape from node metadata for reliability
    input_shape = tuple(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Handle negative dimensions
    start_dim = _get_positive_dim(start_dim, ndim)
    end_dim = _get_positive_dim(end_dim, ndim)

    # Validate dimensions
    if start_dim > end_dim:
        raise ValueError(f"start_dim ({start_dim}) must be <= end_dim ({end_dim})")

    # Build the output shape
    output_shape = []
    flatten_size = 1

    for i, s in enumerate(input_shape):
        if i < start_dim:
            output_shape.append(s)
        elif i <= end_dim:
            if s == -1:
                # Dynamic dimension - use 0 for TensorRT inference
                flatten_size = 0
            elif flatten_size != 0:
                flatten_size *= s
        else:
            if flatten_size is not None:
                output_shape.append(flatten_size if flatten_size != 0 else 0)
                flatten_size = None  # Mark as already added
            output_shape.append(s)

    # Add the flattened dimension if not yet added (end_dim is last dim)
    if flatten_size is not None:
        output_shape.append(flatten_size if flatten_size != 0 else 0)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for flatten {node.name}")

    layer.reshape_dims = trt.Dims(output_shape)
    layer.name = f"flatten_{node.name}"

    logger.debug(
        f"[TensorRT] Created flatten layer: {layer.name}, "
        f"start_dim={start_dim}, end_dim={end_dim}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze_copy.dim", "aten.squeeze_copy.dims", validator_fn=validate_squeeze_unsqueeze)
def convert_squeeze(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch squeeze to TensorRT shuffle layer.

    Removes dimension of size 1 at the specified position.

    Args:
        node: FX node representing the squeeze operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting squeeze node: {node.name}")

    args = node.args

    input_node = args[0]
    dim = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to squeeze must be a node, got {type(input_node)}")

    input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)

    # Get shape from node metadata for reliability
    input_shape = tuple(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Handle dims as list (squeeze.dims variant)
    if isinstance(dim, (list, tuple)):
        dims_to_squeeze = [_get_positive_dim(d, ndim) for d in dim]
    else:
        dims_to_squeeze = [_get_positive_dim(dim, ndim)]

    # Build output shape excluding squeezed dimensions
    output_shape = []
    for i, s in enumerate(input_shape):
        if i in dims_to_squeeze:
            # Only squeeze if size is 1 or dynamic
            if s != 1 and s != -1:
                logger.warning(
                    f"[TensorRT] squeeze on dim {i} with size {s} != 1, not squeezing"
                )
                output_shape.append(s)
            # else: skip this dimension (squeeze it)
        else:
            output_shape.append(s)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for squeeze {node.name}")

    layer.reshape_dims = trt.Dims(output_shape)
    layer.name = f"squeeze_{node.name}"

    logger.debug(
        f"[TensorRT] Created squeeze layer: {layer.name}, "
        f"dims={dims_to_squeeze}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.unsqueeze.default", "aten.unsqueeze_copy.default", validator_fn=validate_squeeze_unsqueeze)
def convert_unsqueeze(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch unsqueeze to TensorRT shuffle layer.

    Inserts a dimension of size 1 at the specified position.

    Args:
        node: FX node representing the unsqueeze operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting unsqueeze node: {node.name}")

    args = node.args

    input_node = args[0]
    dim = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to unsqueeze must be a node, got {type(input_node)}")

    input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)

    # Get shape from node metadata for reliability
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Handle negative dimension (for unsqueeze, target ndim is ndim + 1)
    dim = _get_positive_dim(dim, ndim + 1)

    # Build output shape with new dimension of size 1
    output_shape = input_shape[:dim] + [1] + input_shape[dim:]

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for unsqueeze {node.name}")

    layer.reshape_dims = trt.Dims(output_shape)
    layer.name = f"unsqueeze_{node.name}"

    logger.debug(
        f"[TensorRT] Created unsqueeze layer: {layer.name}, "
        f"dim={dim}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.permute.default", "aten.permute_copy.default", validator_fn=validate_permute)
def convert_permute(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch permute to TensorRT shuffle layer.

    Reorders dimensions according to the specified permutation.

    Args:
        node: FX node representing the permute operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid or dims is malformed.
        RuntimeError: If TensorRT layer creation fails.
    """
    args = node.args

    input_node = args[0]
    dims = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to permute must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(
            f"Input node '{input_node.name}' not found in input_map for "
            f"permute node '{node.name}'"
        )

    if not isinstance(dims, (list, tuple)):
        raise ValueError(f"dims must be list or tuple, got {type(dims)}")

    input_trt = input_map[input_node]

    # Get ndim from node metadata for reliability
    ndim = len(get_node_shape(input_node) or input_trt.shape)

    # Convert dims to list and handle negative indices
    permutation = [_get_positive_dim(d, ndim) for d in dims]

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for permute {node.name}")

    # Use first_transpose for permutation (applies before any reshape)
    layer.first_transpose = trt.Permutation(permutation)
    layer.name = f"permute_{node.name}"

    logger.debug(
        f"[TensorRT] Created permute layer: {layer.name}, permutation={permutation}"
    )

    return layer.get_output(0)


@converter("aten.transpose.int", validator_fn=validate_transpose)
def convert_transpose(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch transpose to TensorRT shuffle layer.

    Swaps two dimensions of the input tensor.

    Args:
        node: FX node representing the transpose operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting transpose node: {node.name}")

    args = node.args

    input_node = args[0]
    dim0 = args[1]
    dim1 = args[2]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to transpose must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(
            f"Input node '{input_node.name}' not found in input_map for "
            f"transpose node '{node.name}'"
        )

    input_trt = input_map[input_node]

    # Get ndim from node metadata for reliability
    ndim = len(get_node_shape(input_node) or input_trt.shape)

    # Handle negative dimensions
    dim0 = _get_positive_dim(dim0, ndim)
    dim1 = _get_positive_dim(dim1, ndim)

    # Build permutation: identity with dim0 and dim1 swapped
    permutation = list(range(ndim))
    permutation[dim0] = dim1
    permutation[dim1] = dim0

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for transpose {node.name}")

    layer.first_transpose = trt.Permutation(permutation)
    layer.name = f"transpose_{node.name}"

    logger.debug(
        f"[TensorRT] Created transpose layer: {layer.name}, "
        f"dim0={dim0}, dim1={dim1}, permutation={permutation}"
    )

    return layer.get_output(0)


@converter("aten.select.int", "aten.select_copy.int", validator_fn=validate_select)
def convert_select(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch select to TensorRT slice layer.

    Select extracts a slice of size 1 along a dimension and removes that dimension.
    Equivalent to tensor[dim, index].

    Args:
        node: FX node representing the select operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program (unused).

    Returns:
        TensorRT output tensor.

    Raises:
        ValueError: If input is invalid.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting select node: {node.name}")

    args = node.args

    input_node = args[0]
    dim = args[1]
    index = args[2]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to select must be a node, got {type(input_node)}")

    input_trt = get_trt_tensor_from_node(network, input_node, input_map, node.name)

    # Get shape from node metadata for reliability
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    ndim = len(input_shape)

    # Handle negative dimension
    dim = _get_positive_dim(dim, ndim)

    # Handle negative index
    if index < 0:
        index = input_shape[dim] + index

    # Build start, shape, stride for slice operation
    start = [0] * ndim
    start[dim] = index

    # Shape: same as input except the selected dim has size 1
    shape = input_shape.copy()
    shape[dim] = 1

    # Stride: 1 for all dims
    stride = [1] * ndim

    # Create slice layer
    layer = network.add_slice(
        input_trt,
        start=trt.Dims(start),
        shape=trt.Dims(shape),
        stride=trt.Dims(stride),
    )
    if layer is None:
        raise RuntimeError(f"Failed to create slice layer for select {node.name}")

    layer.name = f"select_slice_{node.name}"
    slice_output = layer.get_output(0)

    # Now squeeze the dimension to remove the size-1 dim
    output_shape = input_shape[:dim] + input_shape[dim + 1:]

    squeeze_layer = network.add_shuffle(slice_output)
    if squeeze_layer is None:
        raise RuntimeError(
            f"Failed to create shuffle layer for select squeeze {node.name}"
        )

    squeeze_layer.reshape_dims = trt.Dims(output_shape)
    squeeze_layer.name = f"select_squeeze_{node.name}"

    logger.debug(
        f"[TensorRT] Created select layer: {layer.name}, "
        f"dim={dim}, index={index}, output_shape={output_shape}"
    )

    return squeeze_layer.get_output(0)


__all__ = [
    "convert_view",
    "convert_reshape",
    "convert_flatten",
    "convert_squeeze",
    "convert_unsqueeze",
    "convert_permute",
    "convert_transpose",
    "convert_select",
    "validate_view_reshape",
    "validate_flatten",
    "validate_squeeze_unsqueeze",
    "validate_permute",
    "validate_transpose",
    "validate_select",
    "_compute_view_output_shape",
]
