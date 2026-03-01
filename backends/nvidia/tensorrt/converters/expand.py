# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Expand and Repeat Operations.

Supported operations:
- aten.expand.default: Expands a tensor to a larger size with broadcasting
- aten.expand_copy.default: Same as expand but with copy semantics
- aten.repeat.default: Repeats tensor along specified dimensions

TensorRT uses ISliceLayer with stride for broadcasting or IShuffleLayer for reshaping.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import get_node_shape

logger: logging.Logger = logging.getLogger(__name__)


def validate_expand(node: torch.fx.Node) -> bool:
    """Validate that an expand node can be converted to TensorRT.

    Args:
        node: FX node representing the expand operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_expand: node {node.name} is not call_function"
        )
        return False

    args = node.args
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_expand: node {node.name} has insufficient args"
        )
        return False

    if not isinstance(args[0], torch.fx.Node):
        logger.debug(
            f"[TensorRT] validate_expand: input is not a node, got {type(args[0])}"
        )
        return False

    return True


def _get_expand_output_shape(
    input_shape: Tuple, expand_shape: List
) -> Tuple[int, ...]:
    """Calculate the output shape after expand operation.

    Handles both concrete and symbolic (SymInt / FX Node) dimensions.
    Symbolic dimensions are resolved to ``-1`` so TRT treats them as dynamic.
    """
    from executorch.backends.nvidia.tensorrt.converter_utils import resolve_sym_dim

    input_dims = len(input_shape)
    expand_dims = len(expand_shape)

    if expand_dims > input_dims:
        input_shape = (1,) * (expand_dims - input_dims) + tuple(input_shape)

    output_shape = []
    for inp_dim, exp_dim in zip(input_shape, expand_shape):
        inp_resolved = resolve_sym_dim(inp_dim)
        exp_resolved = resolve_sym_dim(exp_dim)

        if exp_resolved == -1:
            # Symbolic expand target → dynamic
            output_shape.append(-1)
        elif isinstance(exp_dim, int) and exp_dim == -1:
            # Explicit -1 means keep original
            output_shape.append(inp_resolved)
        elif inp_resolved == 1 or inp_resolved == exp_resolved or inp_resolved == -1:
            output_shape.append(exp_resolved)
        else:
            raise ValueError(
                f"Cannot expand: input dim {inp_dim} is not 1 and != {exp_dim}"
            )

    return tuple(output_shape)


@converter("aten.expand.default", validator_fn=validate_expand)
def convert_expand(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch expand to TensorRT.

    PyTorch signature:
        aten.expand(Tensor self, SymInt[] size) -> Tensor

    Expand creates a view of the tensor with additional dimensions.
    For TensorRT, we use ISliceLayer with step=0 for broadcast dimensions.

    Args:
        node: FX node representing the expand operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    logger.debug(f"[TensorRT] Converting expand node: {node.name}")

    args = node.args
    input_node = args[0]
    expand_size = list(args[1]) if len(args) > 1 else list(node.kwargs.get("size", []))

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to expand must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    from executorch.backends.nvidia.tensorrt.converter_utils import resolve_shape

    input_trt = input_map[input_node]

    # Get shape from node metadata; resolve SymInt → -1 for TRT.
    raw_input_shape = get_node_shape(input_node) or tuple(input_trt.shape)
    input_shape = resolve_shape(raw_input_shape)

    # Calculate output shape (already handles FX Node / SymInt args)
    output_shape = _get_expand_output_shape(raw_input_shape, expand_size)

    logger.debug(
        f"[TensorRT] expand: input_shape={input_shape}, output_shape={output_shape}"
    )

    # First, add leading dimensions if needed
    input_dims = len(input_shape)
    output_dims = len(output_shape)

    current_tensor = input_trt

    if output_dims > input_dims:
        new_shape = [1] * (output_dims - input_dims) + input_shape
        shuffle = network.add_shuffle(current_tensor)
        if shuffle is None:
            raise RuntimeError(f"Failed to create shuffle layer for node {node.name}")
        shuffle.reshape_dims = new_shape
        shuffle.name = f"expand_reshape_{node.name}"
        current_tensor = shuffle.get_output(0)
        input_shape = new_shape

    # Broadcast using ISliceLayer: stride=0 for dims that need broadcasting.
    start = [0] * output_dims
    shape = list(output_shape)
    stride = []

    for inp_dim, out_dim in zip(input_shape, output_shape):
        if inp_dim == 1 and out_dim != 1:
            stride.append(0)
        else:
            stride.append(1)

    # Use slice layer for broadcasting
    slice_layer = network.add_slice(
        current_tensor,
        start=start,
        shape=shape,
        stride=stride,
    )
    if slice_layer is None:
        raise RuntimeError(f"Failed to create slice layer for node {node.name}")
    slice_layer.name = f"expand_slice_{node.name}"
    
    logger.debug(
        f"[TensorRT] Created expand layers: {slice_layer.name}, "
        f"output_shape={output_shape}, stride={stride}"
    )

    return slice_layer.get_output(0)


@converter("aten.expand_copy.default", validator_fn=validate_expand)
def convert_expand_copy(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch expand_copy to TensorRT.

    Same as expand but with copy semantics. In TensorRT, there's no
    difference since all operations create new tensors.
    """
    return convert_expand(node, network, input_map, edge_program)


@converter("aten.repeat.default", validator_fn=validate_expand)
def convert_repeat(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch repeat to TensorRT.

    PyTorch signature:
        aten.repeat(Tensor self, SymInt[] repeats) -> Tensor

    Repeat copies data along dimensions. We implement this using
    tile operation via concat layers or a combination of reshape + broadcast.

    Args:
        node: FX node representing the repeat operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    logger.debug(f"[TensorRT] Converting repeat node: {node.name}")

    args = node.args
    input_node = args[0]
    repeats = list(args[1]) if len(args) > 1 else list(node.kwargs.get("repeats", []))

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to repeat must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    from executorch.backends.nvidia.tensorrt.converter_utils import (
        resolve_shape,
        resolve_sym_dim,
    )

    input_trt = input_map[input_node]

    input_shape = resolve_shape(get_node_shape(input_node) or tuple(input_trt.shape))
    repeats = [resolve_sym_dim(r) for r in repeats]

    logger.debug(f"[TensorRT] repeat: input_shape={input_shape}, repeats={repeats}")

    # Pad input shape if repeats has more dimensions
    input_dims = len(input_shape)
    repeat_dims = len(repeats)
    
    current_tensor = input_trt
    
    if repeat_dims > input_dims:
        # Add leading dimensions using shuffle
        new_shape = [1] * (repeat_dims - input_dims) + input_shape
        shuffle = network.add_shuffle(current_tensor)
        if shuffle is None:
            raise RuntimeError(f"Failed to create shuffle layer for node {node.name}")
        shuffle.reshape_dims = new_shape
        shuffle.name = f"repeat_reshape_{node.name}"
        current_tensor = shuffle.get_output(0)
        input_shape = new_shape

    # For each dimension with repeat > 1, concatenate the tensor
    for dim, repeat_count in enumerate(repeats):
        if repeat_count > 1:
            # Create concat of repeat_count copies along this dimension
            tensors_to_concat = [current_tensor] * repeat_count
            concat_layer = network.add_concatenation(tensors_to_concat)
            if concat_layer is None:
                raise RuntimeError(
                    f"Failed to create concat layer for repeat dim {dim}"
                )
            concat_layer.axis = dim
            concat_layer.name = f"repeat_concat_dim{dim}_{node.name}"
            current_tensor = concat_layer.get_output(0)

    logger.debug(f"[TensorRT] Created repeat layers for node: {node.name}")

    return current_tensor


__all__ = [
    "convert_expand",
    "convert_expand_copy",
    "convert_repeat",
    "validate_expand",
]
