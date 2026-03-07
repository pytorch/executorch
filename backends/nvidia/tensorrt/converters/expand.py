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
from executorch.backends.nvidia.tensorrt.converter_utils import (
    get_node_shape,
    input_has_dynamic_dims,
)

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


@converter("aten.expand.default", validator_fn=validate_expand, supports_dynamic_shapes=True)
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

    # Force dynamic (-1) for dims where expand_size is an FX Node (shape tensor),
    # since concretized SymInts report concrete values in metadata.
    for i, es in enumerate(expand_size):
        if isinstance(es, torch.fx.Node) and es in input_map and i < len(output_shape):
            output_shape = list(output_shape)
            output_shape[i] = -1
    output_shape = tuple(output_shape)

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

    import tensorrt as trt
    import numpy as np

    # Compute strides: 0 = broadcast from 1, 1 = keep.
    stride = []
    for inp_dim, out_dim in zip(input_shape, output_shape):
        stride.append(0 if inp_dim == 1 and out_dim != 1 else 1)

    has_dynamic = any(d < 0 for d in output_shape) or input_has_dynamic_dims(current_tensor)

    if not has_dynamic:
        # Static path — all dims are concrete.
        slice_layer = network.add_slice(
            current_tensor,
            start=[0] * output_dims,
            shape=list(output_shape),
            stride=stride,
        )
    else:
        # Dynamic path — build output shape via shape tensor API so TRT
        # can prove all dims are positive through the optimization profile.

        # Get runtime shape of input as int32.
        shape_layer = network.add_shape(current_tensor)
        shape_layer.name = f"expand_shape_{node.name}"
        shape_i32 = network.add_cast(
            shape_layer.get_output(0), trt.int32
        )
        shape_i32.name = f"expand_shape_i32_{node.name}"
        shape_trt = shape_i32.get_output(0)

        # Build one [1]-shaped component per output dim.
        # Check TRT tensor shape for actual dynamic dims (-1).
        trt_shape = current_tensor.shape
        components = []
        for i, (inp_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            if out_dim >= 0 and stride[i] != 0 and i < len(trt_shape) and trt_shape[i] == -1:
                # Non-broadcast dim that's dynamic in the TRT tensor —
                # gather from runtime input shape instead of baking
                # the trace-time constant.
                idx = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                idx.name = f"expand_dynidx{i}_{node.name}"
                g = network.add_gather(shape_trt, idx.get_output(0), axis=0)
                g.name = f"expand_dyng{i}_{node.name}"
                components.append(g.get_output(0))
            elif out_dim >= 0:
                # Concrete dim (static or broadcast target).
                c = network.add_constant(
                    [1], trt.Weights(np.array([out_dim], dtype=np.int32))
                )
                c.name = f"expand_c{i}_{node.name}"
                components.append(c.get_output(0))
            elif inp_dim == 1:
                # Broadcast: target size comes from expand_size arg.
                raw_target = expand_size[i]
                if isinstance(raw_target, torch.fx.Node) and raw_target in input_map:
                    t = input_map[raw_target]
                    shuf = network.add_shuffle(t)
                    shuf.reshape_dims = trt.Dims([1])
                    shuf.name = f"expand_tgt{i}_{node.name}"
                    cast = network.add_cast(shuf.get_output(0), trt.int32)
                    cast.name = f"expand_tgt_i32_{i}_{node.name}"
                    components.append(cast.get_output(0))
                else:
                    # Fallback: keep input dim.
                    idx = network.add_constant(
                        [1], trt.Weights(np.array([i], dtype=np.int32))
                    )
                    g = network.add_gather(shape_trt, idx.get_output(0), axis=0)
                    g.name = f"expand_g{i}_{node.name}"
                    components.append(g.get_output(0))
            else:
                # Dynamic, no broadcast: extract from input shape.
                idx = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                g = network.add_gather(shape_trt, idx.get_output(0), axis=0)
                g.name = f"expand_g{i}_{node.name}"
                components.append(g.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"expand_outshape_{node.name}"

        stride_c = network.add_constant(
            [len(stride)], trt.Weights(np.array(stride, dtype=np.int32))
        )
        stride_c.name = f"expand_stride_{node.name}"
        start_c = network.add_constant(
            [output_dims], trt.Weights(np.zeros(output_dims, dtype=np.int32))
        )
        start_c.name = f"expand_start_{node.name}"

        slice_layer = network.add_slice(
            current_tensor,
            start=[0] * output_dims,
            shape=[1] * output_dims,
            stride=[1] * output_dims,
        )
        slice_layer.set_input(1, start_c.get_output(0))
        slice_layer.set_input(2, shape_cat.get_output(0))
        slice_layer.set_input(3, stride_c.get_output(0))

    if slice_layer is None:
        raise RuntimeError(f"Failed to create slice layer for node {node.name}")
    slice_layer.name = f"expand_slice_{node.name}"

    logger.debug(
        f"[TensorRT] Created expand layers: {slice_layer.name}, "
        f"output_shape={output_shape}, stride={stride}"
    )

    return slice_layer.get_output(0)


@converter("aten.expand_copy.default", validator_fn=validate_expand, supports_dynamic_shapes=True)
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


@converter("aten.repeat.default", validator_fn=validate_expand, supports_dynamic_shapes=True)
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
