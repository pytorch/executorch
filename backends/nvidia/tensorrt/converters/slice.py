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
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    get_node_shape,
    get_trt_tensor,
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

    # Handle case where input node is not in input_map (e.g., get_attr or lifted buffer)
    if input_node not in input_map:
        # Try to get the value from node metadata or create a constant
        input_val = None
        if "val" in input_node.meta and isinstance(input_node.meta["val"], torch.Tensor):
            input_val = input_node.meta["val"]
        
        if input_val is not None:
            input_trt = get_trt_tensor(network, input_val, f"const_{input_node.name}")
            input_map[input_node] = input_trt  # Cache for future use
            logger.debug(f"[TensorRT] Created constant tensor for {input_node.name} from metadata")
        else:
            raise ValueError(
                f"Input node {input_node.name} not found in input_map and no metadata value available. "
                f"This may be a lifted buffer that wasn't properly added. "
                f"Node op: {input_node.op}, target: {input_node.target}"
            )
    else:
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
    Supports both single-dimension and multi-dimension indexing (advanced indexing).

    For single index tensor: uses simple gather layer
    For multiple index tensors: uses transpose + flatten + gather + reshape pattern
    following TensorRT implementation.

    Args:
        node: FX node representing the index operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
        import numpy as np
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
    
    # Get input shape from metadata (more reliable for dynamic shapes)
    input_shape = list(get_node_shape(input_node) or input_trt.shape)
    rank = len(input_shape)

    # Collect non-None indices and their positions
    adv_indx_indices = []  # Dimension indices where indexing is applied
    tensor_indices = []    # The actual index tensors
    
    for i, idx in enumerate(indices):
        if idx is not None and isinstance(idx, torch.fx.Node):
            adv_indx_indices.append(i)
            if idx not in input_map:
                raise ValueError(f"Index node {idx.name} not found in input_map")
            tensor_indices.append(input_map[idx])

    if not tensor_indices:
        # No valid index tensors - just return input cast to int32
        cast_layer = network.add_cast(input_trt, trt.int32)
        cast_layer.name = f"index_casted_{node.name}"
        return cast_layer.get_output(0)
    
    if len(tensor_indices) == 1:
        # Simple single-dimension indexing - use gather directly
        gather_dim = adv_indx_indices[0]
        index_tensor = tensor_indices[0]
        
        # Ensure index is int32 for TensorRT gather
        if index_tensor.dtype != trt.int32:
            cast_layer = network.add_cast(index_tensor, trt.int32)
            cast_layer.name = f"index_cast_{node.name}"
            index_tensor = cast_layer.get_output(0)
        
        layer = network.add_gather(input_trt, index_tensor, axis=gather_dim)
        if layer is None:
            raise RuntimeError(f"Failed to create gather layer for index {node.name}")
        layer.name = f"index_gather_{node.name}"
        
        logger.debug(
            f"[TensorRT] Created simple index/gather layer: {layer.name}, gather_dim={gather_dim}"
        )
        return layer.get_output(0)
    
    # Multiple index tensors - advanced indexing
    # Follow TensorRT pattern: transpose -> flatten -> compute linear index -> gather -> reshape
    logger.debug(f"[TensorRT] Advanced indexing with {len(tensor_indices)} index tensors at dims {adv_indx_indices}")
    
    adv_indx_count = len(adv_indx_indices)
    
    # Step 1: Transpose input to move indexed dimensions to the front
    # new_order: [indexed dims..., non-indexed dims...]
    new_order = adv_indx_indices.copy()
    for i in range(rank):
        if i not in adv_indx_indices:
            new_order.append(i)
    
    transpose_layer = network.add_shuffle(input_trt)
    transpose_layer.second_transpose = trt.Permutation(new_order)
    transpose_layer.name = f"index_transpose_{node.name}"
    transpose_tensor = transpose_layer.get_output(0)
    
    logger.debug(f"[TensorRT] Transpose order: {new_order}")
    
    # Step 2: Flatten the indexed dimensions into one, and non-indexed dims into another
    # Result shape: [prod(indexed_dims), prod(non_indexed_dims)]
    
    # Compute products for reshape
    mult_d0 = 1  # Product of indexed dimension sizes
    for i in range(adv_indx_count):
        dim_size = input_shape[adv_indx_indices[i]]
        if isinstance(dim_size, int) and dim_size > 0:
            mult_d0 *= dim_size
        else:
            # Dynamic dimension - fall back to simpler approach
            raise NotImplementedError(
                f"Dynamic shapes in indexed dimensions not fully supported. "
                f"Dimension {adv_indx_indices[i]} has dynamic size."
            )
    
    mult_d1 = 1  # Product of non-indexed dimension sizes
    for i in range(rank):
        if i not in adv_indx_indices:
            dim_size = input_shape[i]
            if isinstance(dim_size, int) and dim_size > 0:
                mult_d1 *= dim_size
            else:
                raise NotImplementedError(
                    f"Dynamic shapes in non-indexed dimensions not fully supported. "
                    f"Dimension {i} has dynamic size."
                )
    
    # Create reshape to [mult_d0, mult_d1]
    flatten_layer = network.add_shuffle(transpose_tensor)
    flatten_layer.reshape_dims = trt.Dims([mult_d0, mult_d1])
    flatten_layer.name = f"index_flatten_{node.name}"
    flatten_tensor = flatten_layer.get_output(0)
    
    logger.debug(f"[TensorRT] Flattened shape: [{mult_d0}, {mult_d1}]")
    
    # Step 3: Compute cumulative linear index following TensorRT formula
    # tensor_index = sum_{i=1}^m (ind_i * prod_{j=i+1}^m (dim_j))
    # where ind_i is the i-th index tensor and dim_j is the size of the j-th indexed dimension
    
    # First, find the maximum number of dimensions across all index tensors
    # TensorRT requires matching dimensions for elementwise operations
    max_ndim = 1
    for idx_tensor in tensor_indices:
        idx_ndim = len(idx_tensor.shape)
        if idx_ndim > max_ndim:
            max_ndim = idx_ndim
    
    logger.debug(f"[TensorRT] Max ndim across index tensors: {max_ndim}")
    
    # Helper function to ensure tensor has max_ndim dimensions by prepending 1s
    def ensure_ndim(tensor: Any, name_suffix: str) -> Any:
        """Reshape tensor to have max_ndim dimensions by prepending 1s."""
        current_ndim = len(tensor.shape)
        if current_ndim < max_ndim:
            # Need to prepend (max_ndim - current_ndim) dimensions of size 1
            new_shape = [1] * (max_ndim - current_ndim) + list(tensor.shape)
            reshape_layer = network.add_shuffle(tensor)
            reshape_layer.reshape_dims = trt.Dims(new_shape)
            reshape_layer.name = f"index_reshape_{name_suffix}"
            return reshape_layer.get_output(0)
        return tensor
    
    # Start with the last index tensor (no multiplication needed for the last one)
    cum_index = tensor_indices[adv_indx_count - 1]
    
    # Ensure int32 type
    if cum_index.dtype != trt.int32:
        cast_layer = network.add_cast(cum_index, trt.int32)
        cast_layer.name = f"index_cast_last_{node.name}"
        cum_index = cast_layer.get_output(0)
    
    # Ensure cum_index has max_ndim dimensions
    cum_index = ensure_ndim(cum_index, f"last_{node.name}")
    
    # The multiplier accumulates the product of indexed dimension sizes
    # Start with the size of the LAST indexed dimension
    multiplier = input_shape[adv_indx_indices[adv_indx_count - 1]]
    
    logger.debug(f"[TensorRT] Starting multiplier: {multiplier} (size of dim {adv_indx_indices[adv_indx_count - 1]})")
    
    # Process from second-to-last index tensor backwards to first
    for i in range(adv_indx_count - 2, -1, -1):
        idx_tensor = tensor_indices[i]
        
        # Ensure int32 type
        if idx_tensor.dtype != trt.int32:
            cast_layer = network.add_cast(idx_tensor, trt.int32)
            cast_layer.name = f"index_cast_{i}_{node.name}"
            idx_tensor = cast_layer.get_output(0)
        
        # Ensure idx_tensor has max_ndim dimensions
        idx_tensor = ensure_ndim(idx_tensor, f"{i}_{node.name}")
        
        # Create multiplier constant with max_ndim dimensions for proper broadcasting
        mult_shape = [1] * max_ndim
        mult_const = network.add_constant(
            mult_shape, 
            trt.Weights(np.array([multiplier], dtype=np.int32).reshape(mult_shape))
        )
        mult_const.name = f"index_mult_const_{i}_{node.name}"
        
        # adv_index = idx_tensor * multiplier
        mul_layer = network.add_elementwise(
            idx_tensor, 
            mult_const.get_output(0), 
            trt.ElementWiseOperation.PROD
        )
        mul_layer.name = f"index_mul_{i}_{node.name}"
        
        # cum_index = cum_index + adv_index
        add_layer = network.add_elementwise(
            cum_index, 
            mul_layer.get_output(0), 
            trt.ElementWiseOperation.SUM
        )
        add_layer.name = f"index_add_{i}_{node.name}"
        cum_index = add_layer.get_output(0)
        
        # Update multiplier for next iteration: multiplier *= dim_size[current_indexed_dim]
        dim_size = input_shape[adv_indx_indices[i]]
        if isinstance(dim_size, int) and dim_size > 0:
            multiplier *= dim_size
        else:
            raise NotImplementedError(
                f"Dynamic shapes in indexed dimensions not fully supported. "
                f"Dimension {adv_indx_indices[i]} has dynamic size."
            )
        
        logger.debug(f"[TensorRT] After index {i}: multiplier = {multiplier}")
    
    logger.debug(f"[TensorRT] Computed cumulative index")
    
    # Step 4: Gather using cumulative index on the flattened dimension 0
    gather_layer = network.add_gather(flatten_tensor, cum_index, axis=0)
    if gather_layer is None:
        raise RuntimeError(f"Failed to create gather layer for advanced index {node.name}")
    gather_layer.name = f"index_gather_adv_{node.name}"
    gather_out = gather_layer.get_output(0)
    
    logger.debug(f"[TensorRT] Gather output shape: {gather_out.shape}")
    
    # Step 5: Reshape output to match expected shape from node metadata
    # The gather output shape is [cum_index_shape..., mult_d1]
    # We reshape directly to the expected output shape from node metadata
    
    # Get expected output shape from node metadata (most reliable source)
    expected_output_shape = get_node_shape(node)
    
    if expected_output_shape is not None:
        output_shape = list(expected_output_shape)
        logger.debug(f"[TensorRT] Using expected output shape from metadata: {output_shape}")
        
        # Reshape gather output directly to expected shape
        reshape_layer = network.add_shuffle(gather_out)
        reshape_layer.reshape_dims = trt.Dims(output_shape)
        reshape_layer.name = f"index_final_reshape_{node.name}"
        final_output = reshape_layer.get_output(0)
    else:
        # Fallback: Get index tensor shape from metadata and compute output shape
        # Get the broadcast shape of all index tensors from their node metadata
        idx_broadcast_shape = []
        for i, idx in enumerate(indices):
            if idx is not None and isinstance(idx, torch.fx.Node):
                idx_shape = get_node_shape(idx)
                if idx_shape is not None:
                    idx_shape_list = list(idx_shape)
                    # Broadcast shapes - keep the max along each dimension
                    if not idx_broadcast_shape:
                        idx_broadcast_shape = idx_shape_list
                    else:
                        # Extend to match dimensions
                        while len(idx_broadcast_shape) < len(idx_shape_list):
                            idx_broadcast_shape.insert(0, 1)
                        while len(idx_shape_list) < len(idx_broadcast_shape):
                            idx_shape_list.insert(0, 1)
                        # Take max at each position
                        idx_broadcast_shape = [max(a, b) for a, b in zip(idx_broadcast_shape, idx_shape_list)]
        
        # Collect non-indexed dimensions
        non_indexed_dims = []
        for i in range(rank):
            if i not in adv_indx_indices:
                non_indexed_dims.append(input_shape[i])
        
        # Output shape: [idx_broadcast_shape..., non_indexed_dims...]
        output_shape = idx_broadcast_shape + non_indexed_dims
        logger.debug(f"[TensorRT] Computed output shape: {output_shape}")
        
        reshape_layer = network.add_shuffle(gather_out)
        reshape_layer.reshape_dims = trt.Dims(output_shape)
        reshape_layer.name = f"index_final_reshape_{node.name}"
        final_output = reshape_layer.get_output(0)
    
    logger.debug(f"[TensorRT] Created advanced index with output shape {output_shape}")
    return final_output


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
