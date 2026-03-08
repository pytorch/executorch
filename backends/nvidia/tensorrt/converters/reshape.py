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

import numpy as np
import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    build_reshape_shape_tensor,
    get_node_shape,
    get_shape_with_dynamic_shape,
    get_trt_tensor_from_node,
    input_has_dynamic_dims,
    resolve_shape,
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


@converter("aten.view.default", "aten._unsafe_view.default", "aten.view_copy.default", validator_fn=validate_view_reshape, supports_dynamic_shapes=True)
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

    # Get the actual output shape from node metadata if available
    output_shape = _compute_view_output_shape(node, input_node, input_trt, target_shape)

    # If target_shape contains FX Nodes (shape tensor inputs for dynamic dims),
    # force those dimensions to -1 even if metadata reports concrete values
    # (which happens when SymInt expressions are concretized).
    for i, d in enumerate(target_shape):
        if isinstance(d, torch.fx.Node) and d in input_map and i < len(output_shape):
            output_shape[i] = -1
    logger.debug(f"[TensorRT] view {node.name}: output_shape = {output_shape}")

    num_dynamic = sum(1 for d in output_shape if d < 0)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for view {node.name}")

    # Following torch-tensorrt's reshape pattern: if all shape dims are
    # plain ints, use reshape_dims directly. Otherwise, convert each dim
    # to a [1]-shaped int32 TRT tensor, concatenate, and use set_input(1).
    all_int = all(isinstance(d, int) for d in target_shape)

    if all_int and num_dynamic == 0 and not input_has_dynamic_dims(input_trt):
        layer.reshape_dims = trt.Dims(output_shape)
    elif all_int and num_dynamic <= 1:
        layer.reshape_dims = trt.Dims(output_shape)
    else:
        # Convert each dim to a TRT tensor (like torch-tensorrt's reshape).
        # For -1 dims: compute from input_volume / product_of_other_dims.
        trt_dims = []
        neg1_idx = -1
        for i, d in enumerate(target_shape):
            if isinstance(d, torch.fx.Node) and d in input_map:
                t = input_map[d]
                shuf = network.add_shuffle(t)
                shuf.reshape_dims = trt.Dims([1])
                shuf.name = f"view_sym{i}_{node.name}"
                cast = network.add_cast(shuf.get_output(0), trt.int32)
                cast.name = f"view_cast{i}_{node.name}"
                trt_dims.append(cast.get_output(0))
            elif isinstance(d, int) and d == -1:
                neg1_idx = i
                trt_dims.append(None)  # placeholder, filled below
            else:
                val = int(d) if isinstance(d, int) else output_shape[i]
                c = network.add_constant(
                    [1], trt.Weights(np.array([val], dtype=np.int32))
                )
                c.name = f"view_d{i}_{node.name}"
                trt_dims.append(c.get_output(0))

        # Compute -1 dim: input_volume / product_of_known_dims
        if neg1_idx >= 0:
            # Get input volume as shape tensor
            shape_l = network.add_shape(input_trt)
            shape_l.name = f"view_inshape_{node.name}"
            shape_i32 = network.add_cast(shape_l.get_output(0), trt.int32)
            shape_i32.name = f"view_inshape_i32_{node.name}"
            in_ndim = len(input_trt.shape)
            idx0 = network.add_constant([1], trt.Weights(np.array([0], dtype=np.int32)))
            idx0.name = f"view_vidx0_{node.name}"
            vol = network.add_gather(shape_i32.get_output(0), idx0.get_output(0), axis=0)
            vol.name = f"view_vol0_{node.name}"
            vol_out = vol.get_output(0)
            for j in range(1, in_ndim):
                idx_j = network.add_constant([1], trt.Weights(np.array([j], dtype=np.int32)))
                idx_j.name = f"view_vidx{j}_{node.name}"
                g = network.add_gather(shape_i32.get_output(0), idx_j.get_output(0), axis=0)
                g.name = f"view_vg{j}_{node.name}"
                m = network.add_elementwise(vol_out, g.get_output(0), trt.ElementWiseOperation.PROD)
                m.name = f"view_vmul{j}_{node.name}"
                vol_out = m.get_output(0)

            # Product of known dims (all except -1)
            known_prod = None
            for i, t in enumerate(trt_dims):
                if i == neg1_idx or t is None:
                    continue
                if known_prod is None:
                    known_prod = t
                else:
                    m = network.add_elementwise(known_prod, t, trt.ElementWiseOperation.PROD)
                    m.name = f"view_kp{i}_{node.name}"
                    known_prod = m.get_output(0)

            if known_prod is not None:
                inferred = network.add_elementwise(vol_out, known_prod, trt.ElementWiseOperation.FLOOR_DIV)
                inferred.name = f"view_infer_{node.name}"
                # Reshape to [1] for concat
                shuf_inf = network.add_shuffle(inferred.get_output(0))
                shuf_inf.reshape_dims = trt.Dims([1])
                shuf_inf.name = f"view_infer_shuf_{node.name}"
                trt_dims[neg1_idx] = shuf_inf.get_output(0)
            else:
                # All other dims are -1 or FX nodes — use vol directly
                shuf_v = network.add_shuffle(vol_out)
                shuf_v.reshape_dims = trt.Dims([1])
                shuf_v.name = f"view_vol_shuf_{node.name}"
                trt_dims[neg1_idx] = shuf_v.get_output(0)

        shape_layer = network.add_concatenation(trt_dims)
        shape_layer.axis = 0
        shape_layer.name = f"view_shape_{node.name}"
        layer.set_input(1, shape_layer.get_output(0))

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

    all_int = all(isinstance(d, int) for d in target_shape)
    num_dynamic = sum(1 for d in output_shape if d < 0)

    if all_int and num_dynamic <= 1:
        layer.reshape_dims = trt.Dims(output_shape)
    else:
        trt_dims = []
        for i, d in enumerate(target_shape):
            if isinstance(d, torch.fx.Node) and d in input_map:
                t = input_map[d]
                shuf = network.add_shuffle(t)
                shuf.reshape_dims = trt.Dims([1])
                shuf.name = f"resh_sym{i}_{node.name}"
                cast = network.add_cast(shuf.get_output(0), trt.int32)
                cast.name = f"resh_cast{i}_{node.name}"
                trt_dims.append(cast.get_output(0))
            else:
                val = int(d) if isinstance(d, int) else output_shape[i]
                c = network.add_constant(
                    [1], trt.Weights(np.array([val], dtype=np.int32))
                )
                c.name = f"resh_d{i}_{node.name}"
                trt_dims.append(c.get_output(0))
        shape_layer = network.add_concatenation(trt_dims)
        shape_layer.axis = 0
        shape_layer.name = f"resh_shape_{node.name}"
        layer.set_input(1, shape_layer.get_output(0))

    layer.name = f"reshape_{node.name}"
    logger.debug(f"[TensorRT] Created reshape layer: {layer.name}, shape={output_shape}")

    return layer.get_output(0)


@converter("aten.flatten.using_ints", validator_fn=validate_flatten, supports_dynamic_shapes=True)
def convert_flatten(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch flatten to TensorRT shuffle layer.

    Flatten merges dimensions from start_dim to end_dim (inclusive).
    Supports dynamic shapes via the shape tensor API when multiple
    dimensions are dynamic.

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

    # Resolve shape: SymInts become -1 for TRT dynamic dims
    input_shape = resolve_shape(get_node_shape(input_node) or tuple(input_trt.shape))
    ndim = len(input_shape)

    # Handle negative dimensions
    start_dim = _get_positive_dim(start_dim, ndim)
    end_dim = _get_positive_dim(end_dim, ndim)

    # Validate dimensions
    if start_dim > end_dim:
        raise ValueError(f"start_dim ({start_dim}) must be <= end_dim ({end_dim})")

    # Build output shape: [pre_range..., flattened_dim, post_range...]
    pre_range = input_shape[:start_dim]
    flat_range = input_shape[start_dim:end_dim + 1]
    post_range = input_shape[end_dim + 1:]

    # Compute flattened dim statically if all dims in range are concrete
    all_flat_concrete = all(d > 0 for d in flat_range)
    if all_flat_concrete:
        flat_size = 1
        for d in flat_range:
            flat_size *= d
    else:
        flat_size = -1  # Dynamic — needs runtime computation

    output_shape = list(pre_range) + [flat_size] + list(post_range)

    num_dynamic = sum(1 for d in output_shape if d < 0)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for flatten {node.name}")

    if num_dynamic <= 1 and not input_has_dynamic_dims(input_trt):
        layer.reshape_dims = trt.Dims(output_shape)
    elif num_dynamic <= 1:
        # TRT handles at most one -1 in reshape natively.
        layer.reshape_dims = trt.Dims(output_shape)
    else:
        # Multiple dynamic dims: build shape tensor.
        shape_layer = network.add_shape(input_trt)
        shape_layer.name = f"flatten_shape_{node.name}"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"flatten_shape_i32_{node.name}"
        shape_trt = shape_i32.get_output(0)

        components: List[trt.ITensor] = []

        # Pre-range dims: pass through from input
        for i in range(start_dim):
            if input_shape[i] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([input_shape[i]], dtype=np.int32))
                )
                c.name = f"flatten_pre{i}_{node.name}"
                components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                idx_c.name = f"flatten_preidx{i}_{node.name}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"flatten_preg{i}_{node.name}"
                components.append(g.get_output(0))

        # Flattened dim: product of dims in [start_dim, end_dim]
        if all_flat_concrete:
            c = network.add_constant(
                [1], trt.Weights(np.array([flat_size], dtype=np.int32))
            )
            c.name = f"flatten_flat_{node.name}"
            components.append(c.get_output(0))
        else:
            # Runtime product: gather each dim, multiply together iteratively
            idx_c = network.add_constant(
                [1], trt.Weights(np.array([start_dim], dtype=np.int32))
            )
            idx_c.name = f"flatten_flatidx{start_dim}_{node.name}"
            product = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
            product.name = f"flatten_flatg{start_dim}_{node.name}"
            product_out = product.get_output(0)

            for j in range(start_dim + 1, end_dim + 1):
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([j], dtype=np.int32))
                )
                idx_c.name = f"flatten_flatidx{j}_{node.name}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"flatten_flatg{j}_{node.name}"
                mul = network.add_elementwise(
                    product_out, g.get_output(0),
                    trt.ElementWiseOperation.PROD,
                )
                mul.name = f"flatten_flatmul{j}_{node.name}"
                product_out = mul.get_output(0)

            components.append(product_out)

        # Post-range dims: pass through from input
        for i in range(end_dim + 1, ndim):
            if input_shape[i] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([input_shape[i]], dtype=np.int32))
                )
                c.name = f"flatten_post{i}_{node.name}"
                components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                idx_c.name = f"flatten_postidx{i}_{node.name}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"flatten_postg{i}_{node.name}"
                components.append(g.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"flatten_outshape_{node.name}"
        layer.set_input(1, shape_cat.get_output(0))

    layer.name = f"flatten_{node.name}"

    logger.debug(
        f"[TensorRT] Created flatten layer: {layer.name}, "
        f"start_dim={start_dim}, end_dim={end_dim}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze_copy.dim", "aten.squeeze_copy.dims", validator_fn=validate_squeeze_unsqueeze, supports_dynamic_shapes=True)
def convert_squeeze(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch squeeze to TensorRT shuffle layer.

    Removes dimension of size 1 at the specified position.
    Supports dynamic shapes via the shape tensor API when multiple
    dimensions are dynamic.

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

    # Resolve shape: SymInts become -1 for TRT dynamic dims
    input_shape = resolve_shape(get_node_shape(input_node) or tuple(input_trt.shape))
    ndim = len(input_shape)

    # Handle dims as list (squeeze.dims variant)
    if isinstance(dim, (list, tuple)):
        dims_to_squeeze = [_get_positive_dim(d, ndim) for d in dim]
    else:
        dims_to_squeeze = [_get_positive_dim(dim, ndim)]

    # Build output shape excluding squeezed dimensions, tracking kept input indices
    output_shape: List[int] = []
    kept_dims: List[int] = []
    for i, s in enumerate(input_shape):
        if i in dims_to_squeeze:
            # Only squeeze if size is 1 or dynamic
            if s != 1 and s != -1:
                logger.warning(
                    f"[TensorRT] squeeze on dim {i} with size {s} != 1, not squeezing"
                )
                output_shape.append(s)
                kept_dims.append(i)
            # else: skip this dimension (squeeze it)
        else:
            output_shape.append(s)
            kept_dims.append(i)

    num_dynamic = sum(1 for d in output_shape if d < 0)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for squeeze {node.name}")

    if num_dynamic <= 1 and not input_has_dynamic_dims(input_trt):
        layer.reshape_dims = trt.Dims(output_shape)
    elif num_dynamic <= 1:
        # TRT handles at most one -1 in reshape natively.
        layer.reshape_dims = trt.Dims(output_shape)
    else:
        # Multiple dynamic dims: build shape tensor from kept dims of runtime shape.
        shape_layer = network.add_shape(input_trt)
        shape_layer.name = f"squeeze_shape_{node.name}"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"squeeze_shape_i32_{node.name}"
        shape_trt = shape_i32.get_output(0)

        components: List[trt.ITensor] = []
        for out_i, inp_i in enumerate(kept_dims):
            if output_shape[out_i] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([output_shape[out_i]], dtype=np.int32))
                )
                c.name = f"squeeze_c{out_i}_{node.name}"
                components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([inp_i], dtype=np.int32))
                )
                idx_c.name = f"squeeze_idx{out_i}_{node.name}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"squeeze_g{out_i}_{node.name}"
                components.append(g.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"squeeze_outshape_{node.name}"
        layer.set_input(1, shape_cat.get_output(0))

    layer.name = f"squeeze_{node.name}"

    logger.debug(
        f"[TensorRT] Created squeeze layer: {layer.name}, "
        f"dims={dims_to_squeeze}, output_shape={output_shape}"
    )

    return layer.get_output(0)


@converter("aten.unsqueeze.default", "aten.unsqueeze_copy.default", validator_fn=validate_squeeze_unsqueeze, supports_dynamic_shapes=True)
def convert_unsqueeze(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> trt.ITensor:
    """
    Convert PyTorch unsqueeze to TensorRT shuffle layer.

    Inserts a dimension of size 1 at the specified position.
    Supports dynamic shapes via the shape tensor API when multiple
    dimensions are dynamic.

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

    # Resolve shape: SymInts become -1 for TRT dynamic dims
    input_shape = resolve_shape(get_node_shape(input_node) or tuple(input_trt.shape))
    ndim = len(input_shape)

    # Handle negative dimension (for unsqueeze, target ndim is ndim + 1)
    dim = _get_positive_dim(dim, ndim + 1)

    # Build output shape with new dimension of size 1
    output_shape = input_shape[:dim] + [1] + input_shape[dim:]

    num_dynamic = sum(1 for d in output_shape if d < 0)

    layer = network.add_shuffle(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create shuffle layer for unsqueeze {node.name}")

    if num_dynamic <= 1 and not input_has_dynamic_dims(input_trt):
        layer.reshape_dims = trt.Dims(output_shape)
    elif num_dynamic <= 1:
        # TRT handles at most one -1 in reshape natively.
        layer.reshape_dims = trt.Dims(output_shape)
    else:
        # Multiple dynamic dims: build shape tensor from runtime input shape.
        shape_layer = network.add_shape(input_trt)
        shape_layer.name = f"unsqueeze_shape_{node.name}"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"unsqueeze_shape_i32_{node.name}"
        shape_trt = shape_i32.get_output(0)

        components: List[trt.ITensor] = []
        input_idx = 0
        for i in range(len(output_shape)):
            if i == dim:
                # Inserted dimension of size 1
                c = network.add_constant(
                    [1], trt.Weights(np.array([1], dtype=np.int32))
                )
                c.name = f"unsqueeze_one_{node.name}"
                components.append(c.get_output(0))
            else:
                if output_shape[i] >= 0:
                    c = network.add_constant(
                        [1], trt.Weights(np.array([output_shape[i]], dtype=np.int32))
                    )
                    c.name = f"unsqueeze_c{i}_{node.name}"
                    components.append(c.get_output(0))
                else:
                    idx_c = network.add_constant(
                        [1], trt.Weights(np.array([input_idx], dtype=np.int32))
                    )
                    idx_c.name = f"unsqueeze_idx{i}_{node.name}"
                    g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                    g.name = f"unsqueeze_g{i}_{node.name}"
                    components.append(g.get_output(0))
                input_idx += 1

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"unsqueeze_outshape_{node.name}"
        layer.set_input(1, shape_cat.get_output(0))

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


@converter("aten.select.int", "aten.select_copy.int", validator_fn=validate_select, supports_dynamic_shapes=True)
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
    Supports dynamic shapes via the shape tensor API when multiple
    dimensions are dynamic.

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

    # Resolve shape: SymInts become -1 for TRT dynamic dims
    input_shape = resolve_shape(get_node_shape(input_node) or tuple(input_trt.shape))
    ndim = len(input_shape)

    # Handle negative dimension
    dim = _get_positive_dim(dim, ndim)

    # Handle negative index
    if isinstance(index, int) and index < 0:
        if input_shape[dim] > 0:
            index = input_shape[dim] + index

    # Build start, shape, stride for slice operation
    start = [0] * ndim
    start[dim] = index

    # Shape: same as input except the selected dim has size 1
    slice_shape = list(input_shape)
    slice_shape[dim] = 1

    # Stride: 1 for all dims
    stride = [1] * ndim

    has_dynamic_slice = any(d < 0 for d in slice_shape)

    if not has_dynamic_slice:
        # Static: all dims concrete
        layer = network.add_slice(
            input_trt,
            start=trt.Dims(start),
            shape=trt.Dims(slice_shape),
            stride=trt.Dims(stride),
        )
    else:
        # Dynamic: build shape tensor for slice output size
        shape_layer = network.add_shape(input_trt)
        shape_layer.name = f"select_shape_{node.name}"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"select_shape_i32_{node.name}"
        shape_trt = shape_i32.get_output(0)

        components: List[trt.ITensor] = []
        for i in range(ndim):
            if i == dim:
                # Selected dim: always size 1
                c = network.add_constant(
                    [1], trt.Weights(np.array([1], dtype=np.int32))
                )
                c.name = f"select_one_{node.name}"
                components.append(c.get_output(0))
            elif slice_shape[i] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([slice_shape[i]], dtype=np.int32))
                )
                c.name = f"select_sc{i}_{node.name}"
                components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                idx_c.name = f"select_sidx{i}_{node.name}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"select_sg{i}_{node.name}"
                components.append(g.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"select_slice_shape_{node.name}"

        # Placeholder shape overridden by set_input(2, ...)
        layer = network.add_slice(
            input_trt,
            start=trt.Dims(start),
            shape=trt.Dims([1] * ndim),
            stride=trt.Dims(stride),
        )
        layer.set_input(2, shape_cat.get_output(0))

    if layer is None:
        raise RuntimeError(f"Failed to create slice layer for select {node.name}")

    layer.name = f"select_slice_{node.name}"
    slice_output = layer.get_output(0)

    # Squeeze the selected dim: output = slice_shape without dim
    output_shape = [s for i, s in enumerate(slice_shape) if i != dim]

    num_dynamic = sum(1 for d in output_shape if d < 0)

    squeeze_layer = network.add_shuffle(slice_output)
    if squeeze_layer is None:
        raise RuntimeError(
            f"Failed to create shuffle layer for select squeeze {node.name}"
        )

    if num_dynamic <= 1:
        squeeze_layer.reshape_dims = trt.Dims(output_shape)
    else:
        # Multiple dynamic dims: build shape tensor for squeezed output
        sq_shape_layer = network.add_shape(slice_output)
        sq_shape_layer.name = f"select_sqshape_{node.name}"
        sq_shape_i32 = network.add_cast(sq_shape_layer.get_output(0), trt.int32)
        sq_shape_i32.name = f"select_sqshape_i32_{node.name}"
        sq_shape_trt = sq_shape_i32.get_output(0)

        sq_components: List[trt.ITensor] = []
        out_idx = 0
        for i in range(ndim):
            if i == dim:
                continue  # Skip the squeezed dim
            if output_shape[out_idx] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([output_shape[out_idx]], dtype=np.int32))
                )
                c.name = f"select_sqc{out_idx}_{node.name}"
                sq_components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                idx_c.name = f"select_sqidx{out_idx}_{node.name}"
                g = network.add_gather(sq_shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"select_sqg{out_idx}_{node.name}"
                sq_components.append(g.get_output(0))
            out_idx += 1

        sq_shape_cat = network.add_concatenation(sq_components)
        sq_shape_cat.axis = 0
        sq_shape_cat.name = f"select_sqoutshape_{node.name}"
        squeeze_layer.set_input(1, sq_shape_cat.get_output(0))

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
