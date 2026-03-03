# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Layer Normalization Operations.

Supported operations:
- aten.native_layer_norm.default: PyTorch native layer normalization
- aten.layer_norm.default: PyTorch layer normalization

LayerNorm normalizes across the last N dimensions (normalized_shape).
Formula: y = (x - E[x]) / sqrt(Var[x] + eps) * weight + bias

TensorRT supports layer normalization via INormalizationLayer (TRT 8.6+).
For older versions, we implement it using element-wise operations.

Note: native_layer_norm returns (output, mean, rstd), but TensorRT only computes
the output. Mean and rstd are returned as zero-filled placeholder tensors with
correct shapes. If downstream code requires actual mean/rstd values, a manual
implementation would be needed.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    create_constant,
    get_node_shape,
    get_trt_tensor,
)

logger: logging.Logger = logging.getLogger(__name__)


def validate_layer_norm(node: torch.fx.Node) -> bool:
    """Validate that a layer_norm node can be converted to TensorRT.

    Args:
        node: FX node representing the layer_norm operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_layer_norm: node {node.name} is not call_function"
        )
        return False

    args = node.args
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_layer_norm: node {node.name} has insufficient args"
        )
        return False

    if not isinstance(args[0], torch.fx.Node):
        logger.debug(
            f"[TensorRT] validate_layer_norm: input is not a node, got {type(args[0])}"
        )
        return False

    return True


def _get_axes_for_normalized_shape(
    input_ndim: int, normalized_shape: Union[List[int], Tuple[int, ...]]
) -> int:
    """Calculate the axes bitmask for normalization.

    Args:
        input_ndim: Number of dimensions in input tensor
        normalized_shape: Shape of dimensions to normalize over

    Returns:
        Bitmask indicating which axes to normalize
    """
    num_normalized_dims = len(normalized_shape)
    axes = 0
    for i in range(input_ndim - num_normalized_dims, input_ndim):
        axes |= 1 << i
    return axes


def _reshape_scale_bias_for_norm(
    network: trt.INetworkDefinition,
    tensor: Optional[trt.ITensor],
    input_ndim: int,
    normalized_shape: List[int],
    name: str,
) -> Optional[trt.ITensor]:
    """Reshape 1D scale/bias tensor to match input ndims for INormalizationLayer.

    TensorRT's INormalizationLayer requires scale and bias tensors to have the
    same number of dimensions as the input tensor.

    Args:
        network: TensorRT network definition.
        tensor: Scale or bias tensor to reshape (can be None).
        input_ndim: Number of dimensions in the input tensor.
        normalized_shape: Shape of dimensions being normalized.
        name: Name prefix for the reshape layer.

    Returns:
        Reshaped tensor, or None if input tensor was None.

    Raises:
        RuntimeError: If shuffle layer creation fails.
    """
    if tensor is None:
        return None

    tensor_shape = tensor.shape
    tensor_ndim = len(tensor_shape)

    if tensor_ndim == input_ndim:
        return tensor  # Already correct shape

    # Compute target shape: [1, 1, ..., normalized_shape]
    # E.g., for input shape [batch, seq, features] with normalized_shape=[features],
    # scale should be [1, 1, features]
    num_leading_ones = input_ndim - len(normalized_shape)
    target_shape = [1] * num_leading_ones + list(normalized_shape)

    # Reshape using shuffle layer
    shuffle_layer = network.add_shuffle(tensor)
    if shuffle_layer is None:
        raise RuntimeError(
            f"Failed to create shuffle layer for {name} reshape. "
            f"Input shape: {tensor_shape}, target shape: {target_shape}"
        )
    shuffle_layer.reshape_dims = trt.Dims(target_shape)
    shuffle_layer.name = f"{name}_reshape"
    return shuffle_layer.get_output(0)


def _get_layer_norm_param(
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    param_node: Optional[Any],
    edge_program: Optional[Any],
    name: str,
) -> Optional[trt.ITensor]:
    """Get weight or bias tensor for layer normalization.

    Attempts to get the parameter from input_map first, then from edge_program
    state dict if available.

    Args:
        network: TensorRT network definition (for creating constants).
        input_map: Mapping from FX nodes to TensorRT tensors.
        param_node: FX node for weight/bias (can be None).
        edge_program: Edge program for accessing state dict.
        name: Name for the parameter tensor.

    Returns:
        TensorRT tensor for the parameter, or None if not found.
    """
    if param_node is None:
        return None

    if not isinstance(param_node, torch.fx.Node):
        return None

    # Try to get from input_map first
    if param_node in input_map:
        return input_map[param_node]

    # Try to get from edge_program state dict
    if hasattr(param_node, "target") and edge_program is not None:
        param_name = param_node.target
        if hasattr(edge_program, "graph_module"):
            gm = edge_program.graph_module
            if hasattr(gm, "state_dict"):
                state_dict = gm.state_dict()
                if param_name in state_dict:
                    param_data = state_dict[param_name].detach().cpu().numpy()
                    return create_constant(network, param_data, name)

    return None


@converter("aten.native_layer_norm.default", validator_fn=validate_layer_norm)
def convert_native_layer_norm(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    """Convert PyTorch native_layer_norm to TensorRT.

    PyTorch signature:
        aten.native_layer_norm(
            Tensor input,
            SymInt[] normalized_shape,
            Tensor? weight,
            Tensor? bias,
            float eps
        ) -> (Tensor, Tensor, Tensor)

    Returns tuple of (output, mean, rstd).
    Note: TensorRT only computes output; mean and rstd are zero-filled placeholders.

    Args:
        node: FX node representing the native_layer_norm operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        Tuple of (output, mean, rstd) TensorRT tensors.

    Raises:
        ValueError: If input node is invalid or not found in input_map.
        RuntimeError: If TensorRT layer creation fails.
    """
    logger.debug(f"[TensorRT] Converting native_layer_norm node: {node.name}")

    args = node.args
    input_node = args[0]
    normalized_shape = list(args[1]) if len(args) > 1 else []
    weight_node = args[2] if len(args) > 2 else None
    bias_node = args[3] if len(args) > 3 else None
    eps = args[4] if len(args) > 4 else 1e-5

    # Validate input
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(
            f"Input to native_layer_norm must be a node, got {type(input_node)}"
        )

    if input_node not in input_map:
        raise ValueError(
            f"Input node '{input_node.name}' not found in input_map for "
            f"native_layer_norm node '{node.name}'"
        )

    input_trt = input_map[input_node]

    # Get shape from node metadata for reliability
    input_shape = tuple(
        get_node_shape(input_node) or tuple(input_trt.shape)
    )
    input_ndim = len(input_shape)

    logger.debug(
        f"[TensorRT] native_layer_norm: input_shape={input_shape}, "
        f"normalized_shape={normalized_shape}, eps={eps}"
    )

    # Get weight and bias tensors
    weight_trt = _get_layer_norm_param(
        network, input_map, weight_node, edge_program, f"ln_weight_{node.name}"
    )
    bias_trt = _get_layer_norm_param(
        network, input_map, bias_node, edge_program, f"ln_bias_{node.name}"
    )

    # Calculate axes for normalization
    axes = _get_axes_for_normalized_shape(input_ndim, normalized_shape)

    # Reshape weight and bias for INormalizationLayer
    weight_trt_reshaped = _reshape_scale_bias_for_norm(
        network, weight_trt, input_ndim, normalized_shape, f"ln_weight_{node.name}"
    )
    bias_trt_reshaped = _reshape_scale_bias_for_norm(
        network, bias_trt, input_ndim, normalized_shape, f"ln_bias_{node.name}"
    )

    # Try to use TensorRT's native normalization layer (TRT 8.6+)
    if hasattr(network, "add_normalization"):
        layer = network.add_normalization(
            input_trt, weight_trt_reshaped, bias_trt_reshaped, axes
        )
        if layer is not None:
            layer.epsilon = eps
            layer.name = f"layer_norm_{node.name}"
            output = layer.get_output(0)

            # Create placeholder tensors for mean and rstd
            mean_placeholder = _create_placeholder_tensor(
                network, input_shape, input_ndim, normalized_shape, f"mean_{node.name}"
            )
            rstd_placeholder = _create_placeholder_tensor(
                network, input_shape, input_ndim, normalized_shape, f"rstd_{node.name}"
            )

            logger.debug(f"[TensorRT] Created native layer_norm: {layer.name}")
            return (output, mean_placeholder, rstd_placeholder)

    # Fallback: Implement layer norm using element-wise operations
    logger.debug(
        f"[TensorRT] add_normalization not available, using manual implementation"
    )
    output = _manual_layer_norm(
        network, input_trt, normalized_shape, weight_trt, bias_trt, eps, node.name
    )

    # Create placeholder tensors for mean and rstd
    mean_placeholder = _create_placeholder_tensor(
        network, input_shape, input_ndim, normalized_shape, f"mean_{node.name}"
    )
    rstd_placeholder = _create_placeholder_tensor(
        network, input_shape, input_ndim, normalized_shape, f"rstd_{node.name}"
    )

    return (output, mean_placeholder, rstd_placeholder)


def _create_placeholder_tensor(
    network: Any,
    input_shape: Tuple[int, ...],
    input_ndim: int,
    normalized_shape: List[int],
    name: str,
) -> Any:
    """Create a placeholder tensor for mean/rstd outputs.

    The shape is input_shape with normalized dimensions reduced to 1.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    num_normalized_dims = len(normalized_shape)
    # Shape for mean/rstd is input shape with normalized dims as 1
    reduced_shape = list(input_shape)
    for i in range(input_ndim - num_normalized_dims, input_ndim):
        if i < len(reduced_shape):
            reduced_shape[i] = 1

    # Create a constant tensor of zeros
    zero_data = np.zeros(reduced_shape, dtype=np.float32)
    return create_constant(network, zero_data, name)


def _manual_layer_norm(
    network: Any,
    input_trt: Any,
    normalized_shape: List[int],
    weight_trt: Optional[Any],
    bias_trt: Optional[Any],
    eps: float,
    node_name: str,
) -> Any:
    """Implement layer normalization using element-wise operations.

    LayerNorm formula:
        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    Args:
        network: TensorRT network definition
        input_trt: Input tensor
        normalized_shape: Dimensions to normalize over
        weight_trt: Optional scale tensor
        bias_trt: Optional bias tensor
        eps: Epsilon for numerical stability
        node_name: Name for layer naming

    Returns:
        Output tensor after layer normalization
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required") from e

    input_shape = input_trt.shape
    input_ndim = len(input_shape)
    num_normalized_dims = len(normalized_shape)

    # Calculate axes for reduction (last num_normalized_dims dimensions)
    axes = 0
    for i in range(input_ndim - num_normalized_dims, input_ndim):
        axes |= 1 << i

    # Step 1: Calculate mean
    mean_layer = network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes, True)
    if mean_layer is None:
        raise RuntimeError(f"Failed to create mean layer for {node_name}")
    mean_layer.name = f"ln_mean_{node_name}"
    mean = mean_layer.get_output(0)

    # Step 2: Calculate (x - mean)
    sub_mean = network.add_elementwise(
        input_trt, mean, trt.ElementWiseOperation.SUB
    )
    if sub_mean is None:
        raise RuntimeError(f"Failed to create sub_mean layer for {node_name}")
    sub_mean.name = f"ln_sub_mean_{node_name}"
    centered = sub_mean.get_output(0)

    # Step 3: Calculate (x - mean)^2
    sq_layer = network.add_elementwise(
        centered, centered, trt.ElementWiseOperation.PROD
    )
    if sq_layer is None:
        raise RuntimeError(f"Failed to create sq layer for {node_name}")
    sq_layer.name = f"ln_sq_{node_name}"
    squared = sq_layer.get_output(0)

    # Step 4: Calculate variance = mean((x - mean)^2)
    var_layer = network.add_reduce(squared, trt.ReduceOperation.AVG, axes, True)
    if var_layer is None:
        raise RuntimeError(f"Failed to create var layer for {node_name}")
    var_layer.name = f"ln_var_{node_name}"
    variance = var_layer.get_output(0)

    # Step 5: Add epsilon to variance
    eps_const = get_trt_tensor(network, eps, f"ln_eps_{node_name}")
    var_eps = network.add_elementwise(
        variance, eps_const, trt.ElementWiseOperation.SUM
    )
    if var_eps is None:
        raise RuntimeError(f"Failed to create var_eps layer for {node_name}")
    var_eps.name = f"ln_var_eps_{node_name}"

    # Step 6: Calculate sqrt(var + eps)
    sqrt_layer = network.add_unary(var_eps.get_output(0), trt.UnaryOperation.SQRT)
    if sqrt_layer is None:
        raise RuntimeError(f"Failed to create sqrt layer for {node_name}")
    sqrt_layer.name = f"ln_sqrt_{node_name}"
    std = sqrt_layer.get_output(0)

    # Step 7: Calculate (x - mean) / sqrt(var + eps)
    div_layer = network.add_elementwise(centered, std, trt.ElementWiseOperation.DIV)
    if div_layer is None:
        raise RuntimeError(f"Failed to create div layer for {node_name}")
    div_layer.name = f"ln_div_{node_name}"
    normalized = div_layer.get_output(0)

    # Step 8: Apply weight (scale) if provided
    if weight_trt is not None:
        scale_layer = network.add_elementwise(
            normalized, weight_trt, trt.ElementWiseOperation.PROD
        )
        if scale_layer is None:
            raise RuntimeError(f"Failed to create scale layer for {node_name}")
        scale_layer.name = f"ln_scale_{node_name}"
        normalized = scale_layer.get_output(0)

    # Step 9: Apply bias if provided
    if bias_trt is not None:
        bias_layer = network.add_elementwise(
            normalized, bias_trt, trt.ElementWiseOperation.SUM
        )
        if bias_layer is None:
            raise RuntimeError(f"Failed to create bias layer for {node_name}")
        bias_layer.name = f"ln_bias_{node_name}"
        normalized = bias_layer.get_output(0)

    logger.debug(f"[TensorRT] Created manual layer_norm layers for: {node_name}")

    return normalized


@converter("aten.layer_norm.default", validator_fn=validate_layer_norm)
def convert_layer_norm(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch layer_norm to TensorRT.

    PyTorch signature:
        aten.layer_norm(
            Tensor input,
            SymInt[] normalized_shape,
            Tensor? weight=None,
            Tensor? bias=None,
            float eps=1e-5,
            bool cudnn_enable=True
        ) -> Tensor

    This is a wrapper around native_layer_norm that returns only the output.
    """
    result = convert_native_layer_norm(node, network, input_map, edge_program)
    if isinstance(result, tuple):
        return result[0]  # Return only the output tensor
    return result


__all__ = [
    "convert_native_layer_norm",
    "convert_layer_norm",
    "validate_layer_norm",
]
