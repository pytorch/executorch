# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converters for Pooling Operations.

This module provides converters for PyTorch pooling operations to TensorRT
pooling layers.

Supported operations:
- aten.avg_pool2d.default: 2D average pooling
- aten.max_pool2d.default: 2D max pooling (when indices are not used)
- aten.max_pool2d_with_indices.default: 2D max pooling with indices output
- aten.adaptive_avg_pool2d.default: Adaptive 2D average pooling (for SE blocks)

Notes:
- TensorRT doesn't support dilation != 1 for pooling
- TensorRT doesn't support divisor_override for avg_pool
- Adaptive pooling requires static spatial dimensions
- max_pool_with_indices: indices output is NOT supported, only values are returned
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger: logging.Logger = logging.getLogger(__name__)


def validate_avg_pool2d(node: torch.fx.Node) -> bool:
    """
    Validate that an avg_pool2d node can be converted to TensorRT.

    Args:
        node: FX node representing the avg_pool2d operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_avg_pool2d: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Minimum args: input, kernel_size
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_avg_pool2d: node {node.name} has insufficient args"
        )
        return False

    # Check for divisor_override - not supported by TensorRT
    if len(args) > 6 and args[6] is not None:
        logger.debug(
            f"[TensorRT] validate_avg_pool2d: divisor_override not supported"
        )
        return False

    return True


def validate_max_pool2d(node: torch.fx.Node) -> bool:
    """
    Validate that a max_pool2d node can be converted to TensorRT.

    Args:
        node: FX node representing the max_pool2d operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_max_pool2d: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Minimum args: input, kernel_size
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_max_pool2d: node {node.name} has insufficient args"
        )
        return False

    # Check for dilation - only dilation=1 is supported
    if len(args) > 4:
        dilation = args[4]
        if dilation is not None:
            if isinstance(dilation, (list, tuple)):
                if any(d != 1 for d in dilation):
                    logger.debug(
                        f"[TensorRT] validate_max_pool2d: dilation != 1 not supported"
                    )
                    return False
            elif dilation != 1:
                logger.debug(
                    f"[TensorRT] validate_max_pool2d: dilation != 1 not supported"
                )
                return False

    return True


def validate_adaptive_avg_pool2d(node: torch.fx.Node) -> bool:
    """
    Validate that an adaptive_avg_pool2d node can be converted to TensorRT.

    Args:
        node: FX node representing the adaptive_avg_pool2d operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_adaptive_avg_pool2d: node {node.name} is not call_function"
        )
        return False

    args = node.args
    # Args: input, output_size
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_adaptive_avg_pool2d: node {node.name} has insufficient args"
        )
        return False

    return True


def _extend_to_tuple(value: Any, length: int) -> Tuple[int, ...]:
    """
    Extend a value to a tuple of given length.

    Args:
        value: An int, tuple, or list.
        length: Desired tuple length.

    Returns:
        Tuple of integers with the specified length.
    """
    if value is None:
        return (0,) * length
    if isinstance(value, int):
        return (value,) * length
    if isinstance(value, (list, tuple)):
        if len(value) == length:
            return tuple(value)
        if len(value) == 1:
            return (value[0],) * length
    return tuple(value)


@converter("aten.avg_pool2d.default", validator_fn=validate_avg_pool2d)
def convert_avg_pool2d(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch avg_pool2d to TensorRT pooling layer.

    Args:
        node: FX node representing the avg_pool2d operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.

    Raises:
        ImportError: If TensorRT is not installed.
        ValueError: If required inputs are missing.
        RuntimeError: If TensorRT layer creation fails.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "TensorRT is required for convert_avg_pool2d"
        ) from e

    logger.debug(f"[TensorRT] Converting avg_pool2d node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    # Extract arguments
    # avg_pool2d(input, kernel_size, stride=[], padding=0, ceil_mode=False,
    #            count_include_pad=True, divisor_override=None)
    input_node = args[0]
    kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
    stride = args[2] if len(args) > 2 else kwargs.get("stride", [])
    padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
    ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
    count_include_pad = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)

    # Validate input
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to avg_pool2d must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Normalize parameters to tuples
    kernel_size = _extend_to_tuple(kernel_size, 2)

    # Default stride to kernel_size if empty or None
    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride = kernel_size
    else:
        stride = _extend_to_tuple(stride, 2)

    padding = _extend_to_tuple(padding, 2)

    logger.debug(
        f"[TensorRT] avg_pool2d parameters: kernel={kernel_size}, "
        f"stride={stride}, padding={padding}, ceil_mode={ceil_mode}, "
        f"count_include_pad={count_include_pad}"
    )

    # Create pooling layer using add_pooling_nd
    layer = network.add_pooling_nd(
        input=input_trt,
        type=trt.PoolingType.AVERAGE,
        window_size=trt.Dims(kernel_size),
    )
    if layer is None:
        raise RuntimeError(
            f"Failed to create avg_pool2d layer for node {node.name}"
        )

    layer.stride_nd = trt.Dims(stride)
    layer.padding_nd = trt.Dims(padding)
    layer.name = f"avg_pool2d_{node.name}"

    # Handle count_include_pad
    # TensorRT: average_count_excludes_padding = True means padding is NOT included
    layer.average_count_excludes_padding = not count_include_pad

    # Handle ceil_mode
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    logger.debug(f"[TensorRT] Created avg_pool2d layer: {layer.name}")

    return layer.get_output(0)


@converter(
    "aten.max_pool2d.default",
    "aten.max_pool2d_with_indices.default",
    validator_fn=validate_max_pool2d,
)
def convert_max_pool2d(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch max_pool2d to TensorRT pooling layer.

    Note: For max_pool2d_with_indices, only the values output is returned.
    The indices are NOT supported by TensorRT. If indices are actually used
    in the model, conversion will fail during graph execution.

    Args:
        node: FX node representing the max_pool2d operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor (only values, no indices).

    Raises:
        ImportError: If TensorRT is not installed.
        ValueError: If required inputs are missing.
        RuntimeError: If TensorRT layer creation fails or dilation != 1.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "TensorRT is required for convert_max_pool2d"
        ) from e

    logger.debug(f"[TensorRT] Converting max_pool2d node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    # Extract arguments
    # max_pool2d(input, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False)
    input_node = args[0]
    kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
    stride = args[2] if len(args) > 2 else kwargs.get("stride", [])
    padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
    dilation = args[4] if len(args) > 4 else kwargs.get("dilation", 1)
    ceil_mode = args[5] if len(args) > 5 else kwargs.get("ceil_mode", False)

    # Validate input
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to max_pool2d must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Normalize parameters to tuples
    kernel_size = _extend_to_tuple(kernel_size, 2)

    # Default stride to kernel_size if empty or None
    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride = kernel_size
    else:
        stride = _extend_to_tuple(stride, 2)

    padding = _extend_to_tuple(padding, 2)
    dilation = _extend_to_tuple(dilation, 2)

    # Validate dilation
    if dilation != (1, 1):
        raise RuntimeError(
            f"TensorRT only supports dilation=(1, 1) for max_pool2d, got {dilation}"
        )

    logger.debug(
        f"[TensorRT] max_pool2d parameters: kernel={kernel_size}, "
        f"stride={stride}, padding={padding}, ceil_mode={ceil_mode}"
    )

    # Create pooling layer using add_pooling_nd
    layer = network.add_pooling_nd(
        input=input_trt,
        type=trt.PoolingType.MAX,
        window_size=trt.Dims(kernel_size),
    )
    if layer is None:
        raise RuntimeError(
            f"Failed to create max_pool2d layer for node {node.name}"
        )

    layer.stride_nd = trt.Dims(stride)
    layer.padding_nd = trt.Dims(padding)
    layer.name = f"max_pool2d_{node.name}"

    # Handle ceil_mode
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    logger.debug(f"[TensorRT] Created max_pool2d layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.adaptive_avg_pool2d.default", validator_fn=validate_adaptive_avg_pool2d)
def convert_adaptive_avg_pool2d(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch adaptive_avg_pool2d to TensorRT pooling layer.

    Adaptive pooling computes kernel_size and stride from input and output shapes:
    - stride = input_size // output_size
    - kernel_size = input_size - (output_size - 1) * stride

    This is critical for Squeeze-and-Excitation (SE) blocks in MobileNetV3.

    Limitations:
    - Input spatial dimensions (H, W) must be static (not -1)
    - Input size must be evenly divisible by output size

    Args:
        node: FX node representing the adaptive_avg_pool2d operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.

    Returns:
        TensorRT output tensor.

    Raises:
        ImportError: If TensorRT is not installed.
        ValueError: If required inputs are missing.
        RuntimeError: If TensorRT layer creation fails or constraints not met.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "TensorRT is required for convert_adaptive_avg_pool2d"
        ) from e

    logger.debug(f"[TensorRT] Converting adaptive_avg_pool2d node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    # Extract arguments
    # adaptive_avg_pool2d(input, output_size)
    input_node = args[0]
    output_size = args[1] if len(args) > 1 else kwargs.get("output_size")

    # Validate input
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to adaptive_avg_pool2d must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Normalize output_size to tuple
    output_size = _extend_to_tuple(output_size, 2)

    # Get input shape (expecting NCHW format)
    input_shape = input_trt.shape
    if len(input_shape) != 4:
        raise RuntimeError(
            f"adaptive_avg_pool2d expects 4D input (NCHW), got shape {input_shape}"
        )

    # Extract spatial dimensions (H, W)
    input_h = input_shape[2]
    input_w = input_shape[3]

    # Check for dynamic dimensions
    if input_h == -1 or input_w == -1:
        raise RuntimeError(
            f"adaptive_avg_pool2d doesn't support dynamic spatial dimensions. "
            f"Input shape: {input_shape}. H and W must be static."
        )

    output_h, output_w = output_size

    # Validate divisibility
    if input_h % output_h != 0:
        raise RuntimeError(
            f"Input height ({input_h}) must be divisible by output height ({output_h})"
        )
    if input_w % output_w != 0:
        raise RuntimeError(
            f"Input width ({input_w}) must be divisible by output width ({output_w})"
        )

    # Calculate kernel_size and stride
    # Formula: stride = input_size // output_size
    #          kernel = input_size - (output_size - 1) * stride
    stride_h = input_h // output_h
    stride_w = input_w // output_w

    kernel_h = input_h - (output_h - 1) * stride_h
    kernel_w = input_w - (output_w - 1) * stride_w

    kernel_size = (kernel_h, kernel_w)
    stride = (stride_h, stride_w)

    logger.debug(
        f"[TensorRT] adaptive_avg_pool2d: output_size={output_size}, "
        f"input_spatial=({input_h}, {input_w}), computed kernel={kernel_size}, stride={stride}"
    )

    # Create pooling layer
    layer = network.add_pooling_nd(
        input=input_trt,
        type=trt.PoolingType.AVERAGE,
        window_size=trt.Dims(kernel_size),
    )
    if layer is None:
        raise RuntimeError(
            f"Failed to create adaptive_avg_pool2d layer for node {node.name}"
        )

    layer.stride_nd = trt.Dims(stride)
    layer.name = f"adaptive_avg_pool2d_{node.name}"

    logger.debug(f"[TensorRT] Created adaptive_avg_pool2d layer: {layer.name}")

    return layer.get_output(0)


__all__ = [
    "convert_avg_pool2d",
    "convert_max_pool2d",
    "convert_adaptive_avg_pool2d",
    "validate_avg_pool2d",
    "validate_max_pool2d",
    "validate_adaptive_avg_pool2d",
]
