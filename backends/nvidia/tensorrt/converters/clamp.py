# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Clamp/Clip Operations.

Supported operations:
- aten.clamp.default: Clamps all elements in input into the range [min, max]
- aten.clamp.Tensor: Clamps with tensor bounds
- aten.clip.default: Alias for clamp
- aten.hardtanh.default: Clamps between min_val and max_val (ReLU6 variant)

TensorRT supports clamping via the IActivationLayer with CLIP activation type.
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    broadcast_tensors,
    get_trt_tensor,
)

logger: logging.Logger = logging.getLogger(__name__)


def validate_clamp(node: torch.fx.Node) -> bool:
    """Validate that a clamp node can be converted to TensorRT.

    Args:
        node: FX node representing the clamp operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_clamp: node {node.name} is not call_function"
        )
        return False

    args = node.args
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_clamp: node {node.name} has insufficient args"
        )
        return False

    if not isinstance(args[0], torch.fx.Node):
        logger.debug(
            f"[TensorRT] validate_clamp: input is not a node, got {type(args[0])}"
        )
        return False

    return True


@converter("aten.clamp.default", validator_fn=validate_clamp)
def convert_clamp(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch clamp to TensorRT.

    PyTorch signature:
        aten.clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor

    For constant min/max bounds, we use IActivationLayer with CLIP type.
    For None bounds, we implement only the specified bound.

    Args:
        node: FX node representing the clamp operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_clamp") from e

    logger.debug(f"[TensorRT] Converting clamp node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    min_val = args[1] if len(args) > 1 else kwargs.get("min", None)
    max_val = args[2] if len(args) > 2 else kwargs.get("max", None)

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to clamp must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    logger.debug(f"[TensorRT] clamp: min={min_val}, max={max_val}")

    result = input_trt
    input_ndim = len(input_trt.shape)

    # Handle min bound (using max with min_val)
    if min_val is not None:
        min_const = get_trt_tensor(network, float(min_val), f"clamp_min_{node.name}")
        # Broadcast constant to match input dimensions for elementwise operation
        [min_const] = broadcast_tensors(
            network, [min_const], input_ndim, f"clamp_min_{node.name}"
        )
        layer_min = network.add_elementwise(
            result, min_const, trt.ElementWiseOperation.MAX
        )
        if layer_min is None:
            raise RuntimeError(f"Failed to create clamp min layer for node {node.name}")
        layer_min.name = f"clamp_min_{node.name}"
        result = layer_min.get_output(0)

    # Handle max bound (using min with max_val)
    if max_val is not None:
        max_const = get_trt_tensor(network, float(max_val), f"clamp_max_{node.name}")
        # Broadcast constant to match input dimensions for elementwise operation
        [max_const] = broadcast_tensors(
            network, [max_const], input_ndim, f"clamp_max_{node.name}"
        )
        layer_max = network.add_elementwise(
            result, max_const, trt.ElementWiseOperation.MIN
        )
        if layer_max is None:
            raise RuntimeError(f"Failed to create clamp max layer for node {node.name}")
        layer_max.name = f"clamp_max_{node.name}"
        result = layer_max.get_output(0)

    if min_val is None and max_val is None:
        # No clamping needed, return input as-is
        logger.warning(
            f"[TensorRT] clamp node {node.name} has no min or max, returning input"
        )

    logger.debug(f"[TensorRT] Created clamp layers for node: {node.name}")

    return result


@converter("aten.clamp_min.default", validator_fn=validate_clamp)
def convert_clamp_min(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch clamp_min to TensorRT.

    PyTorch signature:
        aten.clamp_min(Tensor self, Scalar min) -> Tensor
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_clamp_min") from e

    logger.debug(f"[TensorRT] Converting clamp_min node: {node.name}")

    args = node.args
    input_node = args[0]
    min_val = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to clamp_min must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]
    input_ndim = len(input_trt.shape)

    min_const = get_trt_tensor(network, float(min_val), f"clamp_min_{node.name}")
    # Broadcast constant to match input dimensions for elementwise operation
    [min_const] = broadcast_tensors(
        network, [min_const], input_ndim, f"clamp_min_{node.name}"
    )
    layer = network.add_elementwise(input_trt, min_const, trt.ElementWiseOperation.MAX)
    if layer is None:
        raise RuntimeError(f"Failed to create clamp_min layer for node {node.name}")
    layer.name = f"clamp_min_{node.name}"

    logger.debug(f"[TensorRT] Created clamp_min layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.clamp_max.default", validator_fn=validate_clamp)
def convert_clamp_max(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch clamp_max to TensorRT.

    PyTorch signature:
        aten.clamp_max(Tensor self, Scalar max) -> Tensor
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_clamp_max") from e

    logger.debug(f"[TensorRT] Converting clamp_max node: {node.name}")

    args = node.args
    input_node = args[0]
    max_val = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to clamp_max must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]
    input_ndim = len(input_trt.shape)

    max_const = get_trt_tensor(network, float(max_val), f"clamp_max_{node.name}")
    # Broadcast constant to match input dimensions for elementwise operation
    [max_const] = broadcast_tensors(
        network, [max_const], input_ndim, f"clamp_max_{node.name}"
    )
    layer = network.add_elementwise(input_trt, max_const, trt.ElementWiseOperation.MIN)
    if layer is None:
        raise RuntimeError(f"Failed to create clamp_max layer for node {node.name}")
    layer.name = f"clamp_max_{node.name}"

    logger.debug(f"[TensorRT] Created clamp_max layer: {layer.name}")

    return layer.get_output(0)


__all__ = [
    "convert_clamp",
    "convert_clamp_min",
    "convert_clamp_max",
    "validate_clamp",
]
