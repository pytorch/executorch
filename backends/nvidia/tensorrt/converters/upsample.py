# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""TensorRT Converters for Upsample Operations.

Supported operations:
- aten.upsample_bilinear2d.vec: 2D bilinear upsampling
- aten.upsample_nearest2d.vec: 2D nearest neighbor upsampling

TensorRT uses IResizeLayer for upsampling operations with configurable
interpolation modes (LINEAR for bilinear, NEAREST for nearest neighbor).
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger: logging.Logger = logging.getLogger(__name__)


def validate_upsample(node: torch.fx.Node) -> bool:
    """Validate that an upsample node can be converted to TensorRT.

    Args:
        node: FX node representing the upsample operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_upsample: node {node.name} is not call_function"
        )
        return False

    args = node.args
    if len(args) < 1:
        logger.debug(
            f"[TensorRT] validate_upsample: node {node.name} has insufficient args"
        )
        return False

    if not isinstance(args[0], torch.fx.Node):
        logger.debug(
            f"[TensorRT] validate_upsample: input is not a node, got {type(args[0])}"
        )
        return False

    return True


@converter("aten.upsample_bilinear2d.vec", validator_fn=validate_upsample)
def convert_upsample_bilinear2d(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch upsample_bilinear2d to TensorRT resize layer.

    PyTorch signature:
        aten.upsample_bilinear2d.vec(
            Tensor input,
            SymInt[]? output_size,
            bool align_corners,
            float[]? scale_factors
        ) -> Tensor

    Args:
        node: FX node representing the upsample_bilinear2d operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "TensorRT is required for convert_upsample_bilinear2d"
        ) from e

    logger.debug(f"[TensorRT] Converting upsample_bilinear2d node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
    align_corners = args[2] if len(args) > 2 else kwargs.get("align_corners", False)
    scale_factors = args[3] if len(args) > 3 else kwargs.get("scale_factors")

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(
            f"Input to upsample_bilinear2d must be a node, got {type(input_node)}"
        )

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    logger.debug(
        f"[TensorRT] upsample_bilinear2d: output_size={output_size}, "
        f"align_corners={align_corners}, scale_factors={scale_factors}"
    )

    layer = network.add_resize(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create resize layer for node {node.name}")

    # TensorRT 10.x uses InterpolationMode (ResizeMode was deprecated)
    layer.resize_mode = trt.InterpolationMode.LINEAR

    if align_corners:
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        )
    else:
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    if output_size is not None:
        input_shape = input_trt.shape
        output_shape = tuple(input_shape[:2]) + tuple(output_size)
        layer.shape = output_shape
        logger.debug(f"[TensorRT] Set resize output shape: {output_shape}")
    elif scale_factors is not None:
        full_scales = [1.0, 1.0] + list(scale_factors)
        layer.scales = full_scales
        logger.debug(f"[TensorRT] Set resize scales: {full_scales}")
    else:
        raise ValueError(
            f"upsample_bilinear2d node {node.name} must have "
            f"either output_size or scale_factors"
        )

    layer.name = f"upsample_bilinear2d_{node.name}"

    logger.debug(f"[TensorRT] Created upsample_bilinear2d layer: {layer.name}")

    return layer.get_output(0)


@converter("aten.upsample_nearest2d.vec", validator_fn=validate_upsample)
def convert_upsample_nearest2d(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch upsample_nearest2d to TensorRT resize layer.

    PyTorch signature:
        aten.upsample_nearest2d.vec(
            Tensor input,
            SymInt[]? output_size,
            float[]? scale_factors
        ) -> Tensor

    Args:
        node: FX node representing the upsample_nearest2d operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_upsample_nearest2d") from e

    logger.debug(f"[TensorRT] Converting upsample_nearest2d node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
    scale_factors = args[2] if len(args) > 2 else kwargs.get("scale_factors")

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(
            f"Input to upsample_nearest2d must be a node, got {type(input_node)}"
        )

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    logger.debug(
        f"[TensorRT] upsample_nearest2d: output_size={output_size}, "
        f"scale_factors={scale_factors}"
    )

    layer = network.add_resize(input_trt)
    if layer is None:
        raise RuntimeError(f"Failed to create resize layer for node {node.name}")

    # TensorRT 10.x uses InterpolationMode (ResizeMode was deprecated)
    layer.resize_mode = trt.InterpolationMode.NEAREST

    if output_size is not None:
        input_shape = input_trt.shape
        output_shape = tuple(input_shape[:2]) + tuple(output_size)
        layer.shape = output_shape
        logger.debug(f"[TensorRT] Set resize output shape: {output_shape}")
    elif scale_factors is not None:
        full_scales = [1.0, 1.0] + list(scale_factors)
        layer.scales = full_scales
        logger.debug(f"[TensorRT] Set resize scales: {full_scales}")
    else:
        raise ValueError(
            f"upsample_nearest2d node {node.name} must have "
            f"either output_size or scale_factors"
        )

    layer.name = f"upsample_nearest2d_{node.name}"

    logger.debug(f"[TensorRT] Created upsample_nearest2d layer: {layer.name}")

    return layer.get_output(0)


__all__ = [
    "convert_upsample_bilinear2d",
    "convert_upsample_nearest2d",
    "validate_upsample",
]
