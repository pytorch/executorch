# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for PixelShuffle and Depth2Space Operations.

Supported operations:
- aten.pixel_shuffle.default: Rearranges elements in a tensor
  from (*, C*r^2, H, W) to (*, C, H*r, W*r), where r is an upscale factor.

TensorRT uses IShuffleLayer to implement pixel shuffle via reshape and permute.
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger: logging.Logger = logging.getLogger(__name__)


def validate_pixel_shuffle(node: torch.fx.Node) -> bool:
    """Validate that a pixel_shuffle node can be converted to TensorRT.

    Args:
        node: FX node representing the pixel_shuffle operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_pixel_shuffle: node {node.name} is not call_function"
        )
        return False

    args = node.args
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_pixel_shuffle: node {node.name} has insufficient args"
        )
        return False

    if not isinstance(args[0], torch.fx.Node):
        logger.debug(
            f"[TensorRT] validate_pixel_shuffle: input is not a node, got {type(args[0])}"
        )
        return False

    return True


@converter("aten.pixel_shuffle.default", validator_fn=validate_pixel_shuffle)
def convert_pixel_shuffle(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch pixel_shuffle to TensorRT shuffle layers.

    PyTorch signature:
        aten.pixel_shuffle(Tensor input, int upscale_factor) -> Tensor

    Pixel shuffle rearranges elements from (N, C*r^2, H, W) to (N, C, H*r, W*r).
    This is equivalent to depth-to-space operation.

    Implementation uses reshape + permute + reshape:
    1. Reshape: (N, C*r^2, H, W) -> (N, C, r, r, H, W)
    2. Permute: (N, C, r, r, H, W) -> (N, C, H, r, W, r)
    3. Reshape: (N, C, H, r, W, r) -> (N, C, H*r, W*r)

    Args:
        node: FX node representing the pixel_shuffle operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_pixel_shuffle") from e

    logger.debug(f"[TensorRT] Converting pixel_shuffle node: {node.name}")

    args = node.args
    input_node = args[0]
    upscale_factor = args[1]

    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(
            f"Input to pixel_shuffle must be a node, got {type(input_node)}"
        )

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]
    input_shape = input_trt.shape  # (N, C*r^2, H, W)

    logger.debug(
        f"[TensorRT] pixel_shuffle: input_shape={input_shape}, upscale_factor={upscale_factor}"
    )

    # Get dimensions
    if len(input_shape) != 4:
        raise ValueError(
            f"pixel_shuffle requires 4D input, got {len(input_shape)}D: {input_shape}"
        )

    batch_size = input_shape[0]
    channels_r2 = input_shape[1]  # C * r^2
    height = input_shape[2]
    width = input_shape[3]

    r = upscale_factor
    out_channels = channels_r2 // (r * r)

    # Step 1: Reshape to (N, C, r, r, H, W)
    reshape1 = network.add_shuffle(input_trt)
    if reshape1 is None:
        raise RuntimeError(f"Failed to create reshape layer for node {node.name}")

    reshape1.reshape_dims = (batch_size, out_channels, r, r, height, width)
    reshape1.name = f"pixel_shuffle_reshape1_{node.name}"

    # Step 2: Permute to (N, C, H, r, W, r)
    permute = network.add_shuffle(reshape1.get_output(0))
    if permute is None:
        raise RuntimeError(f"Failed to create permute layer for node {node.name}")

    permute.second_transpose = trt.Permutation([0, 1, 4, 2, 5, 3])
    permute.name = f"pixel_shuffle_permute_{node.name}"

    # Step 3: Reshape to (N, C, H*r, W*r)
    reshape2 = network.add_shuffle(permute.get_output(0))
    if reshape2 is None:
        raise RuntimeError(f"Failed to create reshape2 layer for node {node.name}")

    reshape2.reshape_dims = (batch_size, out_channels, height * r, width * r)
    reshape2.name = f"pixel_shuffle_reshape2_{node.name}"

    return reshape2.get_output(0)


__all__ = [
    "convert_pixel_shuffle",
    "validate_pixel_shuffle",
]
