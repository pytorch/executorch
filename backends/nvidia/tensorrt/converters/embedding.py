# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Embedding Operations.

Supported operations:
- aten.embedding.default: Lookup embeddings from a weight matrix using indices

TensorRT implements embeddings via IGatherLayer which indexes into a tensor.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import create_constant

logger: logging.Logger = logging.getLogger(__name__)


def validate_embedding(node: torch.fx.Node) -> bool:
    """Validate that an embedding node can be converted to TensorRT.

    Args:
        node: FX node representing the embedding operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        logger.debug(
            f"[TensorRT] validate_embedding: node {node.name} is not call_function"
        )
        return False

    args = node.args
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_embedding: node {node.name} has insufficient args"
        )
        return False

    # First arg is weight, second is indices
    if not isinstance(args[1], torch.fx.Node):
        logger.debug(
            f"[TensorRT] validate_embedding: indices is not a node, got {type(args[1])}"
        )
        return False

    return True


@converter("aten.embedding.default", validator_fn=validate_embedding)
def convert_embedding(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch embedding to TensorRT.

    PyTorch signature:
        aten.embedding(
            Tensor weight,
            Tensor indices,
            int padding_idx=-1,
            bool scale_grad_by_freq=False,
            bool sparse=False
        ) -> Tensor

    TensorRT uses IGatherLayer to implement embedding lookup.
    The gather operation indexes into the weight matrix using the indices.

    Args:
        node: FX node representing the embedding operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: Optional edge program for accessing weights/constants.

    Returns:
        TensorRT output tensor.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_embedding") from e

    logger.debug(f"[TensorRT] Converting embedding node: {node.name}")

    args = node.args
    weight_node = args[0]  # Embedding weight matrix
    indices_node = args[1]  # Indices to look up

    # Get weight tensor
    weight_trt = None
    if isinstance(weight_node, torch.fx.Node):
        if weight_node in input_map:
            weight_trt = input_map[weight_node]
        elif hasattr(weight_node, "target") and edge_program is not None:
            # Try to get weight from graph state dict
            weight_name = weight_node.target
            if hasattr(edge_program, "graph_module"):
                gm = edge_program.graph_module
                if hasattr(gm, "state_dict"):
                    state_dict = gm.state_dict()
                    if weight_name in state_dict:
                        weight_data = state_dict[weight_name].detach().cpu().numpy()
                        weight_trt = create_constant(
                            network, weight_data, f"embedding_weight_{node.name}"
                        )

    if weight_trt is None:
        raise ValueError(f"Could not get embedding weight for node {node.name}")

    # Get indices tensor
    if not isinstance(indices_node, torch.fx.Node):
        raise ValueError(
            f"Indices for embedding must be a node, got {type(indices_node)}"
        )

    if indices_node not in input_map:
        raise ValueError(f"Indices node {indices_node.name} not found in input_map")

    indices_trt = input_map[indices_node]

    weight_shape = weight_trt.shape
    indices_shape = indices_trt.shape

    logger.debug(
        f"[TensorRT] embedding: weight_shape={weight_shape}, indices_shape={indices_shape}"
    )

    # Use gather layer to implement embedding lookup
    # Gather along axis 0 (the vocabulary dimension)
    gather_layer = network.add_gather(weight_trt, indices_trt, axis=0)
    if gather_layer is None:
        raise RuntimeError(f"Failed to create gather layer for node {node.name}")

    gather_layer.name = f"embedding_gather_{node.name}"

    logger.debug(f"[TensorRT] Created embedding gather layer: {gather_layer.name}")

    return gather_layer.get_output(0)


@converter("aten.embedding_renorm_.default", validator_fn=validate_embedding)
def convert_embedding_renorm(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch embedding_renorm_ to TensorRT.

    This is an in-place operation for renormalizing embeddings.
    In TensorRT, we treat this as a no-op since it's typically used during training.
    """
    logger.warning(
        f"[TensorRT] embedding_renorm_ is a training operation, treating as no-op"
    )
    # Return the weight tensor unchanged
    weight_node = node.args[0]
    if isinstance(weight_node, torch.fx.Node) and weight_node in input_map:
        return input_map[weight_node]
    raise ValueError(f"Could not get embedding weight for node {node.name}")


__all__ = [
    "convert_embedding",
    "convert_embedding_renorm",
    "validate_embedding",
]
