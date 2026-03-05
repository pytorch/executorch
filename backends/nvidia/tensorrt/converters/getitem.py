# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converter for getitem Operations.

This module provides converters for Python getitem operations, which are used
to extract elements from tuples/lists (e.g., extracting the first output from
batch_norm which returns multiple values).

Supported operations:
- _operator.getitem
- operator.getitem
- getitem
"""

import logging
from typing import Any, Dict, Optional

import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter

logger: logging.Logger = logging.getLogger(__name__)


def validate_getitem(node: torch.fx.Node) -> bool:
    """
    Validate that a getitem node can be converted to TensorRT.

    Args:
        node: FX node representing the getitem operation.

    Returns:
        True if the node can be converted, False otherwise.
    """
    if node.op != "call_function":
        return False

    args = node.args
    # getitem takes 2 args: container and index
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_getitem: node {node.name} has insufficient args"
        )
        return False

    return True


@converter(
    "_operator.getitem",
    "operator.getitem",
    "getitem",
    validator_fn=validate_getitem,
)
def convert_getitem(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Any] = None,
) -> Any:
    """
    Convert Python getitem operation to pass through the correct tensor.

    This is used when operations like batch_norm return multiple values
    (output, mean, var) and we need to extract just one of them.

    Args:
        node: FX node representing the getitem operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: ExportedProgram for extracting weights.

    Returns:
        TensorRT output tensor (the extracted item).
    """
    logger.debug(f"[TensorRT] Converting getitem node: {node.name}")

    args = node.args
    container = args[0]
    index = args[1]

    if not isinstance(container, torch.fx.Node):
        raise ValueError(f"Container to getitem must be a node, got {type(container)}")

    if container not in input_map:
        raise ValueError(f"Container node {container.name} not found in input_map")

    # The container should already be mapped to a TensorRT tensor
    # (For batch_norm, we already return just the first output)
    result = input_map[container]

    # If the result is a tuple/list (which it shouldn't be in TensorRT),
    # extract the indexed element
    if isinstance(result, (list, tuple)):
        result = result[index]

    logger.debug(
        f"[TensorRT] getitem: extracting index {index} from {container.name}"
    )

    return result


__all__ = ["convert_getitem", "validate_getitem"]
