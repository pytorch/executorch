# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TensorRT Converter for Linear (fully connected) Operations.

This module provides converters for PyTorch linear operations to TensorRT
fully connected layers.

Supported operations:
- aten.linear.default: Fully connected layer (y = x @ weight.T + bias)

The linear operation is implemented as:
1. Matrix multiplication: x @ weight.T
2. Add bias (if present)
"""

import logging
from typing import Any, Dict, Optional, Union

import torch

from executorch.backends.nvidia.tensorrt.converter_registry import converter

from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export.exported_program import ExportedProgram

logger: logging.Logger = logging.getLogger(__name__)


def _is_get_attr_node(node: Any) -> bool:
    """Check if node is a get_attr node."""
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def _get_param_tensor(
    exp_prog: Optional[ExportedProgram],
    node: Any,
) -> Optional[torch.Tensor]:
    """Extract a constant tensor from an ExportedProgram."""
    if node is None:
        return None

    if isinstance(node, torch.Tensor):
        return node

    if not isinstance(node, torch.fx.Node):
        return None

    if exp_prog is not None:
        if is_param(exp_prog, node):
            return get_param(exp_prog, node)
        elif is_buffer(exp_prog, node):
            return get_buffer(exp_prog, node)
        elif is_lifted_tensor_constant(exp_prog, node):
            return get_lifted_tensor_constant(exp_prog, node)

    if _is_get_attr_node(node):
        if exp_prog is not None:
            try:
                target = node.target
                if isinstance(target, str):
                    return getattr(exp_prog.graph_module, target)
            except AttributeError:
                pass
        try:
            if hasattr(node, "graph") and hasattr(node.graph, "owning_module"):
                target = node.target
                if isinstance(target, str):
                    return getattr(node.graph.owning_module, target)
        except AttributeError:
            pass

    return None


def validate_linear(node: torch.fx.Node) -> bool:
    """Validate that a linear node can be converted to TensorRT."""
    if node.op != "call_function":
        return False

    args = node.args
    # linear requires at least input and weight
    if len(args) < 2:
        logger.debug(
            f"[TensorRT] validate_linear: node {node.name} has insufficient args"
        )
        return False

    return True


@converter("aten.linear.default", validator_fn=validate_linear, needs_edge_program=True)
def convert_linear(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Union[ExportedProgram, torch.fx.GraphModule]] = None,
    ctx: Any = None,
) -> Any:  # trt.ITensor
    """
    Convert PyTorch linear operation to TensorRT layers.

    Linear is defined as: y = x @ weight.T + bias

    Args:
        node: FX node representing the linear operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        edge_program: ExportedProgram for extracting weights.

    Returns:
        TensorRT output tensor.

    Raises:
        ImportError: If TensorRT is not installed.
        ValueError: If required inputs are missing.
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "TensorRT is required for convert_linear"
        ) from e

    logger.debug(f"[TensorRT] Converting linear node: {node.name}")

    args = node.args
    kwargs = node.kwargs

    # Extract arguments
    # linear(input, weight, bias=None)
    input_node = args[0]
    weight_node = args[1]
    bias_node = args[2] if len(args) > 2 else kwargs.get("bias", None)

    # Validate input
    if not isinstance(input_node, torch.fx.Node):
        raise ValueError(f"Input to linear must be a node, got {type(input_node)}")

    if input_node not in input_map:
        raise ValueError(f"Input node {input_node.name} not found in input_map")

    input_trt = input_map[input_node]

    # Get weight tensor
    exp_prog = edge_program if isinstance(edge_program, ExportedProgram) else None
    weight_tensor = _get_param_tensor(exp_prog, weight_node)
    if weight_tensor is None:
        raise ValueError(
            f"Could not extract weight tensor for linear node {node.name}. "
            "Weight must be a constant tensor."
        )

    # Weight shape for linear: (out_features, in_features)
    weight_np = weight_tensor.detach().cpu().numpy().astype(np.float32)
    out_features = weight_np.shape[0]
    in_features = weight_np.shape[1]

    logger.debug(
        f"[TensorRT] linear: in_features={in_features}, out_features={out_features}"
    )

    # Create weight as constant tensor with transposed shape for matmul
    # Linear: y = x @ weight.T, so we need weight.T for matmul
    weight_transposed = np.ascontiguousarray(weight_np.T)  # Shape: (in_features, out_features)

    # Store weight to prevent GC before engine build completes
    if not hasattr(convert_linear, '_weight_storage'):
        convert_linear._weight_storage = []
    convert_linear._weight_storage.append(weight_transposed)

    weight_const = network.add_constant(
        trt.Dims(weight_transposed.shape),
        trt.Weights(weight_transposed)
    )
    if weight_const is None:
        raise RuntimeError(f"Failed to create weight constant for linear {node.name}")
    weight_const.name = f"linear_weight_{node.name}"
    weight_trt = weight_const.get_output(0)

    # Matrix multiplication: input @ weight.T
    mm_layer = network.add_matrix_multiply(
        input_trt, trt.MatrixOperation.NONE,
        weight_trt, trt.MatrixOperation.NONE
    )
    if mm_layer is None:
        raise RuntimeError(f"Failed to create matmul layer for linear {node.name}")
    mm_layer.name = f"linear_mm_{node.name}"
    output = mm_layer.get_output(0)

    # Add bias if present
    if bias_node is not None:
        bias_tensor = _get_param_tensor(exp_prog, bias_node)
        if bias_tensor is not None:
            bias_np = bias_tensor.detach().cpu().numpy().astype(np.float32)

            # Reshape bias for broadcasting
            # For 2D input [batch, out_features], bias is [out_features]
            # Need to reshape to [1, out_features] for proper broadcasting
            output_dims = len(output.shape)
            bias_shape = [1] * (output_dims - 1) + [out_features]
            bias_reshaped = bias_np.reshape(bias_shape)

            # Store bias to prevent GC before engine build completes
            convert_linear._weight_storage.append(bias_reshaped)

            bias_const = network.add_constant(
                trt.Dims(bias_reshaped.shape),
                trt.Weights(bias_reshaped)
            )
            if bias_const is None:
                raise RuntimeError(
                    f"Failed to create bias constant for linear {node.name}"
                )
            bias_const.name = f"linear_bias_const_{node.name}"
            bias_trt = bias_const.get_output(0)

            add_layer = network.add_elementwise(
                output, bias_trt, trt.ElementWiseOperation.SUM
            )
            if add_layer is None:
                raise RuntimeError(
                    f"Failed to create bias add layer for linear {node.name}"
                )
            add_layer.name = f"linear_bias_{node.name}"
            output = add_layer.get_output(0)

            logger.debug(f"[TensorRT] Added bias to linear layer {node.name}")

    logger.debug(f"[TensorRT] Created linear layer: {node.name}")

    return output


def clear_weight_storage() -> None:
    """Clear weight storage to free memory after engine build."""
    if hasattr(convert_linear, '_weight_storage'):
        convert_linear._weight_storage.clear()


__all__ = ["clear_weight_storage", "convert_linear", "validate_linear"]
