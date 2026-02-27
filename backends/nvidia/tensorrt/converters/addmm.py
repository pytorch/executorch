# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for addmm (fused add + matrix multiply) operations."""

from typing import Any, Dict

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import set_layer_name


@converter("aten.addmm.default")
def convert_addmm(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.addmm.default to TensorRT MatrixMultiply + ElementWise Add.

    Performs: output = beta * bias + alpha * (mat1 @ mat2)
    Default alpha=1, beta=1, so: output = bias + mat1 @ mat2

    This is the core operation for torch.nn.Linear layers.
    For nn.Linear, the operation is: output = input @ weight.T + bias
    which becomes addmm(bias, input, weight.T)

    Args:
        node: FX node representing the addmm operation.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        ctx: Optional conversion context.

    Returns:
        TensorRT tensor representing the result.

    Raises:
        ValueError: If required inputs are missing or alpha/beta != 1.
    """
    # Validate args
    if len(node.args) < 3:
        raise ValueError(
            f"aten.addmm requires at least 3 arguments (bias, mat1, mat2), "
            f"got {len(node.args)}"
        )

    # Check for alpha/beta scaling factors (not supported)
    beta = node.kwargs.get("beta", 1)
    alpha = node.kwargs.get("alpha", 1)
    if beta != 1 or alpha != 1:
        raise ValueError(
            f"aten.addmm with alpha={alpha}, beta={beta} is not supported. "
            f"Only alpha=1, beta=1 is supported."
        )

    bias_arg = node.args[0]
    mat1_arg = node.args[1]
    mat2_arg = node.args[2]

    # Validate inputs exist in input_map
    for arg, name in [(bias_arg, "bias"), (mat1_arg, "mat1"), (mat2_arg, "mat2")]:
        if isinstance(arg, torch.fx.Node) and arg not in input_map:
            raise ValueError(
                f"Input '{name}' (node '{arg.name}') not found in input_map for "
                f"addmm node '{node.name}'"
            )

    bias_tensor = input_map[bias_arg]
    mat1 = input_map[mat1_arg]
    mat2 = input_map[mat2_arg]

    # Perform matrix multiplication: mat1 @ mat2
    mm_layer = network.add_matrix_multiply(
        mat1, trt.MatrixOperation.NONE, mat2, trt.MatrixOperation.NONE
    )
    if mm_layer is None:
        raise RuntimeError(
            f"Failed to create matrix multiply layer for addmm node '{node.name}'"
        )
    set_layer_name(mm_layer, node, "addmm_mm")
    mm_output = mm_layer.get_output(0)

    # The bias may need to be reshaped to broadcast properly.
    # For nn.Linear, bias has shape [out_features] (1D), but mm_output
    # has shape [batch, out_features] (2D). We need to reshape bias to [1, out_features]
    # so TensorRT can broadcast the addition correctly.
    
    # Get dimensions from node metadata for reliability (TensorRT shapes may have -1)
    if "val" in node.meta and hasattr(node.meta["val"], "shape"):
        mm_dims = len(node.meta["val"].shape)
    else:
        # Fall back to TensorRT shape
        mm_dims = len(mm_output.shape)
    
    # Get bias dimensions from metadata or tensor
    if isinstance(bias_arg, torch.fx.Node) and "val" in bias_arg.meta and hasattr(bias_arg.meta["val"], "shape"):
        bias_dims = len(bias_arg.meta["val"].shape)
    else:
        bias_dims = len(bias_tensor.shape)

    if bias_dims < mm_dims:
        # Need to unsqueeze the bias to match mm_output dimensions
        # For 1D bias [N] -> 2D [1, N]
        if isinstance(bias_arg, torch.fx.Node) and "val" in bias_arg.meta:
            bias_shape = list(bias_arg.meta["val"].shape)
        else:
            bias_shape = list(bias_tensor.shape)
        target_shape = [1] * (mm_dims - bias_dims) + bias_shape
        shuffle_layer = network.add_shuffle(bias_tensor)
        shuffle_layer.reshape_dims = trt.Dims(target_shape)
        set_layer_name(shuffle_layer, node, "addmm_bias_reshape")
        bias_tensor = shuffle_layer.get_output(0)

    # Add the bias: bias + (mat1 @ mat2)
    add_layer = network.add_elementwise(
        mm_output, bias_tensor, trt.ElementWiseOperation.SUM
    )
    if add_layer is None:
        raise RuntimeError(
            f"Failed to create elementwise SUM layer for addmm node '{node.name}'"
        )
    set_layer_name(add_layer, node, "addmm_add")

    return add_layer.get_output(0)
