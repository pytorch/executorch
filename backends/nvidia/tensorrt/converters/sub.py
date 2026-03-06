# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converter for element-wise subtraction operations."""

import logging
from typing import Any, Dict, Optional

import tensorrt as trt
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    broadcast_tensors,
    get_node_dtype,
    get_trt_tensor,
    promote_and_cast_tensors,
    set_layer_name,
)


logger: logging.Logger = logging.getLogger(__name__)


def _get_elementwise_input(
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    arg: Any,
    name: str,
    dtype: Optional[torch.dtype],
) -> trt.ITensor:
    """Get TensorRT tensor for an elementwise operation input.

    Handles:
    - FX nodes already in input_map
    - FX nodes that are lifted buffers/parameters (placeholder nodes with b_ or p_ prefix)
    - Scalar values
    """
    if isinstance(arg, torch.fx.Node):
        if arg in input_map:
            return input_map[arg]

        # Handle lifted buffers and parameters that aren't in input_map
        # These are placeholder nodes with names starting with b_ (buffers) or p_ (parameters)
        # or get_attr nodes. We need to create constants from their metadata values.
        if arg.op == "placeholder" or arg.op == "get_attr":
            if "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                logger.debug(f"[TensorRT] Creating constant for lifted buffer/parameter: {arg.name}")
                trt_tensor = get_trt_tensor(network, arg.meta["val"], f"const_{arg.name}", dtype)
                input_map[arg] = trt_tensor  # Cache for future use
                return trt_tensor

        raise ValueError(
            f"Input node '{arg.name}' not found in input_map and could not be created as constant. "
            f"Node op: {arg.op}, target: {arg.target}. "
            f"Available nodes: {list(n.name for n in input_map.keys())}"
        )
    return get_trt_tensor(network, arg, name, dtype)


def _get_input_ndim(arg: Any, input_map: Dict[torch.fx.Node, Any]) -> int:
    """Get the number of dimensions for an elementwise input argument."""
    if isinstance(arg, torch.fx.Node):
        if "val" in arg.meta and hasattr(arg.meta["val"], "shape"):
            return len(arg.meta["val"].shape)
        if arg in input_map:
            trt_tensor = input_map[arg]
            shape = trt_tensor.shape
            if shape is not None:
                return len(shape)
    return 0


@converter("aten.sub.Tensor", "aten.sub_.Tensor")
def convert_sub(
    node: torch.fx.Node,
    network: trt.INetworkDefinition,
    input_map: Dict[torch.fx.Node, Any],
    ctx: Any = None,
) -> trt.ITensor:
    """Convert aten.sub.Tensor to TensorRT ElementWise SUB.

    Handles tensor - tensor, tensor - scalar, and scalar - tensor cases.
    The alpha parameter (x - alpha * y) is validated to be 1.
    Includes type promotion for mixed-type operands.
    """
    if len(node.args) < 2:
        raise ValueError(
            f"aten.sub requires at least 2 arguments, got {len(node.args)}"
        )

    lhs_arg = node.args[0]
    rhs_arg = node.args[1]

    alpha = node.args[2] if len(node.args) > 2 else node.kwargs.get("alpha", 1)
    if alpha != 1:
        raise ValueError(
            f"aten.sub.Tensor with alpha != 1 is not supported, got alpha={alpha}"
        )

    dtype = get_node_dtype(node)

    lhs = _get_elementwise_input(network, input_map, lhs_arg, f"sub_lhs_{node.name}", dtype)
    rhs = _get_elementwise_input(network, input_map, rhs_arg, f"sub_rhs_{node.name}", dtype)

    # Type promotion: ensure both operands have compatible types
    lhs, rhs = promote_and_cast_tensors(network, lhs, rhs, f"sub_{node.name}")

    # Get target ndim for broadcasting
    lhs_ndim = _get_input_ndim(lhs_arg, input_map)
    rhs_ndim = _get_input_ndim(rhs_arg, input_map)
    target_ndim = max(lhs_ndim, rhs_ndim)

    if target_ndim == 0 and "val" in node.meta and hasattr(node.meta["val"], "shape"):
        target_ndim = len(node.meta["val"].shape)
    if target_ndim == 0:
        target_ndim = 1

    lhs, rhs = broadcast_tensors(network, [lhs, rhs], target_ndim, f"sub_{node.name}")

    layer = network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.SUB)
    if layer is None:
        raise RuntimeError(f"Failed to create elementwise SUB layer for {node.name}")
    set_layer_name(layer, node, "sub")

    return layer.get_output(0)
