# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""TensorRT Converter for Batch Normalization Operations."""

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


def validate_batch_norm(node: torch.fx.Node) -> bool:
    """Validate that a batch norm node can be converted to TensorRT."""
    if node.op != "call_function":
        return False
    if len(node.args) < 5:
        return False
    return True


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

    # Fallback for get_attr nodes
    if isinstance(node, torch.fx.Node) and node.op == "get_attr":
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


@converter(
    "aten._native_batch_norm_legit.default",
    "aten._native_batch_norm_legit_no_training.default",
    "aten.batch_norm.default",
    validator_fn=validate_batch_norm,
    needs_edge_program=True,
)
def convert_batch_norm(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Union[ExportedProgram, torch.fx.GraphModule]] = None,
    ctx: Any = None,
) -> Any:
    """Convert PyTorch batch norm to TensorRT scale layer.

    Implements batch normalization as a fused scale operation:
        output = scale * input + shift
    where:
        scale = gamma / sqrt(running_var + eps)
        shift = beta - running_mean * scale
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_batch_norm.") from e

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    weight_node = args[1] if len(args) > 1 else kwargs.get("weight", None)
    bias_node = args[2] if len(args) > 2 else kwargs.get("bias", None)
    running_mean_node = args[3] if len(args) > 3 else kwargs.get("running_mean", None)
    running_var_node = args[4] if len(args) > 4 else kwargs.get("running_var", None)

    target_str = str(node.target)
    if "no_training" in target_str:
        eps = args[6] if len(args) > 6 else kwargs.get("eps", 1e-5)
    else:
        eps = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)

    if not isinstance(input_node, torch.fx.Node) or input_node not in input_map:
        raise ValueError(f"Input node {input_node} not found in input_map")

    input_trt = input_map[input_node]

    exp_prog = edge_program if isinstance(edge_program, ExportedProgram) else None
    weight_tensor = _get_param_tensor(exp_prog, weight_node)
    bias_tensor = _get_param_tensor(exp_prog, bias_node)
    running_mean_tensor = _get_param_tensor(exp_prog, running_mean_node)
    running_var_tensor = _get_param_tensor(exp_prog, running_var_node)

    if running_mean_tensor is None:
        raise ValueError(f"running_mean must be a constant tensor for {node.name}")
    if running_var_tensor is None:
        raise ValueError(f"running_var must be a constant tensor for {node.name}")

    mean_np = running_mean_tensor.detach().cpu().numpy().astype(np.float32)
    var_np = running_var_tensor.detach().cpu().numpy().astype(np.float32)

    if weight_tensor is not None:
        gamma_np = weight_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        gamma_np = np.ones_like(mean_np, dtype=np.float32)

    if bias_tensor is not None:
        beta_np = bias_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        beta_np = np.zeros_like(mean_np, dtype=np.float32)

    num_channels = mean_np.shape[0]

    # Fuse BN into scale layer: y = scale * x + shift
    fused_scale = np.ascontiguousarray(
        (gamma_np / np.sqrt(var_np + eps)).astype(np.float32)
    )
    fused_shift = np.ascontiguousarray(
        (beta_np - mean_np * fused_scale).astype(np.float32)
    )
    power_weights = np.ascontiguousarray(np.ones(num_channels, dtype=np.float32))

    # Store arrays to prevent GC before engine build completes
    if not hasattr(convert_batch_norm, '_weight_storage'):
        convert_batch_norm._weight_storage = []
    convert_batch_norm._weight_storage.extend([fused_scale, fused_shift, power_weights])

    scale_layer = network.add_scale(
        input_trt,
        trt.ScaleMode.CHANNEL,
        shift=trt.Weights(fused_shift),
        scale=trt.Weights(fused_scale),
        power=trt.Weights(power_weights),
    )
    if scale_layer is None:
        raise RuntimeError(f"Failed to create Scale layer for {node.name}")
    scale_layer.name = f"bn_scale_{node.name}"
    return scale_layer.get_output(0)


def clear_weight_storage() -> None:
    """Clear weight storage to free memory after engine build."""
    if hasattr(convert_batch_norm, '_weight_storage'):
        convert_batch_norm._weight_storage.clear()


__all__ = [
    "clear_weight_storage",
    "convert_batch_norm",
    "validate_batch_norm",
]
