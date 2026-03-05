# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT Converters for Scaled Dot-Product Attention (SDPA).

Supported operations:
- aten.scaled_dot_product_attention: Core attention mechanism for transformers

SDPA computes: softmax(Q @ K^T / sqrt(d_k)) @ V

For TensorRT, we implement this using:
1. Matrix multiply for Q @ K^T
2. Scale by 1/sqrt(d_k)
3. Optional attention mask application
4. Softmax
5. Matrix multiply with V
"""

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    create_constant,
    get_trt_tensor,
)

logger: logging.Logger = logging.getLogger(__name__)


def validate_sdpa(node: torch.fx.Node) -> bool:
    """Validate that an SDPA node can be converted to TensorRT."""
    if node.op != "call_function":
        return False

    args = node.args
    if len(args) < 3:
        return False

    for i in range(3):
        if not isinstance(args[i], torch.fx.Node):
            return False

    return True


@converter("aten.scaled_dot_product_attention.default", validator_fn=validate_sdpa)
def convert_scaled_dot_product_attention(
    node: torch.fx.Node,
    network: Any,  # trt.INetworkDefinition
    input_map: Dict[torch.fx.Node, Any],  # Dict[Node, trt.ITensor]
    edge_program: Optional[Any] = None,
) -> Any:  # trt.ITensor
    """Convert PyTorch scaled_dot_product_attention to TensorRT.

    SDPA formula: softmax(Q @ K^T / sqrt(d_k) + mask) @ V
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_sdpa") from e

    args = node.args
    kwargs = node.kwargs

    query_node = args[0]
    key_node = args[1]
    value_node = args[2]
    attn_mask_node = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)
    is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
    scale = args[6] if len(args) > 6 else kwargs.get("scale", None)

    query_trt = input_map[query_node]
    key_trt = input_map[key_node]
    value_trt = input_map[value_node]

    query_shape = query_trt.shape
    d_k = query_shape[-1]

    # Calculate scale factor
    if scale is not None:
        scale_factor = float(scale)
    elif d_k > 0:
        scale_factor = 1.0 / math.sqrt(float(d_k))
    else:
        query_meta_shape = None
        if isinstance(query_node, torch.fx.Node) and "val" in query_node.meta:
            val = query_node.meta["val"]
            if hasattr(val, "shape"):
                query_meta_shape = val.shape
        if query_meta_shape is not None and len(query_meta_shape) > 0:
            d_k_static = query_meta_shape[-1]
            scale_factor = 1.0 / math.sqrt(float(d_k_static)) if d_k_static > 0 else 1.0
        else:
            raise RuntimeError(
                f"Cannot determine head dimension for SDPA node {node.name}."
            )

    # Step 1: Q @ K^T
    qk_layer = network.add_matrix_multiply(
        query_trt, trt.MatrixOperation.NONE,
        key_trt, trt.MatrixOperation.TRANSPOSE,
    )
    qk_layer.name = f"sdpa_qk_{node.name}"
    qk = qk_layer.get_output(0)

    # Step 2: Scale by 1/sqrt(d_k)
    scale_const = get_trt_tensor(
        network, scale_factor, f"sdpa_scale_{node.name}", dtype=torch.float32
    )
    scaled_qk_layer = network.add_elementwise(
        qk, scale_const, trt.ElementWiseOperation.PROD
    )
    scaled_qk_layer.name = f"sdpa_scale_{node.name}"
    scaled_qk = scaled_qk_layer.get_output(0)

    # Step 3: Apply attention mask if provided
    if attn_mask_node is not None and isinstance(attn_mask_node, torch.fx.Node):
        if attn_mask_node in input_map:
            attn_mask_trt = input_map[attn_mask_node]
            mask_layer = network.add_elementwise(
                scaled_qk, attn_mask_trt, trt.ElementWiseOperation.SUM
            )
            mask_layer.name = f"sdpa_mask_{node.name}"
            scaled_qk = mask_layer.get_output(0)

    # Step 4: Handle causal masking
    if is_causal:
        seq_len = query_shape[-2] if len(query_shape) >= 2 else -1
        if seq_len > 0:
            causal_mask = np.triu(
                np.full((seq_len, seq_len), float("-inf"), dtype=np.float32), k=1
            )
            causal_mask_trt = create_constant(
                network, causal_mask, f"sdpa_causal_mask_{node.name}"
            )
            causal_layer = network.add_elementwise(
                scaled_qk, causal_mask_trt, trt.ElementWiseOperation.SUM
            )
            causal_layer.name = f"sdpa_causal_{node.name}"
            scaled_qk = causal_layer.get_output(0)

    # Step 5: Softmax along the last dimension
    softmax_layer = network.add_softmax(scaled_qk)
    softmax_layer.axes = 1 << (len(query_shape) - 1)
    softmax_layer.name = f"sdpa_softmax_{node.name}"
    attn_weights = softmax_layer.get_output(0)

    # Step 6: attn_weights @ V
    output_layer = network.add_matrix_multiply(
        attn_weights, trt.MatrixOperation.NONE,
        value_trt, trt.MatrixOperation.NONE,
    )
    output_layer.name = f"sdpa_output_{node.name}"

    return output_layer.get_output(0)


@converter("aten._scaled_dot_product_flash_attention.default", validator_fn=validate_sdpa)
def convert_flash_attention(node, network, input_map, edge_program=None):
    """Convert flash attention — reuse SDPA implementation."""
    return convert_scaled_dot_product_attention(node, network, input_map, edge_program)


@converter("aten._scaled_dot_product_efficient_attention.default", validator_fn=validate_sdpa)
def convert_efficient_attention(node, network, input_map, edge_program=None):
    """Convert efficient attention — reuse SDPA implementation."""
    return convert_scaled_dot_product_attention(node, network, input_map, edge_program)
