# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx import Node

# L-shift value used in CMSIS-NN for int8 operations
SHIFT_INT8 = 20


def quantize_val(val, scale, zp, qmin, qmax):
    return min(max(round(val / scale + zp), qmin), qmax)


def dequantize_per_tensor_cmsis(
    qtensor: torch.Tensor, zero_point: int, multiplier: int, shift: int
) -> torch.Tensor:
    """
    Simulate CMSIS-NN fixed-point dequantization:
    result = (qtensor - zero_point) * multiplier * 2^shift / 2^31
    """
    scale = multiplier * (2**shift) / (1 << 31)
    return (qtensor.float() - zero_point) * scale


def quantize_per_tensor_cmsis(
    tensor: torch.Tensor,
    zero_point: int,
    multiplier: int,
    shift: int,
    qmin=-128,
    qmax=127,
) -> torch.Tensor:
    """
    Simulate CMSIS-NN fixed-point quantization:
    result = round(tensor / scale) + zero_point, clamped to [qmin, qmax]
    """
    scale = multiplier * (2**shift) / (1 << 31)
    quantized = torch.round(tensor / scale) + zero_point
    return quantized.clamp(qmin, qmax).to(torch.int8)


def requantize_cmsis(
    tensor: torch.Tensor,
    multiplier: int,
    shift: int,
) -> torch.Tensor:
    """Simulate CMSIS-NN's arm_nn_requantize helper."""

    tensor_64 = tensor.to(torch.int64)
    left_shift = max(shift, 0)
    right_shift = max(-shift, 0)

    # Equivalent to val * (1 << LEFT_SHIFT(shift))
    value = tensor_64 << left_shift

    # arm_nn_doubling_high_mult_no_sat(value, multiplier)
    product = value * int(multiplier)
    product = product + (1 << 30)
    result = product >> 31

    if right_shift:
        remainder_mask = (1 << right_shift) - 1
        remainder = torch.bitwise_and(result, remainder_mask)
        result = result >> right_shift
        threshold = remainder_mask >> 1
        threshold_tensor = torch.full_like(result, threshold, dtype=torch.int64)
        threshold_tensor = torch.where(
            result < 0, threshold_tensor + 1, threshold_tensor
        )
        result = result + torch.where(remainder > threshold_tensor, 1, 0)

    return result.to(torch.int32)


def extract_scalar_value(node_arg) -> float:
    """
    Extract scalar value from various PyTorch scalar representations.
    """
    if hasattr(node_arg, "op") and node_arg.op == "get_attr":
        # Handle case where scalar is a graph attribute
        return float(node_arg.target)
    elif isinstance(node_arg, (int, float)):
        return float(node_arg)
    elif hasattr(node_arg, "item"):
        return float(node_arg.item())
    else:
        # Try to extract from meta if available
        if hasattr(node_arg, "meta") and "val" in node_arg.meta:
            val = node_arg.meta["val"]
            if hasattr(val, "item"):
                return float(val.item())
            return float(val)
        raise ValueError(
            f"Cannot extract scalar value from {type(node_arg)}: {node_arg}"
        )


def is_qualified_int8_node(args) -> bool:
    try:
        if len(args) < 6:
            return False
        qmin = int(args[3])
        qmax = int(args[4])
        dtype_str = str(args[5])
        is_int8_range = (
            qmin >= torch.iinfo(torch.int8).min and qmax <= torch.iinfo(torch.int8).max
        )
        is_int8_dtype = "int8" in dtype_str.lower()
        return is_int8_range and is_int8_dtype
    except (IndexError, ValueError, TypeError):
        return False


def quantize_multiplier_aot(scale: float) -> tuple[int, int]:
    if scale == 0.0:
        return 0, 0
    mantissa, shift = math.frexp(scale)
    q_fixed = int(round(mantissa * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    multiplier = max(
        torch.iinfo(torch.int32).min, min(torch.iinfo(torch.int32).max, q_fixed)
    )
    return multiplier, shift


def cleanup_erased_nodes(graph_module: torch.fx.GraphModule):
    # Placeholder for any additional cleanup if needed
    pass


def transfer_metadata(
    new_node: Node, source_node: Node, pass_name: str = "QuantizedPass"
) -> None:
    """Transfer metadata with proper provenance tracking."""
    if hasattr(source_node, "meta") and source_node.meta:
        new_node.meta = source_node.meta.copy()
        if "from_node" in new_node.meta:
            from_node_list = new_node.meta.get("from_node", []).copy()
            from_node_list.append(
                {"source": source_node.name, "pass": pass_name, "op": "fuse"}
            )
            new_node.meta["from_node"] = from_node_list
        for field in ["tensor_meta", "stack_trace"]:
            if field in source_node.meta:
                new_node.meta[field] = source_node.meta[field]


def is_dequant_node(node: Node) -> bool:
    """Check if node is a dequantize operation."""
    dequant_targets = {
        exir_ops.edge.cortex_m.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    }
    return node.op == "call_function" and node.target in dequant_targets


def is_quant_node(node: Node) -> bool:
    """Check if node is a quantize operation."""
    quant_targets = {
        exir_ops.edge.cortex_m.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    }
    return node.op == "call_function" and node.target in quant_targets


def cleanup_nodes(nodes_to_erase, graph):
    """Clean up marked nodes from graph."""
    failed_nodes = []

    for node in reversed(nodes_to_erase):
        if node in graph.nodes and len(node.users) == 0:
            try:
                graph.erase_node(node)
            except Exception as e:
                print(f"Warning: Failed to erase node {node}: {e}")
                failed_nodes.append(node)
                continue

    if failed_nodes:
        print(f"Warning: {len(failed_nodes)} nodes could not be erased")

    return failed_nodes
