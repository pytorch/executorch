# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch


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
    mantissa, exponent = math.frexp(scale)
    shift = -exponent
    q_fixed = int(round(mantissa * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift -= 1
    multiplier = max(-2147483648, min(2147483647, q_fixed))
    return multiplier, shift


def cleanup_erased_nodes(graph_module: torch.fx.GraphModule):
    # Placeholder for any additional cleanup if needed
    pass
