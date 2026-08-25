# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Callable, NamedTuple, TypeGuard

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from torch.fx import Node

# L-shift value used in CMSIS-NN for int8 operations
SHIFT_INT8 = 20


def quantize_val(val, scale, zp, qmin, qmax):
    return float(min(max(torch.round(torch.Tensor([val / scale + zp])), qmin), qmax))


def extract_constant_scalar(arg: Any) -> float | None:
    if arg is None:
        return None
    if isinstance(arg, (int, float)):
        return float(arg)
    if isinstance(arg, Node):
        if arg.op == "call_function" and arg.target in {
            exir_ops.edge.aten.full_like.default,
            exir_ops.edge.aten.full.default,
            torch.ops.aten.full_like.default,
            torch.ops.aten.full.default,
        }:
            fill_arg = arg.args[1] if len(arg.args) > 1 else None
            return extract_constant_scalar(fill_arg)
        val = arg.meta.get("val")
        if val is None:
            return None
        return extract_constant_scalar(val)
    return None


def get_activation_bounds(node: Node) -> tuple[float | None, float | None] | None:
    bounds: tuple[float | None, float | None]
    match node.target:
        case exir_ops.edge.aten.relu.default | exir_ops.edge.aten.relu_.default:
            bounds = (0.0, None)
        case exir_ops.edge.aten.hardsigmoid.default:
            bounds = (0.0, 1.0)
        case exir_ops.edge.aten.hardtanh.default | exir_ops.edge.aten.hardtanh_.default:
            bounds = (
                extract_constant_scalar(node.args[1]),
                extract_constant_scalar(node.args[2]),
            )
        case exir_ops.edge.aten.clamp.default | exir_ops.edge.aten.clamp.Tensor:
            bounds = (
                extract_constant_scalar(node.args[1]) if len(node.args) > 1 else None,
                extract_constant_scalar(node.args[2]) if len(node.args) > 2 else None,
            )
        case _:
            return None

    min_val, max_val = bounds
    if len(node.args) > 1 and min_val is None and node.args[1] is not None:
        return None
    if len(node.args) > 2 and max_val is None and node.args[2] is not None:
        return None

    return bounds


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


def coerce_int_pair(raw, default: tuple[int, int]) -> tuple[int, int]:
    if hasattr(raw, "meta"):
        raw = raw.meta.get("val", raw)  # type: ignore[attr-defined]
    if raw is None:
        return default
    if isinstance(raw, torch.Tensor):
        raw = raw.flatten().tolist()
    if isinstance(raw, (list, tuple, torch.Size)):
        items = [int(v) for v in raw]
    else:
        items = [int(raw)]
    if not items:
        return default
    if len(items) == 1:
        return (items[0], items[0])
    return (items[0], items[1])


def is_foldable_alpha(alpha: Any) -> TypeGuard[int]:
    """Whether quantized_add can absorb this alpha into an operand multiplier.

    Only an integer can get that far. FoldAndAnnotateQParamsPass re-traces once
    the dequantize nodes are gone, and aten refuses a float alpha on an int8
    add before the lowering ever sees the node.
    """
    return isinstance(alpha, int)


class MaxPool2dParams(NamedTuple):
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]


def max_pool2d_params(node: Node) -> MaxPool2dParams:
    """The kernel, stride, padding and dilation of a max_pool2d node.

    stride's schema default is the empty list, meaning "same as the kernel",
    and export leaves it in place whenever a later argument is named -- which
    ceil_mode always is. int[1] is a legal spelling of an int[2] argument too.
    """
    args = node.args
    # kernel_size is the one required argument, so it is always present.
    kernel = coerce_int_pair(args[1], (1, 1))
    return MaxPool2dParams(
        kernel,
        coerce_int_pair(args[2] if len(args) > 2 else None, kernel),
        coerce_int_pair(args[3] if len(args) > 3 else None, (0, 0)),
        coerce_int_pair(args[4] if len(args) > 4 else None, (1, 1)),
    )


def ceil_mode_is_redundant(
    input_hw: torch.Size,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> bool:
    """Whether ceil_mode picks the same output size that floor_mode would.

    Rounding up adds a row or column whenever the stride does not divide the
    span the windows have to cover. The converse does not hold -- aten drops a
    last window that starts inside the right padding, which can bring the
    ceiling back to the floor -- so this refuses some it could accept.
    """
    if any(not isinstance(n, int) for n in input_hw):
        return False
    return all(
        (n + 2 * p - d * (k - 1) - 1) % s == 0
        for n, k, s, p, d in zip(input_hw, kernel, stride, padding, dilation)
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


def _stable_sigmoid(x: float) -> float:
    # Always exponentiate the non-positive value so `math.exp` never overflows
    # for unusually large `|x|` (e.g. wide-range input qparams). Algebraically
    # identical to `1 / (1 + exp(-x))`.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _stable_silu(x: float) -> float:
    return x * _stable_sigmoid(x)


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def _via_torch(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[float], float]:
    """Evaluate a torch unary at one point, in double precision."""

    def evaluate(x: float) -> float:
        return fn(torch.tensor(x, dtype=torch.float64)).item()

    return evaluate


_ACTIVATION_FNS: dict[EdgeOpOverload, Callable[[float], float]] = {
    exir_ops.edge.aten.sigmoid.default: _stable_sigmoid,
    exir_ops.edge.aten.tanh.default: math.tanh,
    exir_ops.edge.aten.silu.default: _stable_silu,
    exir_ops.edge.aten.gelu.default: _gelu,
    # Only functions with no cheap closed form in the quantized domain belong
    # here; anything expressible as a rescale should not spend a table. exp is
    # deliberately absent despite qualifying: its codomain is unbounded, so an
    # int8 output scale leaves it almost no resolution.
    #
    # Evaluated through torch rather than math so the table inherits IEEE
    # semantics at the edges of each domain: math.log(0) raises where torch
    # returns the -inf that saturates.
    exir_ops.edge.aten.log.default: _via_torch(torch.log),
    exir_ops.edge.aten.log2.default: _via_torch(torch.log2),
    exir_ops.edge.aten.log10.default: _via_torch(torch.log10),
    exir_ops.edge.aten.log1p.default: _via_torch(torch.log1p),
    exir_ops.edge.aten.sqrt.default: _via_torch(torch.sqrt),
    exir_ops.edge.aten.rsqrt.default: _via_torch(torch.rsqrt),
}


def _round_half_away_from_zero(x: float) -> int:
    # Matches the rounding convention `requantize_cmsis` (above) applies after
    # the right-shift step: ties on positive values round toward +∞, ties on
    # negative values round toward -∞. Python's built-in `round` would use
    # banker's rounding instead and disagree at exact half-integers.
    return int(math.copysign(math.floor(abs(x) + 0.5), x)) if x != 0 else 0


def build_activation_lut(
    target,
    input_scale: float,
    input_zp: int,
    output_scale: float,
    output_zp: int,
) -> torch.Tensor:
    """AoT-compute a 256-entry int8 lookup table for a quantized activation.

    `target` is the edge-dialect op being lowered (e.g.
    `exir_ops.edge.aten.sigmoid.default`).

    The LUT is indexed by the input byte value biased by 128: for any int8
    input `q_in`, the kernel reads `lut[q_in + 128]` to get the int8 output.
    Because the LUT is computed in float and quantized once per entry, the
    runtime kernel is a single memory-lookup with no requantization math.
    """
    if target not in _ACTIVATION_FNS:
        raise ValueError(
            f"build_activation_lut: unsupported activation target {target!r} "
            f"(supported: {sorted(t.__name__ for t in _ACTIVATION_FNS)})"
        )
    f = _ACTIVATION_FNS[target]
    defined: dict[int, int] = {}
    undefined: list[int] = []
    for q in range(-128, 128):
        x = (q - input_zp) * input_scale
        y = f(x)
        scaled = y / output_scale + output_zp if math.isfinite(y) else y
        if math.isnan(scaled):
            # log of a negative, rsqrt of a negative. Filled in below.
            undefined.append(q + 128)
            continue
        if not math.isfinite(scaled):
            # A pole. The rail is the closest int8 has to it.
            q_out = 127 if scaled > 0 else -128
        else:
            q_out = _round_half_away_from_zero(scaled)
        defined[q + 128] = max(-128, min(127, q_out))

    if not defined:
        raise ValueError(
            f"build_activation_lut: {target} is undefined across the whole "
            f"input range (scale {input_scale}, zero point {input_zp})"
        )

    lut = torch.empty(256, dtype=torch.int8)
    for index, value in defined.items():
        lut[index] = value
    # Each of these functions is undefined on one side of a boundary, so an
    # undefined entry continues the value at that boundary. That keeps the table
    # monotone; emitting the output zero point instead would put a mid-range
    # value below the pole's rail.
    for index in undefined:
        lut[index] = defined[min(defined, key=lambda d: abs(d - index))]
    return lut


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


def is_channels_last(tensor: torch.Tensor) -> bool:
    """Check if a 4D tensor is in channels last format."""
    if tensor.ndim != 4:
        return False

    if tensor.shape[1] == 1 or tensor.shape[2] == tensor.shape[3] == 1:
        return True

    dim_order = list(tensor.dim_order())
    return dim_order[0:2] == [0, 2]


_NHWC_DIM_ORDER = [0, 2, 3, 1]


def to_physical_order(logical_pad: list[int], tensor: torch.Tensor) -> list[int]:
    """Permute a 4-element NCHW-ordered list to NHWC physical memory order
    when ``tensor`` is in channels_last format, otherwise return unchanged."""
    if not is_channels_last(tensor):
        return logical_pad
    return [logical_pad[_NHWC_DIM_ORDER[i]] for i in range(4)]


def is_channel_broadcast(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    """
    Check if tensor1 is broadcasted to tensor2 along channel dimension.
    Assumes tensor2 has shape [N, C, ...] and tensor1 has shape [N, 1, ...] or [1, C, ...].
    """
    if tensor1.dim() != tensor2.dim():
        return False
    if not is_channels_last(tensor1):
        return False
    if not is_channels_last(tensor2):
        return False

    channel_match = tensor1.size(1) == tensor2.size(1)
    tensor1_channels_only = tensor1.numel() == tensor1.size(1)
    tensor2_channels_only = tensor2.numel() == tensor2.size(1)

    return channel_match and (tensor1_channels_only or tensor2_channels_only)
