#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.exir.scalar_type import ScalarType
from torch.fx.node import Node

if TYPE_CHECKING:
    from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder

# When True, always serialize the biases tensor for quantized ops.
# When False, use init-time computation when zero_point is all zeros,
# computing biases = -scales * 2^(bits-1) during the init chain.
QUANTIZED_SERIALIZE_BIASES = False


def get_aten_target(target):
    """
    Unwrap EdgeOpOverload to get the underlying ATen op.

    In Edge IR, ops are wrapped in EdgeOpOverload. This extracts the
    underlying ATen op for consistent comparison.
    """
    if hasattr(target, "_op") and "EdgeOpOverload" in type(target).__name__:
        return target._op
    return target


# Mapping from _copy variants to their non-copy equivalents.
# Edge IR uses _copy variants for certain ops, but for pattern matching
# we want to compare against the semantic operation.
_COPY_TO_NON_COPY = {
    torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
    torch.ops.aten.transpose_copy.int: torch.ops.aten.transpose.int,
    torch.ops.aten.view_copy.default: torch.ops.aten.view.default,
    torch.ops.aten.permute_copy.default: torch.ops.aten.permute.default,
    torch.ops.aten.unsqueeze_copy.default: torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze_copy.dim: torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze_copy.dims: torch.ops.aten.squeeze.dims,
    torch.ops.aten.squeeze_copy.default: torch.ops.aten.squeeze.default,
    torch.ops.aten.expand_copy.default: torch.ops.aten.expand.default,
    torch.ops.aten.alias_copy.default: torch.ops.aten.alias.default,
}


def get_aten_target_normalized(target):
    """
    Get ATen target, mapping _copy variants to their non-copy equivalents.

    Use this for pattern matching where Edge IR uses _copy variants but
    we want to match the semantic operation.

    E.g., aten.transpose_copy.int -> aten.transpose.int
    """
    target = get_aten_target(target)
    return _COPY_TO_NON_COPY.get(target, target)


def emit_stop_position(
    P: "MLXProgramBuilder",
    start: "Union[int, Slot]",
    length_tensor: "Slot",
    length_dim: int,
    length_meta: "Optional[torch.Tensor]" = None,
) -> "Union[int, Slot]":
    """
    Emit nodes to compute stop = start + length for slice operations.

    May emit SymSizeNode and/or AddIntNode depending on whether
    start and length are static or dynamic.

    Args:
        P: The program builder
        start: Start position (int or Slot)
        length_tensor: The tensor slot whose dimension gives the length
        length_dim: Which dimension of length_tensor contains the length
        length_meta: Optional tensor metadata for static length extraction

    Returns:
        stop position as int (if fully static) or Slot (if any dynamic)
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        AddIntNode,
        IntOrVid,
        SymSizeNode,
    )

    # Check if seq_len is symbolic (dynamic)
    seq_len_is_symbolic = False
    seq_len_concrete = None

    if length_meta is not None:
        seq_len_dim = length_meta.shape[length_dim]
        if hasattr(seq_len_dim, "node"):
            seq_len_is_symbolic = True
        else:
            seq_len_concrete = int(seq_len_dim)

    if seq_len_is_symbolic or length_meta is None:
        # Dynamic seq_len: emit SymSizeNode to get length at runtime
        _, seq_len_slot = P.slot_manager.make_tmp_value_slot()
        P.emit(
            SymSizeNode(
                a=P.slot_to_tid(length_tensor),
                dim=length_dim,
                out=P.slot_to_vid(seq_len_slot),
            )
        )
        _, stop_slot = P.slot_manager.make_tmp_value_slot()
        if isinstance(start, Slot):
            start_iov = P.to_int_or_vid(start)
        else:
            start_iov = IntOrVid.from_literal(int(start))
        P.emit(
            AddIntNode(
                a=start_iov,
                b=IntOrVid.from_vid(P.slot_to_vid(seq_len_slot)),
                out=P.slot_to_vid(stop_slot),
            )
        )
        return stop_slot
    else:
        # Static seq_len
        if isinstance(start, Slot):
            # Dynamic start + static length
            _, stop_slot = P.slot_manager.make_tmp_value_slot()
            P.emit(
                AddIntNode(
                    a=P.to_int_or_vid(start),
                    b=IntOrVid.from_literal(seq_len_concrete),
                    out=P.slot_to_vid(stop_slot),
                )
            )
            return stop_slot
        else:
            # Both static - just return the sum
            return start + seq_len_concrete


def emit_lifted_constant(P: "MLXProgramBuilder", value, dtype: torch.dtype) -> Slot:
    """Lift a scalar to a 0-D tensor.

    Concrete scalars (int/float/bool) become deduplicated constants.
    Dynamic values (SymInt Slots) emit a FullNode at runtime.
    """

    if isinstance(value, (int, float, bool)):
        return P.make_or_get_constant(
            f"_scalar_{value}", torch.tensor(value, dtype=dtype)  # 0-D
        )

    from executorch.backends.mlx.serialization.mlx_graph_schema import FullNode

    _, slot = P.make_tmp_slot()
    P.emit(
        FullNode(
            shape=[],
            v=P.to_float_or_vid(value),
            scalar_type=torch_dtype_to_scalar_type(dtype),
            out=P.slot_to_tid(slot),
        )
    )
    return slot


def emit_quantized_biases(
    P: "MLXProgramBuilder",
    zero_point_key: str,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int,
    B: torch.Tensor,
    scale_slot: "Slot",
) -> "Slot":
    """Emit biases for quantized ops, computing at init time when possible.

    When zero_point is all zeros and QUANTIZED_SERIALIZE_BIASES is False,
    avoids serializing the biases tensor by computing biases = scales * -offset
    during the init chain instead.

    Returns the biases Slot.
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import MultiplyNode
    from torch._subclasses.fake_tensor import FakeTensor

    is_scale_only = False
    if not isinstance(zero_point, FakeTensor):
        if torch.sum(torch.abs(zero_point)).item() == 0:
            is_scale_only = True

    if QUANTIZED_SERIALIZE_BIASES or not is_scale_only:
        return P.make_or_get_constant(f"{zero_point_key}_to_biases", B)

    scale_dtype = scale.dtype
    offset = 1 << (bits - 1)
    neg_offset = emit_lifted_constant(P, -offset, scale_dtype)
    biases = P.make_or_get_constant(
        f"{zero_point_key}_to_biases_dummy", torch.tensor(0.0, dtype=B.dtype)
    )
    P.emit_init(
        MultiplyNode(
            a=P.slot_to_tid(scale_slot),
            b=P.slot_to_tid(neg_offset),
            out=P.slot_to_tid(biases),
        )
    )
    return biases


def to_mlx_qparams(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int,
    compute_biases: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert TorchAO quantization params to MLX format.

    TorchAO uses: s * (q - z), with q signed
    MLX uses: S * Q + B, with Q unsigned

    s * (q - z)
      = s ((q + offset) - (z + offset))
      = s Q + B,
    where Q = q + offset, B = -s * (z + offset)

    Args:
        compute_biases: If False, skip bias computation (for scale_only mode).
                       Returns (Q, None) in this case. This is valid when
                       zero_point is all zeros, as the C++ runtime will compute
                       biases = -scales * 2^(bits-1).
    """
    assert qdata.dtype == torch.int8
    offset = 2 ** (bits - 1)
    Q = qdata.to(torch.int32) + offset

    # Pack data tightly into uint32
    assert 32 % bits == 0
    vals_per_uint32 = 32 // bits
    assert qdata.shape[1] % vals_per_uint32 == 0

    Q = Q.reshape(-1, vals_per_uint32)
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)

    # Convert to int64 for shift/packing
    Q = Q.to(torch.int64)
    Q = (Q << shifts).sum(dim=-1)
    Q = Q.to(torch.uint32)
    Q = Q.reshape(qdata.shape[0], -1)

    if compute_biases:
        B = -scale * (zero_point.to(scale.dtype) + offset)
        return Q, B
    else:
        return Q, None


def parse_dequant_nvfp4_node(
    node: Node,
) -> Optional[Tuple[Node, Node, Node, torch.dtype]]:
    """Parse a torchao.dequantize_nvfp4 node.

    Returns (qdata, scale, per_tensor_scale, output_dtype) or None if not a
    dequantize_nvfp4 node or the custom op is not registered.
    """
    target = get_aten_target(node.target)
    try:
        import executorch.extension.llm.export.nvfp4  # noqa: F401
    except ImportError:
        return None

    if target is not torch.ops.torchao.dequantize_nvfp4.default:
        return None

    qdata, scale, per_tensor_scale = node.args[0:3]

    output_dtype = torch.float32
    if len(node.args) > 4:
        output_dtype = node.args[4]
    elif "output_dtype" in node.kwargs:
        output_dtype = node.kwargs["output_dtype"]

    return qdata, scale, per_tensor_scale, output_dtype


def parse_dequant_node(
    node: Node,
) -> Optional[Tuple[Node, Node, Node, int, int, Optional[torch.dtype], int]]:
    """Parse a torchao.dequantize_affine node.

    Accepts N-dimensional block_size with a single non-1 element identifying
    the quantized dimension and group_size. For example:
      - Linear weights (2D):  block_size=[1, 32]       → quantized_dim=1
      - Conv2d weights (4D):  block_size=[1, 32, 1, 1] → quantized_dim=1

    Returns (qdata, scale, zero_point, group_size, bits, out_dtype, quantized_dim)
    or None if unsupported.
    """
    qdata, block_size, scale, zero_point, dtype, qmin, qmax = node.args[0:7]
    out_dtype = (
        node.args[7] if len(node.args) > 7 else node.kwargs.get("output_dtype", None)
    )
    if dtype != torch.int8:
        return None
    if len(block_size) < 2:
        return None
    non_one = [(i, d) for i, d in enumerate(block_size) if d != 1]
    if len(non_one) != 1:
        return None
    quantized_dim, group_size = non_one[0]
    if group_size not in [32, 64, 128]:
        return None
    if qmin == -8 and qmax == 7:
        bits = 4
    elif qmin == -128 and qmax == 127:
        bits = 8
    else:
        return None
    return qdata, scale, zero_point, group_size, bits, out_dtype, quantized_dim


# Mapping from torch dtype to ET ScalarType int value
# See executorch/exir/scalar_type.py for ScalarType enum
_TORCH_DTYPE_TO_SCALAR_TYPE: Dict[torch.dtype, int] = {
    torch.float16: ScalarType.HALF,
    torch.float32: ScalarType.FLOAT,
    torch.bfloat16: ScalarType.BFLOAT16,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.uint32: ScalarType.UINT32,
    torch.uint8: ScalarType.BYTE,
    torch.bool: ScalarType.BOOL,
    torch.int8: ScalarType.CHAR,
}


def torch_dtype_to_scalar_type(dtype: torch.dtype) -> int:
    """Convert torch dtype to ET ScalarType int value."""
    if dtype not in _TORCH_DTYPE_TO_SCALAR_TYPE:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return int(_TORCH_DTYPE_TO_SCALAR_TYPE[dtype])
