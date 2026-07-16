#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.exir.scalar_type import ScalarType
from torch.fx.node import Node

if TYPE_CHECKING:
    from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
    from executorch.backends.mlx.serialization.mlx_graph_schema import IntOrVid

# When True, always serialize the biases tensor for quantized ops.
# When False, use init-time computation when zero_point is all zeros,
# computing biases = -scales * 2^(bits-1) during the init chain.
QUANTIZED_SERIALIZE_BIASES = False

# Row-chunk size (in elements) for the int64 bit-packer in ``to_mlx_qparams``.
# Packing widths that don't divide 32 (5/6-bit) needs int64 scratch; chunking the
# rows bounds that scratch to ~this many elements instead of the whole weight
# (a full lm_head would otherwise be tens of GB of int64 -> OOM).
_PACK_CHUNK_ELEMS = 1 << 24  # ~16M int64 elements (~128 MB per scratch tensor)


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
            f"_scalar_{value}",
            torch.tensor(value, dtype=dtype),  # 0-D
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


def emit_shape(
    P: "MLXProgramBuilder",
    node: Node,
    slot: Slot,
    *,
    end_dim: "Optional[int]" = None,
) -> "list[IntOrVid]":
    """Return the shape of ``node`` as a list of ``IntOrVid``.

    Each static dim becomes a literal ``IntOrVid``; each dynamic dim
    emits a ``SymSizeNode`` against ``slot`` and is wrapped via
    ``P.to_int_or_vid``.

    Args:
        P: program builder.
        node: FX node whose shape to walk (must have ``meta['val']``).
        slot: slot corresponding to ``node`` (used as the
            ``SymSize`` source for any dynamic dim).
        end_dim: stop index (exclusive). ``None`` means the full ndim.
            Negative values index from the end (e.g. ``-1`` is "all
            leading dims, drop the last").

    Returns:
        ``list[IntOrVid]`` of length ``end_dim`` (after normalization).
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        IntOrVid,
        SymSizeNode,
    )

    shape = node.meta["val"].shape
    ndim = len(shape)
    if end_dim is None:
        end_dim = ndim
    elif end_dim < 0:
        end_dim += ndim

    out: "list[IntOrVid]" = []
    for dim_idx in range(end_dim):
        s = shape[dim_idx]
        if isinstance(s, int):
            out.append(IntOrVid.from_literal(int(s)))
        else:
            _, d_val = P.make_tmp_value_slot()
            P.emit(
                SymSizeNode(
                    a=P.slot_to_tid(slot),
                    dim=dim_idx,
                    out=P.slot_to_vid(d_val),
                )
            )
            out.append(P.to_int_or_vid(d_val))
    return out


def emit_product(
    P: "MLXProgramBuilder",
    dims: "list[IntOrVid]",
) -> "IntOrVid":
    """Multiplicative reduction over a list of ``IntOrVid`` values.

    Folds all literal entries AOT into a single static product, then
    emits ``MultiplyIntNode`` only for the dynamic entries (and one
    final node combining the static product with the dynamic accumulator
    when both contribute).

    Args:
        P: program builder.
        dims: list of ``IntOrVid``. May be empty (returns
            ``IntOrVid.from_literal(1)``), all literals, or a mix.

    Returns:
        An ``IntOrVid`` representing the product. Always literal when
        every entry is literal (or ``dims`` is empty).
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        IntOrVid,
        MultiplyIntNode,
    )

    static_product = 1
    dynamic_dims: "list[IntOrVid]" = []
    for d in dims:
        if d.is_vid:
            dynamic_dims.append(d)
        else:
            static_product *= d.literal

    if not dynamic_dims:
        return IntOrVid.from_literal(static_product)

    acc = dynamic_dims[0]
    for d in dynamic_dims[1:]:
        _, acc_val = P.make_tmp_value_slot()
        P.emit(MultiplyIntNode(a=acc, b=d, out=P.slot_to_vid(acc_val)))
        acc = P.to_int_or_vid(acc_val)

    if static_product == 1:
        return acc

    _, final_val = P.make_tmp_value_slot()
    P.emit(
        MultiplyIntNode(
            a=IntOrVid.from_literal(static_product),
            b=acc,
            out=P.slot_to_vid(final_val),
        )
    )
    return P.to_int_or_vid(final_val)


def emit_add_int(
    P: "MLXProgramBuilder",
    a: "IntOrVid",
    b: "IntOrVid",
) -> "IntOrVid":
    """Emit ``a + b``, folding to a literal when both operands are static."""
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        AddIntNode,
        IntOrVid,
    )

    if not a.is_vid and not b.is_vid:
        return IntOrVid.from_literal(a.literal + b.literal)

    _, out_slot = P.make_tmp_value_slot()
    P.emit(AddIntNode(a=a, b=b, out=P.slot_to_vid(out_slot)))
    return P.to_int_or_vid(out_slot)


def emit_sub_int(
    P: "MLXProgramBuilder",
    a: "IntOrVid",
    b: "IntOrVid",
) -> "IntOrVid":
    """Emit ``a - b``, folding to a literal when both operands are static."""
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        IntOrVid,
        SubtractIntNode,
    )

    if not a.is_vid and not b.is_vid:
        return IntOrVid.from_literal(a.literal - b.literal)

    _, out_slot = P.make_tmp_value_slot()
    P.emit(SubtractIntNode(a=a, b=b, out=P.slot_to_vid(out_slot)))
    return P.to_int_or_vid(out_slot)


def emit_ceil_div(
    P: "MLXProgramBuilder",
    a: "IntOrVid",
    b: int,
) -> "IntOrVid":
    """Emit ``ceil(a / b)``, folding to a literal when ``a`` is static.

    Computes ``(a + b - 1) // b``.
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        FloorDivideIntNode,
        IntOrVid,
    )

    if not a.is_vid:
        return IntOrVid.from_literal((a.literal + b - 1) // b)

    sum_iov = emit_add_int(P, a, IntOrVid.from_literal(b - 1))
    _, out_slot = P.make_tmp_value_slot()
    P.emit(
        FloorDivideIntNode(
            a=sum_iov,
            b=IntOrVid.from_literal(b),
            out=P.slot_to_vid(out_slot),
        )
    )
    return P.to_int_or_vid(out_slot)


def emit_if_else(
    P: "MLXProgramBuilder",
    cond: "IntOrVid",
    emit_then: Callable[[], None],
    emit_else: Callable[[], None],
) -> None:
    """Emit a branch on ``cond``: nonzero -> then, zero -> else.

    If ``cond`` is a literal, no IfNode or chains are emitted; the
    selected callback is invoked directly in the current chain.
    Otherwise both callbacks are emitted into fresh chains and an
    ``IfNode`` selects between them at runtime. Nodes emitted inside a
    callback land in that branch's chain only.
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import IfNode

    if not cond.is_vid:
        if cond.literal:
            emit_then()
        else:
            emit_else()
        return

    with P.new_chain() as then_idx:
        emit_then()
    with P.new_chain() as else_idx:
        emit_else()

    P.emit(
        IfNode(
            cond=cond,
            then_chain_idx=then_idx,
            else_chain_idx=else_idx,
        )
    )


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


def emit_quantized_gather(
    P: MLXProgramBuilder,
    out: Slot,
    indices_slot: Slot,
    qdata_slot: Slot,
    scales_slot: Slot,
    biases_slot: Optional[Slot],
    *,
    group_size: int,
    bits: int,
    mode: str,
    out_dtype: torch.dtype,
) -> None:
    """Gather quantized rows by index and dequantize them into ``out``.

    Emits ``TakeNode`` for qdata and scales (and biases when present), then a
    ``DequantizeNode``.
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import (
        DequantizeNode,
        IntOrVidOrTid,
        TakeNode,
    )

    ids_index = IntOrVidOrTid.from_tid(P.slot_to_tid(indices_slot))

    _, wq_sel = P.make_tmp_slot()
    P.emit(
        TakeNode(
            x=P.slot_to_tid(qdata_slot),
            index=ids_index,
            out=P.slot_to_tid(wq_sel),
            axis=0,
        )
    )

    _, sc_sel = P.make_tmp_slot()
    P.emit(
        TakeNode(
            x=P.slot_to_tid(scales_slot),
            index=ids_index,
            out=P.slot_to_tid(sc_sel),
            axis=0,
        )
    )

    biases_tid = None
    if biases_slot is not None:
        _, b_sel = P.make_tmp_slot()
        P.emit(
            TakeNode(
                x=P.slot_to_tid(biases_slot),
                index=ids_index,
                out=P.slot_to_tid(b_sel),
                axis=0,
            )
        )
        biases_tid = P.slot_to_tid(b_sel)

    P.emit(
        DequantizeNode(
            w=P.slot_to_tid(wq_sel),
            scales=P.slot_to_tid(sc_sel),
            out=P.slot_to_tid(out),
            biases=biases_tid,
            group_size=group_size,
            bits=bits,
            mode=mode,
            dtype=torch_dtype_to_scalar_type(out_dtype),
        )
    )


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

    # Pack data into a contiguous uint32 bitstream. cols*bits must be a
    # multiple of 32 (holds since in_features is a multiple of group_size>=32).
    rows, cols = qdata.shape
    assert (cols * bits) % 32 == 0

    if bits == 4:
        # 4-bit: view(uint8) + wrapping add + pack 2 nibbles per byte → view as uint32
        q = qdata.view(torch.uint8) + offset
        q3 = q.reshape(rows, cols // 2, 2)
        Q = (q3[:, :, 0] | (q3[:, :, 1] << 4)).view(torch.uint32)
    elif bits == 2:
        # 2-bit: pack 4×2-bit values per byte in uint8, then view as uint32
        Q = ((qdata.view(torch.uint8) + offset) & 0x3).reshape(rows, cols // 4, 4)
        packed = Q[:, :, 0] | (Q[:, :, 1] << 2) | (Q[:, :, 2] << 4) | (Q[:, :, 3] << 6)
        Q = packed.contiguous().view(torch.uint32)
    elif bits == 8:
        # 8-bit: each byte maps 1:1 to a uint32 slot — no shifting needed
        q = qdata.view(torch.uint8) + offset
        Q = q.contiguous().view(torch.uint32).reshape(rows, -1)
    else:
        # Contiguous LSB-first bit-packing for widths that don't divide 32
        # (e.g. 5/6-bit), matching MLX's affine pack_and_quantize (ops.cpp).
        #
        # We scatter each column's value directly into its uint32 word(s) rather
        # than expanding to a per-bit stream. Column j occupies global bits
        # [j*bits, (j+1)*bits): word j*bits//32 at shift j*bits%32, plus a carry
        # into the next word when it straddles the boundary (at most one carry
        # since bits <= 32). Column bit-ranges within a word are disjoint, so
        # index_add_ (sum) is equivalent to OR.
        #
        # The scatter needs int64 (a value shifted by up to 31 overflows int32),
        # so we pack the rows in chunks to bound the peak int64 working set --
        # packing a full lm_head in one shot would otherwise materialize a
        # multi-GB int64 tensor.
        n_words = cols * bits // 32
        pos = torch.arange(cols, dtype=torch.int64) * bits
        word = pos // 32
        shift = pos % 32
        straddle = (shift + bits) > 32
        has_straddle = bool(straddle.any())
        word_carry = (word + 1).clamp(max=n_words - 1) if has_straddle else None

        rows_per_chunk = max(1, _PACK_CHUNK_ELEMS // cols)
        chunks = []
        for r0 in range(0, rows, rows_per_chunk):
            q = qdata[r0 : r0 + rows_per_chunk].to(torch.int64) + offset
            packed = torch.zeros(q.shape[0], n_words, dtype=torch.int64)
            packed.index_add_(1, word, q << shift)  # low bits (+ overflow)
            if has_straddle:
                carry = torch.where(straddle, q >> (32 - shift), torch.zeros_like(q))
                packed.index_add_(1, word_carry, carry)
                del carry
            del q
            chunks.append((packed & 0xFFFFFFFF).to(torch.int32))
            del packed
        packed_i32 = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        del chunks
        Q = packed_i32.contiguous().view(torch.uint32)

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


def parse_dequant_int4_node(
    node: Node,
) -> Optional[Tuple[Node, Node, Node, int, Optional[torch.dtype]]]:
    """Parse a torchao.dequantize_int4_tensor node.

    Returns (qdata, scale, zero_point, group_size, output_dtype) or None if not a
    dequantize_int4_tensor node or the custom op is not registered.
    """
    target = get_aten_target(node.target)
    try:
        import executorch.extension.llm.export.int4  # noqa: F401
    except ImportError:
        return None

    if target is not torch.ops.torchao.dequantize_int4_tensor.default:
        return None

    qdata, scale, zero_point, group_size = node.args[0:4]

    output_dtype = None
    if len(node.args) > 4:
        output_dtype = node.args[4]
    elif "output_dtype" in node.kwargs:
        output_dtype = node.kwargs["output_dtype"]

    return qdata, scale, zero_point, group_size, output_dtype


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
    if group_size not in [16, 32, 64, 128]:
        return None

    # MLX supports 2,3,4,5,6,8-bit affine quantization. to_mlx_qparams packs
    # 2/4/8 via fast paths and other widths (e.g. 5, 6) via a general
    # contiguous bit-packer, so enable 5 and 6 here too.
    bits = (qmax - qmin + 1).bit_length() - 1
    if bits not in [2, 4, 5, 6, 8]:
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
