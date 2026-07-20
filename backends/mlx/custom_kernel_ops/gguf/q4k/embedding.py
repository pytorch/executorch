#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q4_K** embedding lowering for the MLX GGUF pattern handler.

Lowers a ``dequantize_gguf -> embedding`` pattern to a fused gather Metal kernel
that reads raw ``block_q4_K`` bytes directly (same approach as :mod:`..q6k.embedding`).
"""

from __future__ import annotations

import torch
from executorch.backends.mlx.builder.op_helpers import (
    emit_product,
    emit_shape,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.common import (
    _Q4K_HEADER,
    Q4K_BLOCK_BYTES,
    QK_K,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
)
from torch.fx.node import Node


# ---------------------------------------------------------------------------
# Metal kernel source
# ---------------------------------------------------------------------------


# One thread per output element. grid = (K, num_idx, 1): x picks the feature j,
# y picks the gathered row; each thread dequantizes a single Q4_K element.
_Q4K_EMBED_SOURCE = """
    const uint j = thread_position_in_grid.x;       // 0..K-1
    const uint r = thread_position_in_grid.y;       // gathered row
    const int  row = (int) indices[r];
    if (row < 0 || row >= V) {
        out[r * (uint)K + j] = (OutT)0;
        return;
    }
    const int  nb  = K / QK_K;
    device const block_q4_K * blk =
        ((device const block_q4_K *) weight) + (uint)row * nb + (j / QK_K);
    out[r * (uint)K + j] = (OutT) dequant_q4k_elem(blk, j % QK_K);
"""


def _emit_embedding_fused(
    P: MLXProgramBuilder,
    head: Node,
    weight_node: Node,
    indices_node: Node,
    output_dtype: torch.dtype,
) -> Slot:
    """Lower a Q4_K ``dequantize_gguf`` -> ``embedding`` pattern to a fused gather.

    ``weight_node`` is the raw GGUF blob (the dequantize op's weight input) and
    ``head`` is the ``aten.embedding`` node that owns the output slot.
    """
    weight_slot, indices_slot = P.slot_map([weight_node, indices_node])

    weight_meta = weight_node.meta["val"]
    if weight_meta.dim() != 2:
        raise NotImplementedError(
            f"gguf q4k embedding: weight must be 2-D (vocab, row_bytes); got "
            f"shape {tuple(weight_meta.shape)}"
        )
    vocab = weight_meta.shape[0]
    row_bytes = weight_meta.shape[1]
    if not isinstance(vocab, int) or not isinstance(row_bytes, int):
        raise NotImplementedError(
            "gguf q4k embedding: weight shape must be statically known"
        )
    if row_bytes % Q4K_BLOCK_BYTES != 0:
        raise ValueError(
            f"gguf q4k embedding: weight row bytes {row_bytes} must be a "
            f"multiple of {Q4K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q4K_BLOCK_BYTES) * QK_K

    out_dtype_int = torch_dtype_to_scalar_type(output_dtype)

    out = P.make_or_get_slot(head)
    leading = emit_shape(P, indices_node, indices_slot, end_dim=None)
    num_idx_iov = emit_product(P, leading)
    out_shape_flat = leading + [IntOrVid.from_literal(K)]

    if K % QK_K != 0:
        raise AssertionError(f"gguf q4k embedding: K={K} must be divisible by {QK_K}")
    tg_x = QK_K

    P.emit(
        MetalKernelNode(
            name="gguf_q4k_embedding",
            source=_Q4K_EMBED_SOURCE,
            header=_Q4K_HEADER,
            inputs=[P.slot_to_tid(weight_slot), P.slot_to_tid(indices_slot)],
            outputs=[P.slot_to_tid(out)],
            grid=[IntOrVid.from_literal(K), num_idx_iov, IntOrVid.from_literal(1)],
            threadgroup=[
                IntOrVid.from_literal(tg_x),
                IntOrVid.from_literal(1),
                IntOrVid.from_literal(1),
            ],
            input_names=["weight", "indices"],
            output_names=["out"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[out_dtype_int],
            template_arg_names=["OutT", "K", "V"],
            template_arg_kinds=[2, 0, 0],  # dtype, int, int
            template_arg_values=[out_dtype_int, K, vocab],
        )
    )

    return out


def emit_embedding(
    P: MLXProgramBuilder,
    head: Node,
    weight_node: Node,
    indices_node: Node,
    output_dtype: torch.dtype,
) -> Slot:
    """Dispatch to fused Metal gather or the MLX-native repack path."""
    from executorch.backends.mlx.custom_kernel_ops.gguf.q4k import emit_direct_gguf

    if emit_direct_gguf():
        return _emit_embedding_fused(P, head, weight_node, indices_node, output_dtype)

    from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.embedding_mlx_native import (
        emit_embedding as emit_embedding_mlx_native,
    )

    return emit_embedding_mlx_native(P, head, weight_node, indices_node, output_dtype)
