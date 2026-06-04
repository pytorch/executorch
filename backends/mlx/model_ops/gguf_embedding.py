#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::gguf_embedding``: embedding gather against a GGUF-quantized table.

    out = dequant(weight[indices])

The embedding table is the raw GGUF ``block_q6_K`` blob (one quantized row per
vocab entry). This is the gather counterpart to ``mlx::gguf_linear`` and exists
because MLX's affine dequantize has no group_size=16 Metal kernel, so a Q6_K
embedding (group_size 16) cannot use the generic quantized-embedding path.

``format`` selects the GGUF quant type; only ``"q6k"`` is supported. Output is
bfloat16.

Usage::

    import executorch.backends.mlx.model_ops.gguf_embedding  # noqa: F401

    out = torch.ops.mlx.gguf_embedding(weight, indices, "q6k")
    # weight:  (vocab, (K/256)*210)  uint8  GGUF q6_K blob
    # indices: (...)                 int
    # out:     (..., K)              bfloat16
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.fx.node import Node

from executorch.backends.mlx.model_ops.gguf_linear import (
    _Q6K_HEADER,
    dequantize_q6_k,
    Q6K_BLOCK_BYTES,
    QK_K,
)


# ---------------------------------------------------------------------------
# Custom op + eager fallback
# ---------------------------------------------------------------------------


@torch.library.custom_op("mlx::gguf_embedding", mutates_args=())
def gguf_embedding(weight: Tensor, indices: Tensor, format: str) -> Tensor:
    """Gather + dequantize rows of a GGUF-quantized embedding table.

    Args:
        weight: ``(vocab, (K/256)*210)`` uint8 GGUF ``q6_K`` blob.
        indices: integer token ids of any shape.
        format: GGUF quant type; only ``"q6k"`` supported.

    Returns:
        ``(*indices.shape, K)`` bfloat16.
    """
    if format != "q6k":
        raise NotImplementedError(
            f"mlx::gguf_embedding: unsupported format {format!r}; only 'q6k' "
            f"is supported"
        )
    if weight.dim() != 2:
        raise ValueError(
            f"mlx::gguf_embedding: weight must be 2-D (vocab, row_bytes); got "
            f"shape {tuple(weight.shape)}"
        )
    row_bytes = weight.shape[1]
    if row_bytes % Q6K_BLOCK_BYTES != 0:
        raise ValueError(
            f"mlx::gguf_embedding: weight row bytes {row_bytes} must be a "
            f"multiple of {Q6K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q6K_BLOCK_BYTES) * QK_K

    rows = weight[indices.reshape(-1).long()]  # (num, row_bytes)
    deq = dequantize_q6_k(rows, K)  # (num, K) float32
    return deq.reshape(*indices.shape, K).to(torch.bfloat16)


@torch.library.register_fake("mlx::gguf_embedding")
def gguf_embedding_fake(weight: Tensor, indices: Tensor, format: str) -> Tensor:
    row_bytes = weight.shape[1]
    K = (row_bytes // Q6K_BLOCK_BYTES) * QK_K
    return indices.new_empty((*indices.shape, K), dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# MLX handler
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_helpers import (
    emit_product,
    emit_shape,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
)


# One thread per output element. grid = (K, num_idx, 1): x picks the feature j,
# y picks the gathered row; each thread dequantizes a single Q6_K element.
_Q6K_EMBED_SOURCE = """
    const uint j = thread_position_in_grid.x;       // 0..K-1
    const uint r = thread_position_in_grid.y;       // gathered row
    const int  row = (int) indices[r];
    const int  nb  = K / QK_K;
    device const block_q6_K * blk =
        ((device const block_q6_K *) weight) + (uint)row * nb + (j / QK_K);
    out[r * (uint)K + j] = (OutT) dequant_q6k_elem(blk, j % QK_K);
"""


@REGISTRY.register(target=[torch.ops.mlx.gguf_embedding.default])
def _gguf_embedding_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Lower ``mlx::gguf_embedding`` to a fused Q6_K gather Metal kernel."""
    args = P.args(n)
    if len(args) != 3:
        raise ValueError(
            f"mlx::gguf_embedding: expected 3 args (weight, indices, format); "
            f"got {len(args)}"
        )
    weight_slot, indices_slot, fmt = args
    weight_node = n.args[0]
    indices_node = n.args[1]

    if fmt != "q6k":
        raise NotImplementedError(
            f"mlx::gguf_embedding: unsupported format {fmt!r}; only 'q6k' "
            f"is supported"
        )

    weight_meta = weight_node.meta["val"]
    if weight_meta.dim() != 2:
        raise NotImplementedError(
            f"mlx::gguf_embedding: weight must be 2-D (vocab, row_bytes); got "
            f"shape {tuple(weight_meta.shape)}"
        )
    row_bytes = weight_meta.shape[1]
    if not isinstance(row_bytes, int):
        raise NotImplementedError(
            "mlx::gguf_embedding: weight shape must be statically known"
        )
    if row_bytes % Q6K_BLOCK_BYTES != 0:
        raise ValueError(
            f"mlx::gguf_embedding: weight row bytes {row_bytes} must be a "
            f"multiple of {Q6K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q6K_BLOCK_BYTES) * QK_K

    out_dtype_int = torch_dtype_to_scalar_type(torch.bfloat16)

    out = P.make_or_get_slot(n)
    leading = emit_shape(P, indices_node, indices_slot, end_dim=None)
    num_idx_iov = emit_product(P, leading)
    out_shape_flat = leading + [IntOrVid.from_literal(K)]

    # threadgroup.x must divide grid.x (= K, a multiple of 256).
    tg_x = 256 if K % 256 == 0 else K

    P.emit(
        MetalKernelNode(
            name="gguf_q6k_embedding",
            source=_Q6K_EMBED_SOURCE,
            header=_Q6K_HEADER,
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
            template_arg_names=["OutT", "K"],
            template_arg_kinds=[2, 0],  # dtype, int
            template_arg_values=[out_dtype_int, K],
        )
    )

    return out
