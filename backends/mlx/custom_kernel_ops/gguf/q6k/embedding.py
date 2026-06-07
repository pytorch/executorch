#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q6_K** embedding implementation.

Provides the Q6_K embedding lowering used by the MLX GGUF pattern handler
(:mod:`..patterns`):

* :func:`emit_embedding` -- lowers a ``dequantize_gguf -> embedding`` pattern to
  a fused Q6_K gather Metal kernel.

This is the gather counterpart to :mod:`.linear` and exists because MLX's affine
dequantize has no group_size=16 Metal kernel, so a Q6_K embedding (group_size 16)
cannot use the generic quantized-embedding path.
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
from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.common import (
    _Q6K_HEADER,
    Q6K_BLOCK_BYTES,
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


def emit_embedding(
    P: MLXProgramBuilder,
    head: Node,
    weight_node: Node,
    indices_node: Node,
    output_dtype: torch.dtype,
) -> Slot:
    """Lower a Q6_K ``dequantize_gguf`` -> ``embedding`` pattern to a fused gather.

    ``weight_node`` is the raw GGUF blob (the dequantize op's weight input) and
    ``head`` is the ``aten.embedding`` node that owns the output slot.
    """
    weight_slot, indices_slot = P.slot_map([weight_node, indices_node])

    weight_meta = weight_node.meta["val"]
    if weight_meta.dim() != 2:
        raise NotImplementedError(
            f"gguf q6k embedding: weight must be 2-D (vocab, row_bytes); got "
            f"shape {tuple(weight_meta.shape)}"
        )
    row_bytes = weight_meta.shape[1]
    if not isinstance(row_bytes, int):
        raise NotImplementedError(
            "gguf q6k embedding: weight shape must be statically known"
        )
    if row_bytes % Q6K_BLOCK_BYTES != 0:
        raise ValueError(
            f"gguf q6k embedding: weight row bytes {row_bytes} must be a "
            f"multiple of {Q6K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q6K_BLOCK_BYTES) * QK_K

    out_dtype_int = torch_dtype_to_scalar_type(output_dtype)

    out = P.make_or_get_slot(head)
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
