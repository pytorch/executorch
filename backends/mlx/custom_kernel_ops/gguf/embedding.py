#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""``mlx::gguf_embedding``: embedding gather against a GGUF-quantized table.

    out = dequant(weight[indices])

This module is a thin **format router**: it owns the ``mlx::gguf_embedding`` op
identity (custom op, fake, and lowering registration) and dispatches on
``format`` to a per-format implementation. Only ``"q6k"`` is supported today
(see :mod:`.q6k.embedding`); other formats raise ``NotImplementedError``.

Usage::

    import executorch.backends.mlx.custom_kernel_ops.gguf.embedding  # noqa: F401

    out = torch.ops.mlx.gguf_embedding(weight, indices, "q6k")
    # weight:  (vocab, (K/256)*210)  uint8  GGUF q6_K blob
    # indices: (...)                 int
    # out:     (..., K)              bfloat16
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.fx.node import Node


# ---------------------------------------------------------------------------
# Custom op + eager fallback (format-agnostic shell; dispatches by format)
# ---------------------------------------------------------------------------


@torch.library.custom_op("mlx::gguf_embedding", mutates_args=())
def gguf_embedding(weight: Tensor, indices: Tensor, format: str) -> Tensor:
    """Gather + dequantize rows of a GGUF-quantized embedding table.

    Args:
        weight: ``(vocab, row_bytes)`` uint8 GGUF quant blob.
        indices: integer token ids of any shape.
        format: GGUF quant type; only ``"q6k"`` supported.

    Returns:
        ``(*indices.shape, K)`` bfloat16.
    """
    if format == "q6k":
        from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.embedding import (
            eager_embedding,
        )

        return eager_embedding(weight, indices)
    raise NotImplementedError(
        f"mlx::gguf_embedding: unsupported format {format!r}; only 'q6k' is supported"
    )


@torch.library.register_fake("mlx::gguf_embedding")
def gguf_embedding_fake(weight: Tensor, indices: Tensor, format: str) -> Tensor:
    from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.common import (
        Q6K_BLOCK_BYTES,
        QK_K,
    )

    row_bytes = weight.shape[1]
    K = (row_bytes // Q6K_BLOCK_BYTES) * QK_K
    return indices.new_empty((*indices.shape, K), dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# MLX handler (format router)
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot


@REGISTRY.register(target=[torch.ops.mlx.gguf_embedding.default])
def _gguf_embedding_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Route ``mlx::gguf_embedding`` lowering to the per-format implementation."""
    args = P.args(n)
    fmt = args[2] if len(args) >= 3 else None
    if fmt == "q6k":
        from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.embedding import (
            emit_embedding,
        )

        return emit_embedding(P, n)
    raise NotImplementedError(
        f"mlx::gguf_embedding: unsupported format {fmt!r}; only 'q6k' is supported"
    )
