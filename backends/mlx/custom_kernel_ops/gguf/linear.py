#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""``mlx::gguf_linear``: linear layer against a GGUF-quantized weight.

    out = x @ dequant(weight)^T (+ bias)

The weight is stored in the **exact GGUF packed block layout** (no repacking),
so weights converted by llama.cpp / gguf-py can be consumed directly. The
``format`` argument selects the GGUF quantization type.

This module is a thin **format router**: it owns the ``mlx::gguf_linear`` op
identity (custom op, fake, and lowering registration) and dispatches on
``format`` to a per-format implementation. Only ``"q6k"`` is supported today
(see :mod:`.q6k.linear`); other formats raise ``NotImplementedError``. To add a
format, implement ``eager_linear`` / ``emit_linear`` in a sibling package (e.g.
``q4k``) and add a branch below.

Usage::

    import executorch.backends.mlx.custom_kernel_ops.gguf.linear  # noqa: F401

    out = torch.ops.mlx.gguf_linear(x, weight, "q6k", bias)
    # x:      (..., K)          bf16 / fp16 / fp32
    # weight: (N, (K/256)*210)  uint8  GGUF q6_K blob
    # bias:   (N,) or None      activation dtype
    # out:    (..., N)          activation dtype
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.fx.node import Node


# ---------------------------------------------------------------------------
# Custom op + eager fallback (format-agnostic shell; dispatches by format)
# ---------------------------------------------------------------------------


@torch.library.custom_op("mlx::gguf_linear", mutates_args=())
def gguf_linear(
    x: Tensor,
    weight: Tensor,
    format: str,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Linear against a GGUF-quantized weight.

    Args:
        x: ``(..., K)`` activations (bf16 / fp16 / fp32).
        weight: ``(N, row_bytes)`` uint8 GGUF quant blob.
        format: GGUF quant type; only ``"q6k"`` supported.
        bias: optional ``(N,)`` of activation dtype.

    Returns:
        ``(..., N)`` of activation dtype.
    """
    if format == "q6k":
        from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.linear import (
            eager_linear,
        )

        return eager_linear(x, weight, bias)
    raise NotImplementedError(
        f"mlx::gguf_linear: unsupported format {format!r}; only 'q6k' is supported"
    )


@torch.library.register_fake("mlx::gguf_linear")
def gguf_linear_fake(
    x: Tensor,
    weight: Tensor,
    format: str,
    bias: Optional[Tensor] = None,
) -> Tensor:
    N = weight.shape[0]
    out_shape = list(x.shape)
    out_shape[-1] = N
    return x.new_empty(out_shape, dtype=x.dtype)


# ---------------------------------------------------------------------------
# MLX handler (format router)
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot


@REGISTRY.register(target=[torch.ops.mlx.gguf_linear.default])
def _gguf_linear_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Route ``mlx::gguf_linear`` lowering to the per-format implementation."""
    args = P.args(n)
    fmt = args[2] if len(args) >= 3 else None
    if fmt == "q6k":
        from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.linear import (
            emit_linear,
        )

        return emit_linear(P, n)
    raise NotImplementedError(
        f"mlx::gguf_linear: unsupported format {fmt!r}; only 'q6k' is supported"
    )
