# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX carrier module for GGUF Q6_K linears.

Wraps the raw GGUF ``block_q6_K`` weight blob and dispatches to the fused
``mlx::gguf_linear`` Metal kernel (decode mat-vec / prefill mat-mat), instead of
the slow non-fused dequantize+matmul path that group_size=16 affine quant takes
through the MLX ``QUANTIZED_LINEAR`` pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Importing the op module registers ``torch.ops.mlx.gguf_linear``.
import executorch.backends.mlx.model_ops.gguf_linear  # noqa: F401
from executorch.backends.mlx.model_ops.gguf_linear import Q6K_BLOCK_BYTES, QK_K


class GGUFLinear(nn.Module):
    """``y = gguf_linear(x, weight_blob, format)`` for a GGUF-quantized linear.

    The weight is the **raw** GGUF block blob, stored as a uint8 buffer of shape
    ``(out_features, n_blocks * block_bytes)``. Gemma linears are bias-free, so
    bias is always ``None``.
    """

    def __init__(self, weight_blob: torch.Tensor, format: str = "q6k"):
        super().__init__()
        if weight_blob.dim() != 2 or weight_blob.dtype != torch.uint8:
            raise ValueError(
                f"GGUFLinear: weight_blob must be 2-D uint8; got "
                f"shape {tuple(weight_blob.shape)} dtype {weight_blob.dtype}"
            )
        if format != "q6k":
            raise NotImplementedError(
                f"GGUFLinear: unsupported format {format!r}; only 'q6k' supported"
            )
        row_bytes = int(weight_blob.shape[1])
        if row_bytes % Q6K_BLOCK_BYTES != 0:
            raise ValueError(
                f"GGUFLinear: weight row bytes {row_bytes} must be a multiple of "
                f"{Q6K_BLOCK_BYTES}"
            )
        self.format = format
        self.out_features = int(weight_blob.shape[0])
        self.in_features = (row_bytes // Q6K_BLOCK_BYTES) * QK_K
        # uint8 cannot be a grad-requiring Parameter; store as a buffer so it is
        # serialized as a constant in the exported program.
        self.register_buffer("weight", weight_blob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.mlx.gguf_linear(x, self.weight, self.format, None)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"format={self.format!r}"
        )


def replace_with_gguf_linear(
    model: nn.Module,
    weight_fqn: str,
    weight_blob: torch.Tensor,
    format: str = "q6k",
) -> None:
    """Replace the ``nn.Linear`` owning ``weight_fqn`` with a ``GGUFLinear``.

    ``weight_fqn`` is the fully-qualified name of the ``.weight`` tensor
    (e.g. ``model.layers.0.mlp.down_proj.weight``). The parent linear module is
    swapped in place on its grandparent module.
    """
    parts = weight_fqn.rsplit(".", 1)
    if len(parts) != 2 or parts[1] != "weight":
        raise ValueError(
            f"replace_with_gguf_linear: expected a '*.weight' fqn; got {weight_fqn!r}"
        )
    linear_fqn = parts[0]
    grandparent_fqn, _, child_name = linear_fqn.rpartition(".")
    grandparent = (
        model.get_submodule(grandparent_fqn) if grandparent_fqn else model
    )
    setattr(grandparent, child_name, GGUFLinear(weight_blob, format=format))
