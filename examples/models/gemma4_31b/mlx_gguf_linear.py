# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX carrier modules for GGUF Q6_K weights.

Wrap raw GGUF ``block_q6_K`` blobs and dispatch to the fused ``mlx::gguf_linear``
(matmul) and ``mlx::gguf_embedding`` (gather) Metal kernels, instead of the slow
non-fused dequantize paths that group_size=16 affine quant takes through the MLX
``QUANTIZED_LINEAR`` / quantized-embedding patterns.
"""

from __future__ import annotations

# Importing the op modules registers the custom ops.
import executorch.backends.mlx.custom_kernel_ops.gguf_embedding  # noqa: F401
import executorch.backends.mlx.custom_kernel_ops.gguf_linear  # noqa: F401

import torch
import torch.nn as nn
from executorch.backends.mlx.custom_kernel_ops.gguf_linear import Q6K_BLOCK_BYTES, QK_K


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


class GGUFEmbedding(nn.Module):
    """``y = gguf_embedding(weight_blob, indices, format)`` for a GGUF table.

    The weight is the **raw** GGUF block blob, stored as a uint8 buffer of shape
    ``(num_embeddings, n_blocks * block_bytes)``. ``forward`` returns bfloat16,
    matching the model's embedding dtype.
    """

    def __init__(self, weight_blob: torch.Tensor, format: str = "q6k"):
        super().__init__()
        if weight_blob.dim() != 2 or weight_blob.dtype != torch.uint8:
            raise ValueError(
                f"GGUFEmbedding: weight_blob must be 2-D uint8; got "
                f"shape {tuple(weight_blob.shape)} dtype {weight_blob.dtype}"
            )
        if format != "q6k":
            raise NotImplementedError(
                f"GGUFEmbedding: unsupported format {format!r}; only 'q6k' supported"
            )
        row_bytes = int(weight_blob.shape[1])
        if row_bytes % Q6K_BLOCK_BYTES != 0:
            raise ValueError(
                f"GGUFEmbedding: weight row bytes {row_bytes} must be a multiple of "
                f"{Q6K_BLOCK_BYTES}"
            )
        self.format = format
        self.num_embeddings = int(weight_blob.shape[0])
        self.embedding_dim = (row_bytes // Q6K_BLOCK_BYTES) * QK_K
        self.register_buffer("weight", weight_blob)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.mlx.gguf_embedding(self.weight, indices, self.format)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, format={self.format!r}"
        )


def replace_with_gguf_embedding(
    model: nn.Module,
    weight_fqn: str,
    weight_blob: torch.Tensor,
    format: str = "q6k",
) -> None:
    """Replace the ``nn.Embedding`` owning ``weight_fqn`` with a ``GGUFEmbedding``."""
    parts = weight_fqn.rsplit(".", 1)
    if len(parts) != 2 or parts[1] != "weight":
        raise ValueError(
            f"replace_with_gguf_embedding: expected a '*.weight' fqn; "
            f"got {weight_fqn!r}"
        )
    module_fqn = parts[0]
    grandparent_fqn, _, child_name = module_fqn.rpartition(".")
    grandparent = (
        model.get_submodule(grandparent_fqn) if grandparent_fqn else model
    )
    setattr(grandparent, child_name, GGUFEmbedding(weight_blob, format=format))
