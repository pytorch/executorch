# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TurboQuant KV cache compression for torch.export(strict=True).

Compresses KV cache to TQ4 nibble-packed format (3.8x memory savings)
using the TurboQuant algorithm (arXiv 2504.19874, ICLR 2026). The
codebook and rotation matrix are precomputed at init time; the forward
path is pure PyTorch ops.

Paired with the fused ``triton::tq4_sdpa`` kernel, attention runs
directly on compressed data — the full decompressed cache is never
materialized.

Usage::

    from executorch.extension.llm.modules.turboquant import TurboQuantKVCache

    # Replace KV cache in attention module, then set the flag:
    attn.kv_cache = TurboQuantKVCache(n_heads, head_dim, max_seq_len)
    attn.turboquant = True

See ``examples/models/qwen3_5_moe/export.py`` for a full integration
example with model-specific replacement logic.
"""

import torch
import torch.nn as nn

from executorch.extension.llm.modules.turboquant.codebook import (
    generate_rotation_matrix,
    solve_lloyd_max,
)


class TurboQuantKVCache(nn.Module):
    """KV cache with TQ4 compression.

    Stores K/V as nibble-packed uint8 indices (2 indices per byte) plus
    bf16 per-vector norms. The ``update()`` method compresses incoming
    K/V and returns the compressed cache buffers for use with the fused
    ``triton::tq4_sdpa`` kernel. A ``_decompress()`` method is provided
    for testing.

    Args:
        n_heads: Number of KV heads.
        head_dim: Dimension per head (must be even).
        max_seq_len: Maximum sequence length (cache is pre-allocated).
        bits: Quantization bits per coordinate (must be 4).
        seed: Random seed for the rotation matrix.

    Note:
        Batch size is fixed to 1 (standard for ExecuTorch inference).
        Input tensors must have shape ``(1, H, T, D)``.
    """

    def __init__(self, n_heads, head_dim, max_seq_len, bits=4, seed=42):
        super().__init__()
        if bits != 4:
            raise ValueError(
                f"Only 4-bit quantization is supported (nibble packing + "
                f"16-entry codebook). Got bits={bits}."
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2

        centroids, boundaries = solve_lloyd_max(head_dim, bits)
        rotation = generate_rotation_matrix(head_dim, seed=seed)

        self.register_buffer("centroids", centroids)
        self.register_buffer("boundaries", boundaries.to(torch.bfloat16))
        self.register_buffer("rotation", rotation)
        self.register_buffer("rotation_T", rotation.T.to(torch.bfloat16).contiguous())

        # Compressed cache buffers
        self.register_buffer(
            "k_packed",
            torch.zeros(1, n_heads, max_seq_len, self.half_dim, dtype=torch.uint8),
        )
        self.register_buffer(
            "k_norms",
            torch.zeros(1, n_heads, max_seq_len, 1, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "v_packed",
            torch.zeros(1, n_heads, max_seq_len, self.half_dim, dtype=torch.uint8),
        )
        self.register_buffer(
            "v_norms",
            torch.zeros(1, n_heads, max_seq_len, 1, dtype=torch.bfloat16),
        )

    def _compress(self, x):
        """Compress ``(1, H, T, D)`` tensor to nibble-packed uint8 + bf16 norms.

        All ops are torch.export-compatible: norm, matmul, bucketize, bitwise.
        Stays in bf16 throughout — TQ4 quantization error dominates bf16 rounding.
        """
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_dim).to(self.rotation_T.dtype)

        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ self.rotation_T
        indices = torch.bucketize(rotated, self.boundaries)

        idx_u8 = indices.to(torch.uint8)
        packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

        return (
            packed.reshape(*orig_shape[:-1], self.half_dim),
            norms.reshape(*orig_shape[:-1], 1).to(torch.bfloat16),
        )

    def _decompress(self, packed, norms):
        """Decompress nibble-packed uint8 + norms back to float tensor.

        Provided for testing — the fused ``tq4_sdpa`` kernel decompresses
        per-tile in the attention inner loop, never calling this method.
        """
        orig_batch_shape = packed.shape[:-1]
        flat_packed = packed.reshape(-1, self.half_dim)
        flat_norms = norms.reshape(-1, 1).float()

        high = (flat_packed >> 4).long()
        low = (flat_packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(-1, self.head_dim)

        reconstructed = self.centroids.float()[indices]
        unrotated = reconstructed @ self.rotation_T.float().T
        scaled = unrotated * flat_norms

        return scaled.reshape(*orig_batch_shape, self.head_dim)

    def forward(self, input_pos, k_val, v_val):
        return self.update(input_pos, k_val, v_val)

    def update(self, input_pos, k_val, v_val):
        """Compress and store K/V, return compressed cache buffers.

        Args:
            input_pos: ``(T,)`` position indices.
            k_val: ``(1, H, T, D)`` key tensor (batch size must be 1).
            v_val: ``(1, H, T, D)`` value tensor (batch size must be 1).

        Returns:
            Tuple of ``(k_packed, k_norms, v_packed, v_norms)`` — the full
            compressed cache (all positions, not just the new tokens).
        """
        k_packed, k_norms = self._compress(k_val)
        v_packed, v_norms = self._compress(v_val)

        self.k_packed[:, :, input_pos] = k_packed
        self.k_norms[:, :, input_pos] = k_norms
        self.v_packed[:, :, input_pos] = v_packed
        self.v_norms[:, :, input_pos] = v_norms

        return self.k_packed, self.k_norms, self.v_packed, self.v_norms
