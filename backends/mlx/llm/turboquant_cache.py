#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TurboQuant TQ4 KV cache for the MLX backend.

Subclass of the backend-agnostic
``extension/llm/modules/turboquant/kv_cache.py::TurboQuantKVCache``.

The cache stores K and V in **rotated space** (post-multiplied by R^T)
as nibble-packed uint8 codebook indices plus per-vector bf16 norms.
SDPA runs in rotated space and undoes the rotation on the output side
(both Q and output rotations are ``T_q × D²``, much smaller than
applying the inverse rotation to K/V which would be ``T_kv × D²``).

Reference:
    TurboQuant: Online Vector Quantization with Near-optimal
    Distortion Rate. arXiv:2504.19874 (ICLR 2026).
"""

from typing import Optional, Tuple

# Register the MLX custom ops used by this cache.
import executorch.backends.mlx.custom_ops  # noqa: F401  mlx::custom_sdpa, mlx::kv_cache_update
import executorch.backends.mlx.model_ops.tq4_compress  # noqa: F401  mlx::tq4_compress
import executorch.backends.mlx.model_ops.tq_dequant  # noqa: F401  mlx::tq_dequant
import executorch.backends.mlx.model_ops.tq_norm  # noqa: F401  mlx::tq_norm

import torch

from executorch.extension.llm.modules.turboquant.kv_cache import (
    TurboQuantKVCache as _SharedTurboQuantKVCache,
)


class TurboQuantKVCache(_SharedTurboQuantKVCache):
    """
    TurboQuant TQ4 KV cache, MLX-backend variant.

    Drop-in replacement for ``backends/mlx/llm/cache.py::KVCache``.

    Args:
        max_batch_size: Must be 1 (TQ4 is batch=1 only).
        max_context_length: Maximum sequence length.
        n_heads: Number of KV heads.
        head_dim: Per-head dimension. Must be even and a multiple of 64.
        enable_dynamic_shape: Accepted for interface parity; ignored.
        dtype: Compute dtype (bf16). Used for pre-cast buffers.
        bits: Quantization bits (must be 4).
        seed: RNG seed for the orthogonal rotation matrix.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool,
        dtype: torch.dtype = torch.bfloat16,
        bits: int = 4,
        seed: int = 42,
    ):
        if max_batch_size != 1:
            raise ValueError(
                f"TurboQuantKVCache only supports max_batch_size=1, "
                f"got {max_batch_size}"
            )
        if bits != 4:
            raise ValueError(
                f"TurboQuantKVCache only supports bits=4 "
                f"(16-entry codebook), got bits={bits}"
            )
        # MLX-backend Metal kernels need ``head_dim % 64 == 0``: ``tq_norm``
        # uses 32 SIMD lanes (so D must be a multiple of 32), and
        # ``tq_dequant`` packs 2 dims per byte across 32 lanes (so D must
        # be a multiple of 64). Take the stricter constraint here.
        if head_dim % 64 != 0:
            raise ValueError(
                f"TurboQuantKVCache requires head_dim to be "
                f"a multiple of 64 (Metal SIMD + 4-bit pack constraint), "
                f"got {head_dim}"
            )
        super().__init__(
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=max_context_length,
            bits=bits,
            seed=seed,
        )
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.enable_dynamic_shape = enable_dynamic_shape

        # Replace parent's fp32 ``rotation`` and ``centroids`` buffers
        # with compute-dtype versions in-place. Avoids a per-call
        # ``_to_copy`` cast in the lowered graph at every use site.
        # Parent's ``_decompress`` (testing-only) is the sole consumer
        # of these as fp32 and is not called at runtime.
        self.register_buffer(
            "rotation",
            self.rotation.to(dtype).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "centroids",
            self.centroids.to(dtype).contiguous(),
            persistent=False,
        )
        # Pre-cast eps for the divide-by-zero guard in _compress.
        self.register_buffer(
            "norm_eps",
            torch.tensor(1e-10, dtype=dtype),
            persistent=False,
        )

    def _compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress ``(1, H, T, D)`` → packed ``(1, H, T, D//2)`` u8 +
        norms ``(1, H, T, 1)`` bf16.

        The L2-norm reduction uses ``mlx::tq_norm`` (one Metal kernel
        with fp32 sum-of-squares in registers via ``simd_sum``); the
        bucketize + nibble-pack tail uses ``mlx::tq4_compress`` (one
        Metal kernel for both steps).
        """
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_dim)

        norms = torch.ops.mlx.tq_norm(flat)
        normalized = flat / (norms + self.norm_eps)
        rotated = normalized @ self.rotation_T
        packed = torch.ops.mlx.tq4_compress(rotated, self.boundaries)

        return (
            packed.reshape(*orig_shape[:-1], self.half_dim),
            norms.reshape(*orig_shape[:-1], 1),
        )

    def update(
        self,
        input_pos,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress + write K/V at ``input_pos``, return the full
        compressed cache buffers.

        Accepts ``input_pos`` as either a ``(T,)`` LongTensor of
        positions or a Python int / SymInt ``start_pos``. Writes go
        through ``mlx::kv_cache_update`` (matching the non-TQ
        ``MLXKVCache`` path) which lowers to a tighter in-place
        scatter than ``index_copy_`` would.
        """
        if isinstance(input_pos, torch.Tensor):
            start_pos = input_pos[0].item()
            seq_len = k_val.size(2)
            torch._check(seq_len == v_val.size(2))
            torch._check(start_pos >= 0)
            torch._check(start_pos + seq_len <= self.max_context_length)
        else:
            start_pos = input_pos

        k_packed, k_norms = self._compress(k_val)
        v_packed, v_norms = self._compress(v_val)

        torch.ops.mlx.kv_cache_update(self.k_packed, k_packed, start_pos)
        torch.ops.mlx.kv_cache_update(self.k_norms, k_norms, start_pos)
        torch.ops.mlx.kv_cache_update(self.v_packed, v_packed, start_pos)
        torch.ops.mlx.kv_cache_update(self.v_norms, v_norms, start_pos)

        # Slices on the return create new graph nodes so the same node
        # is not both BUFFER_MUTATION and USER_OUTPUT.
        return (
            self.k_packed[:, :, :, :],
            self.k_norms[:, :, :, :],
            self.v_packed[:, :, :, :],
            self.v_norms[:, :, :, :],
        )

    # forward() is inherited from the parent (delegates to update).

    def sdpa(
        self,
        query: torch.Tensor,
        start_pos,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """SDPA over the compressed cache.

        Runs attention in rotated space:
          1. Q_rot = Q @ R^T               (T_q x D^2)
          2. K_rot, V_rot = tq_dequant(...) (rotated-space K/V)
          3. out_rot = custom_sdpa(Q_rot, K_rot, V_rot, ...)
          4. out = out_rot @ R              (T_q x D^2)

        Since R is orthogonal, score = (Q·R^T)·(K·R^T)^T = Q·K^T, so
        attention is invariant under matched rotation of Q and K. The
        ``T_kv x D^2`` inverse-rotation matmul on K/V is replaced with
        two ``T_q x D^2`` matmuls (Q and output).

        Args:
            query: ``(B, H_q, T_q, D)`` bf16.
            start_pos: int or SymInt — absolute position of the first
                query token.
            scale: 1/sqrt(D) if None.

        Returns:
            ``(B, H_q, T_q, D)`` bf16 attention output, in original
            (un-rotated) space.
        """
        seq_len = query.size(2)
        end_pos = start_pos + seq_len
        torch._check(start_pos >= 0)
        torch._check(end_pos <= self.max_context_length)

        q_rot = query @ self.rotation_T

        k_packed_live = self.k_packed[:, :, :end_pos, :]
        k_norms_live = self.k_norms[:, :, :end_pos, :]
        v_packed_live = self.v_packed[:, :, :end_pos, :]
        v_norms_live = self.v_norms[:, :, :end_pos, :]

        # TODO: optimize with a fused dequant + SDPA
        k_rot = torch.ops.mlx.tq_dequant(k_packed_live, k_norms_live, self.centroids)
        v_rot = torch.ops.mlx.tq_dequant(v_packed_live, v_norms_live, self.centroids)

        out_rot = torch.ops.mlx.custom_sdpa(
            q_rot,
            k_rot,
            v_rot,
            start_pos,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            scale,
        )

        return out_rot @ self.rotation
