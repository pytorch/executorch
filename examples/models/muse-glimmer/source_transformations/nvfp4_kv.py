# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 KV cache for the CUDA Muse Glimmer export.

Replaces every layer's BF16 ``RingKVCache`` / ``FlatKVCache`` with an NVFP4
cache (see ``backends/cuda/triton/kernels/nvfp4_kv.py`` for the layout) and
swaps the attention forward so that

  * K-norm + RoPE + NVFP4 quantization + cache store run as ONE fused kernel
    (``triton::nvfp4_kv_write``) straight from the K/V projection outputs
    (works for NVFP4 and BF16 projections alike: it only sees their outputs);
  * attention reads the NVFP4 cache directly (``triton::nvfp4_sdpa``): split-K
    decode, small-query split-K, chunked prefill; global (causal, kv_len
    bounded) and sliding-window layers. K and V are decoded exactly to FP16
    in registers; both products run on FP16 MMAs with FP32 accumulation.

Sliding-window layers keep the model's ring geometry (``RingKVCache``: slot
``pos % capacity``); the kernel derives the dense ring mask's visibility from
the absolute positions, so the ring capacity is the only geometry parameter.

Enabled by ``export_solo.py --kv-cache-dtype nvfp4``
(``cuda_source_transformations(use_nvfp4_kv_cache=True)``).
"""

from __future__ import annotations

import types

import executorch.backends.cuda.triton.kernels.nvfp4_kv  # noqa: F401  (registers ops)

import torch
import torch.nn as nn
from executorch.examples.models.muse_glimmer.model.model import rotate_interleaved
from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
    _project_qkvo,
)


class NVFP4KVCache(nn.Module):
    """K/V cache stored as NVFP4 (E2M1 codes, E4M3 K16 scales, unit global scale)."""

    def __init__(self, n_kv_heads: int, length: int, head_dim: int, ring: bool, device):
        super().__init__()
        assert head_dim % 32 == 0 and head_dim & (head_dim - 1) == 0
        self.ring = ring
        for kind in ("k", "v"):
            self.register_buffer(
                f"{kind}_q",
                torch.zeros(
                    (1, n_kv_heads, length, head_dim // 2),
                    dtype=torch.uint8,
                    device=device,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"{kind}_s",
                torch.zeros(
                    (1, n_kv_heads, length, head_dim // 16),
                    dtype=torch.uint8,
                    device=device,
                ),
                persistent=False,
            )


def _nvfp4_kv_attention_forward(
    self, x: torch.Tensor, input_pos: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    B, T, _ = x.shape
    h = self.qkv_proj_norm(x)
    xq, xk, xv, og = _project_qkvo(self, h)
    xq = xq.view(B, T, self.n_heads, self.head_dim)
    xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
    xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

    if self.q_norm is not None:
        xq = self.q_norm(xq)
    xq = xq.transpose(1, 2)
    if self.use_rope:
        freqs = torch.outer(input_pos.float(), self.inv_freq)
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        xq = rotate_interleaved(xq, cos, sin)
    xq = xq.to(torch.bfloat16)

    c = self.kv_cache
    # Fused: K RMS-norm (scaleless QK-norm) + RoPE + NVFP4 quantize + store.
    torch.ops.triton.nvfp4_kv_write(
        xk,
        xv,
        input_pos,
        c.k_q,
        c.k_s,
        c.v_q,
        c.v_s,
        self.inv_freq if self.use_rope else None,
        c.ring,
        self.k_norm.eps if self.k_norm is not None else None,
    )
    # Absolute number of tokens written after this step (CUDA-graph safe).
    kv_len = input_pos[0] + input_pos.shape[0]
    if self.is_sliding:
        # Same visibility as the model's dense ring mask, computed in-kernel.
        y = torch.ops.triton.nvfp4_sdpa(
            xq,
            c.k_q,
            c.k_s,
            c.v_q,
            c.v_s,
            None,
            False,
            self.attn_scale,
            kv_len,
            self.window_size,
        )
    else:
        y = torch.ops.triton.nvfp4_sdpa(
            xq,
            c.k_q,
            c.k_s,
            c.v_q,
            c.v_s,
            None,
            True,
            self.attn_scale,
            kv_len,
        )
    y = y.transpose(1, 2).contiguous()
    if self.use_o_gate and og is not None:
        og = og.view(B, T, self.n_heads, self.head_dim)
        y = torch.sigmoid(og) * y
    return self.o_proj(y.reshape(B, T, -1))


def install_nvfp4_kv_cache(model: nn.Module) -> None:
    """Swap every layer to the NVFP4 KV cache (same lengths as the BF16 caches)."""
    for layer in model.layers:
        attn = layer.self_attn
        old = attn.kv_cache
        device = old.k_cache.device if old.k_cache.device.type != "meta" else "cpu"
        attn.kv_cache = NVFP4KVCache(
            attn.n_kv_heads,
            old.k_cache.shape[2],
            attn.head_dim,
            attn.is_sliding,
            device,
        )
        attn.forward = types.MethodType(_nvfp4_kv_attention_forward, attn)
    print(f"[muse_glimmer cuda] NVFP4 KV cache: {len(model.layers)} layers")
