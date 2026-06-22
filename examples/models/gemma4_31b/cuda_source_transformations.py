# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA source transformations for Gemma 4 31B-IT.

Currently only adds optional TurboQuant TQ4 KV cache compression for
full-attention layers, leaving sliding-window layers untouched. When
``use_turboquant=True`` is passed:

- ``Gemma4Attention.kv_cache`` is replaced with
  ``extension.llm.modules.turboquant.TurboQuantKVCache`` on every
  full-attention layer (sliding layers keep their ``RingKVCache``).
- The attention forward is monkey-patched to call
  ``torch.ops.triton.tq4_sdpa`` (the fused TQ4 attention kernel) instead
  of ``F.scaled_dot_product_attention``.

The model file (``model.py``) stays backend-agnostic — all CUDA
TurboQuant specifics live here.
"""

from __future__ import annotations

import types

# Importing this module registers ``torch.ops.triton.tq4_sdpa``.
import executorch.backends.cuda.triton.kernels.tq4_sdpa  # noqa: F401

import torch
import torch.nn as nn

from executorch.examples.models.gemma4.text_decoder import apply_rotary_emb
from executorch.extension.llm.modules.turboquant import TurboQuantKVCache


def _turboquant_attention_forward(
    self,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for ``Gemma4Attention.forward`` that uses
    ``torch.ops.triton.tq4_sdpa`` over a ``TurboQuantKVCache``.

    Mirrors the default forward up to (and including) RoPE; only the
    cache update and SDPA call differ.
    """
    B, T, _ = x.shape

    q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
    raw_k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
    if self.k_eq_v:
        raw_v = raw_k
    else:
        raw_v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(raw_k)
    v = self.v_norm(raw_v)

    # (B, H, T, D) for SDPA / KV cache.
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # RoPE: same code path as default forward.
    freqs = torch.outer(input_pos.float(), self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = torch.cos(emb)
    sin = torch.sin(emb)
    q, k = apply_rotary_emb(q, k, cos, sin)

    # Compress + write. Returns the full compressed cache tensors —
    # tq4_sdpa decompresses per tile in its inner loop, so the full
    # uncompressed K/V is never materialized.
    k_packed, k_norms, v_packed, v_norms = self.kv_cache.update(input_pos, k, v)

    # Number of valid (filled) KV positions = input_pos[0] + T. Passing this to
    # tq4_sdpa bounds its KV loop to the actual context instead of the full
    # pre-allocated buffer (max_seq_len for global layers), making attention
    # O(context) instead of O(max_seq_len). Kept as a GPU scalar (no ``.item()``)
    # so the bound is captured correctly by the decode CUDA graph. Decode: T=1 ->
    # input_pos+1; prefill chunk: T -> chunk_end.
    # NOTE: this call-site argument was dropped during a rebase, which silently
    # disabled the O(context) bound and forced a full max_seq_len sweep every
    # step (catastrophic at 128k: ~2.7 tok/s decode vs ~37+ when bounded).
    kv_len = input_pos[0] + input_pos.shape[0]

    # ``scale=self.scaling`` (= 1.0 for Gemma 4) — overrides tq4_sdpa's
    # default ``1/sqrt(D)`` because Gemma's QK-norm has absorbed the
    # 1/sqrt(d) factor into trained weights.
    y = torch.ops.triton.tq4_sdpa(
        q,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        self.kv_cache.centroids,
        self.kv_cache.rotation,
        attn_mask,
        False,  # is_causal — attn_mask already encodes causal masking
        self.scaling,
        kv_len,
    )

    y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
    return self.o_proj(y)


def cuda_source_transformations(
    model: nn.Module,
    *,
    use_turboquant: bool = False,
) -> None:
    """Apply CUDA source transformations to a Gemma 4 31B model in place.

    Args:
        model: ``Gemma4_31B`` instance to transform.
        use_turboquant: When True, swap full-attention layers' KV caches
            for the backend-agnostic ``TurboQuantKVCache`` (~3.8× cache
            memory savings) and route their SDPA through
            ``torch.ops.triton.tq4_sdpa``. Sliding-window layers are
            unaffected.
    """
    if not use_turboquant:
        return

    config = model.config
    n_swapped = 0
    for layer in model.layers:
        attn = layer.self_attn
        if attn.is_sliding:
            continue
        attn.kv_cache = TurboQuantKVCache(
            n_heads=attn.n_kv_heads,
            head_dim=attn.head_dim,
            max_seq_len=config.max_seq_len,
        )
        attn.forward = types.MethodType(_turboquant_attention_forward, attn)
        n_swapped += 1

    print(
        f"[gemma4_31b cuda] TurboQuant: swapped {n_swapped} full-attention "
        f"KV caches with TurboQuantKVCache (TQ4)"
    )
