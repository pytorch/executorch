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

# Importing this module registers ``torch.ops.triton.sdpa`` /
# ``torch.ops.triton.sdpa_decode_splitk`` (the length-aware bf16 attention ops
# used by the non-TurboQuant full-attention path below).
import executorch.backends.cuda.triton.kernels.sdpa  # noqa: F401

# Importing this module registers ``torch.ops.triton.tq4_sdpa``.
import executorch.backends.cuda.triton.kernels.tq4_sdpa  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        False,  # is_causal: attn_mask already encodes causal masking
        self.scaling,
        kv_len,
        True,  # mask_is_causal: Gemma full-attention mask is standard causal
    )

    y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
    return self.o_proj(y)


def _lenaware_attention_forward(
    self,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Drop-in ``Gemma4Attention.forward`` for full-attention layers on the
    non-TurboQuant CUDA path that bounds SDPA to the valid context length.

    Identical to the default forward (plain bf16 KV cache) except the final
    ``F.scaled_dot_product_attention`` is replaced with
    ``torch.ops.triton.sdpa(..., kv_len=...)``. Passing ``kv_len`` bounds the
    attention KV loop to the actual filled context instead of the full
    pre-allocated buffer (``max_seq_len`` for global layers), making decode
    O(context) instead of O(max_seq_len) — and routes L_q==1 decode through the
    length-aware split-K flash-decoding kernel. Sliding-window layers are not
    patched (they already use a bounded ring buffer).
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

    # Update cache and read back the full (pre-allocated) K/V buffers.
    k, v = self.kv_cache.update(input_pos, k, v)

    # Number of valid (filled) KV positions = input_pos[0] + T. Passing this to
    # sdpa bounds its KV loop to the actual context instead of the full
    # pre-allocated buffer (max_seq_len for global layers), making attention
    # O(context) instead of O(max_seq_len). Kept as a GPU scalar (no ``.item()``)
    # so the bound is captured correctly by the decode CUDA graph. Decode: T=1 ->
    # input_pos+1; prefill chunk: T -> chunk_end.
    kv_len = input_pos[0] + input_pos.shape[0]

    # ``scale=self.scaling`` (= 1.0 for Gemma 4) — Gemma's QK-norm has absorbed
    # the 1/sqrt(d) factor into trained weights. ``enable_gqa=True`` lets the
    # kernel handle the head ratio without materializing expanded K/V.
    y = torch.ops.triton.sdpa(
        q,
        k,
        v,
        attn_mask,
        0.0,  # dropout_p
        False,  # is_causal: attn_mask already encodes causal masking
        self.scaling,
        True,  # enable_gqa
        kv_len,
    )

    y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
    return self.o_proj(y)


def _fused_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in ``Gemma4MLP.forward`` over a fused gate|up projection.

    Identical math to ``down(gelu(gate(x)) * up(x))``: the single
    ``gate_up_proj`` emits ``[gate | up]`` concatenated on the last dim,
    which is then split. One W4A8 matmul (and one activation-quant of ``x``)
    instead of two.
    """
    h = self.gate_up_proj(x)
    gate = h[..., : self.intermediate_size]
    up = h[..., self.intermediate_size :]
    return self.down_proj(F.gelu(gate, approximate="tanh") * up)


def _concat_coalesced_int4_along_n(a, b):
    """Concatenate two ``CudaCoalescedInt4Tensor`` along the output (N) dim.

    qdata is ``[N, K/2]`` and scale/zero_point are ``[N, n_groups]`` in the
    coalesced layout, so a per-output-row concat on dim 0 is exact: the W4A8
    dp4a matvec reads each output row's qdata/scale/zero independently, so
    out[:N_a] reproduces ``a`` and out[N_a:] reproduces ``b`` bit-for-bit.
    """
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor

    return CudaCoalescedInt4Tensor(
        torch.cat([a.qdata, b.qdata], dim=0),
        torch.cat([a.scale, b.scale], dim=0),
        torch.cat([a.zero_point, b.zero_point], dim=0),
        a.block_size,
        torch.Size([a.shape[0] + b.shape[0], a.shape[1]]),
        None,
        a.activation_dtype,
    )


def _is_fuseable_int4_pair(gate_w, up_w) -> bool:
    """True iff gate/up are both coalesced-int4 with matching K + block_size.

    Q4_K MLP weights become ``CudaCoalescedInt4Tensor`` (fuseable); a Q6_K
    weight becomes ``CudaDp4aPlanarInt6Tensor`` (left alone). ``act_pre_scale``
    is unused on this path but we require it absent so the concat stays exact.
    """
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor

    return (
        isinstance(gate_w, CudaCoalescedInt4Tensor)
        and isinstance(up_w, CudaCoalescedInt4Tensor)
        and list(gate_w.block_size) == list(up_w.block_size)
        and gate_w.shape[1] == up_w.shape[1]
        and gate_w.act_pre_scale is None
        and up_w.act_pre_scale is None
    )


def _fuse_gate_up_proj(model: nn.Module) -> None:
    """Fuse each MLP's ``gate_proj | up_proj`` into one ``gate_up_proj``.

    gate and up share the same input, so the unfused path quantizes ``x`` to
    int8 twice and launches two W4A8 matvecs per layer. Fusing the weights
    into one ``[2*inter, hidden]`` tensor halves both. Weight bytes read are
    unchanged, so the win is launch + activation-quant overhead (decode is
    launch-bound). Only Q4_K (coalesced-int4) layers are fused; any layer
    with a non-int4 weight is left as two matmuls (still correct).

    Must run AFTER weights are packed to ``CudaCoalescedInt4Tensor`` (i.e.
    inside ``_export_cuda``), and is independent of TurboQuant.
    """
    n_fused = 0
    n_skipped = 0
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")):
            continue
        gate_w = mlp.gate_proj.weight
        up_w = mlp.up_proj.weight
        if not _is_fuseable_int4_pair(gate_w, up_w):
            n_skipped += 1
            continue
        inter = up_w.shape[0]
        hidden = up_w.shape[1]
        fused_w = _concat_coalesced_int4_along_n(gate_w, up_w)

        # Container built on meta to avoid materializing a dense
        # [2*inter, hidden] weight before we overwrite it with fused_w.
        gate_up = nn.Linear(hidden, 2 * inter, bias=False, device="meta")
        gate_up.weight = nn.Parameter(fused_w, requires_grad=False)
        mlp.gate_up_proj = gate_up
        mlp.intermediate_size = inter
        del mlp.gate_proj
        del mlp.up_proj
        mlp.forward = types.MethodType(_fused_mlp_forward, mlp)
        n_fused += 1

    msg = f"[gemma4_31b cuda] Fused gate+up on {n_fused} MLP layers"
    if n_skipped:
        msg += f" ({n_skipped} skipped: non-int4 weights)"
    print(msg)


def cuda_source_transformations(
    model: nn.Module,
    *,
    use_turboquant: bool = False,
) -> None:
    """Apply CUDA source transformations to a Gemma 4 31B model in place.

    Always fuses each MLP's ``gate_proj|up_proj`` into a single matmul (one
    activation-quant + one W4A8 matvec per layer instead of two; Q4_K
    coalesced-int4 layers only — other quant types are left untouched).
    Optionally also swaps full-attention KV caches for TurboQuant TQ4.

    Args:
        model: ``Gemma4_31B`` instance to transform.
        use_turboquant: When True, swap full-attention layers' KV caches
            for the backend-agnostic ``TurboQuantKVCache`` (~3.8× cache
            memory savings) and route their SDPA through
            ``torch.ops.triton.tq4_sdpa``. Sliding-window layers are
            unaffected.
    """
    _fuse_gate_up_proj(model)

    if not use_turboquant:
        # Non-TurboQuant path: keep the bf16 KV cache but bound full-attention
        # SDPA to the valid context length via a runtime kv_len scalar (routes
        # through torch.ops.triton.sdpa, which dispatches L_q==1 decode to the
        # length-aware split-K flash-decoding kernel). Sliding-window layers
        # already use a bounded ring buffer, so they are left untouched.
        n_bounded = 0
        for layer in model.layers:
            attn = layer.self_attn
            if attn.is_sliding:
                continue
            attn.forward = types.MethodType(_lenaware_attention_forward, attn)
            n_bounded += 1
        print(
            f"[gemma4_31b cuda] length-aware SDPA: bounded {n_bounded} "
            f"full-attention layers to runtime kv_len (O(context) attention)"
        )
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
