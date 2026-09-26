# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA source transformations for Muse Glimmer.

Global NoPE layers use length-bounded Triton attention, optionally with a
TurboQuant KV cache. Sliding-window RoPE layers retain their ring buffers.
"""

from __future__ import annotations

import types

# Register the length-aware global-attention operator.
import executorch.backends.cuda.triton.kernels.sdpa  # noqa: F401

# Register the TurboQuant attention operator.
import executorch.backends.cuda.triton.kernels.tq4_sdpa  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from executorch.examples.models.muse_glimmer.model.dflash_model import (
    build_dflash_swa_mask,
    rotate_half,
)
from executorch.examples.models.muse_glimmer.model.model import FlatKVCache, RingKVCache
from executorch.examples.models.muse_glimmer.source_transformations.sampler import (
    sample,
)
from executorch.extension.llm.modules.turboquant import TurboQuantKVCache
from torch.library import triton_op, wrap_triton


def add_dflash_hidden_tapping(model: nn.Module, target_layer_ids: list[int]) -> None:
    """Make a CUDA Muse Glimmer target return logits and DFlash tapped hidden states."""
    target_set = set(target_layer_ids)
    model._dflash_target_layer_ids = target_layer_ids

    def _run_embeddings_with_hidden(
        self, x: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sliding_mask, global_mask = self._build_masks(input_pos)
        tapped = []
        for i, layer in enumerate(self.layers):
            if i in target_set:
                tapped.append(x)
            x = layer(x, input_pos, sliding_mask, global_mask)

        return self.output_norm(x), torch.cat(tapped, dim=-1)

    def _run_with_hidden(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_embeddings_with_hidden(self, self.embed_text(tokens), input_pos)

    def _forward_with_hidden(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, tapped = _run_with_hidden(self, tokens, input_pos)
        return self._soft_cap(self.lm_head(x)), tapped

    def _prefill_with_hidden(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, tapped = _run_with_hidden(self, tokens, input_pos)
        return self._soft_cap(self.lm_head(x[:, -1:, :])), tapped

    def _forward_embeddings_with_hidden(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, tapped = _run_embeddings_with_hidden(self, inputs_embeds, input_pos)
        return self._soft_cap(self.lm_head(x)), tapped

    def _prefill_embeddings_with_hidden(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, tapped = _run_embeddings_with_hidden(self, inputs_embeds, input_pos)
        return self._soft_cap(self.lm_head(x[:, -1:, :])), tapped

    model.forward = types.MethodType(_forward_with_hidden, model)
    model.dflash_prefill_forward = types.MethodType(_prefill_with_hidden, model)
    model.target_forward_from_embeddings = types.MethodType(
        _forward_embeddings_with_hidden, model
    )
    model.target_prefill_from_embeddings = types.MethodType(
        _prefill_embeddings_with_hidden, model
    )


class CUDADFlashAttention(nn.Module):
    """CUDA-optimized DFlash cross-attention (flat KV cache + length-aware sdpa).

    Full-module replacement for ``DFlashAttention`` on the CUDA export path,
    mirroring how ``MLXDFlashAttention`` replaces it for MLX. ``dflash_model.py``
    stays backend-agnostic; all CUDA specifics live here:

    - Global (non-sliding) layers write the transient draft block into a flat KV
      cache with block-sized scratch, then bound attention to the valid context
      via a runtime ``kv_len`` GPU scalar routed through
      ``torch.ops.triton.sdpa`` (fixed shapes -> CUDA-graph capturable).
    - Sliding-window layers keep the concat + boolean-mask
      ``F.scaled_dot_product_attention`` path over a ring buffer.
    """

    @classmethod
    def from_attention(cls, attn: nn.Module, block_size: int) -> "CUDADFlashAttention":
        m = cls.__new__(cls)
        nn.Module.__init__(m)

        m.n_heads = attn.n_heads
        m.n_kv_heads = attn.n_kv_heads
        m.head_dim = attn.head_dim
        m.attn_scale = attn.attn_scale
        m.rope_theta = attn.rope_theta
        m.is_sliding = attn.is_sliding
        m.window_size = attn.window_size

        m.q_proj = attn.q_proj
        m.k_proj = attn.k_proj
        m.v_proj = attn.v_proj
        m.o_proj = attn.o_proj
        m.q_norm = attn.q_norm
        m.k_norm = attn.k_norm

        # Own KV cache. Buffers are created on meta and materialized to the
        # activation dtype later by materialize_dflash_runtime_buffers. Global
        # layers get block_size scratch positions so the transient draft block
        # can be written just past the valid context.
        with torch.device("meta"):
            if attn.is_sliding:
                m.kv_cache = RingKVCache(
                    max_batch_size=1,
                    window_size=attn.window_size,
                    num_kv_heads=attn.n_kv_heads,
                    head_dim=attn.head_dim,
                )
            else:
                ctx_len = attn.kv_cache.k_cache.shape[2]
                m.kv_cache = FlatKVCache(
                    max_batch_size=1,
                    max_seq_len=ctx_len + block_size,
                    num_kv_heads=attn.n_kv_heads,
                    head_dim=attn.head_dim,
                )
        return m

    def forward(
        self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        ctx_cos: torch.Tensor,
        ctx_sin: torch.Tensor,
        block_cos: torch.Tensor,
        block_sin: torch.Tensor,
        input_pos: torch.Tensor,
        valid_ctx_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        n_new = target_hidden.shape[1]

        # Project new context K/V, apply k_norm, RoPE, then cache.
        k_new = self.k_proj(target_hidden).view(
            B, n_new, self.n_kv_heads, self.head_dim
        )
        v_new = self.v_proj(target_hidden).view(
            B, n_new, self.n_kv_heads, self.head_dim
        )
        k_new = self.k_norm(k_new)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)
        k_new = rotate_half(k_new, ctx_cos, ctx_sin)
        positions = torch.arange(n_new, device=k_new.device) + input_pos[0]
        k_cached, v_cached = self.kv_cache.update(positions, k_new, v_new)

        total_ctx = input_pos[0] + (
            n_new if valid_ctx_len is None else valid_ctx_len[0]
        )

        if self.is_sliding:
            attn_mask = build_dflash_swa_mask(
                total_ctx,
                T,
                self.kv_cache.buf_size,
                self.window_size,
            )
            k_ctx = k_cached
            v_ctx = v_cached
        else:
            attn_mask = None

        # Block K/V — computed fresh each call.
        k_blk = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v_blk = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        k_blk = self.k_norm(k_blk)
        k_blk = k_blk.transpose(1, 2)
        v_blk = v_blk.transpose(1, 2)
        k_blk = rotate_half(k_blk, block_cos, block_sin)

        # Q from block tokens.
        xq = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = xq.transpose(1, 2)
        xq = rotate_half(xq, block_cos, block_sin)

        if self.is_sliding:
            k = torch.cat([k_ctx, k_blk], dim=2)
            v = torch.cat([v_ctx, v_blk], dim=2)
            # PyTorch SDPA broadcasts the singleton query dimension, while the
            # CUDA Triton lowering requires the explicit [B, 1, T, K] shape.
            attn_mask = attn_mask.expand(B, 1, T, -1)
            y = F.scaled_dot_product_attention(
                xq,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                enable_gqa=True,
                scale=self.attn_scale,
            )
        else:
            # Store the transient block immediately after valid context; the next
            # call overwrites it with accepted context/block data. kv_len
            # (total_ctx + T) is a runtime GPU scalar, so the KV loop stays
            # O(context) and shapes remain CUDA-graph capturable.
            block_positions = torch.arange(T, device=k_blk.device) + total_ctx
            k, v = self.kv_cache.update(block_positions, k_blk, v_blk)
            y = torch.ops.triton.sdpa(
                xq,
                k,
                v,
                None,
                0.0,
                False,
                self.attn_scale,
                True,
                total_ctx + T,
            )
        y = y.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(y)


def _replace_dflash_cuda_layer_forward(layer: nn.Module) -> None:
    """Replace DFlashDecoderLayer.forward to pass precomputed ctx/block RoPE."""

    def _cuda_dflash_layer_forward(
        self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        ctx_cos: torch.Tensor,
        ctx_sin: torch.Tensor,
        block_cos: torch.Tensor,
        block_sin: torch.Tensor,
        input_pos: torch.Tensor,
        valid_ctx_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            target_hidden,
            ctx_cos,
            ctx_sin,
            block_cos,
            block_sin,
            input_pos,
            valid_ctx_len,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    layer.forward = types.MethodType(_cuda_dflash_layer_forward, layer)


def _replace_dflash_cuda_model_forward(model: nn.Module) -> None:
    """Replace DFlashDraftModel.forward to precompute ctx/block RoPE (no .item())."""

    def _cuda_dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
        valid_ctx_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = self.hidden_norm(self.fc(target_hidden))
        n_new = ctx.shape[1]
        valid_new = n_new if valid_ctx_len is None else valid_ctx_len[0]
        T = noise_embedding.shape[1]
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=input_pos.device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )
        ctx_positions = (
            torch.arange(n_new, device=input_pos.device, dtype=torch.float32)
            + input_pos[0]
        )
        block_positions = (
            torch.arange(T, device=input_pos.device, dtype=torch.float32)
            + input_pos[0]
            + valid_new
        )
        ctx_freqs = torch.outer(ctx_positions, inv_freq)
        block_freqs = torch.outer(block_positions, inv_freq)
        ctx_cos = torch.cos(ctx_freqs).unsqueeze(0).unsqueeze(0)
        ctx_sin = torch.sin(ctx_freqs).unsqueeze(0).unsqueeze(0)
        block_cos = torch.cos(block_freqs).unsqueeze(0).unsqueeze(0)
        block_sin = torch.sin(block_freqs).unsqueeze(0).unsqueeze(0)

        x = noise_embedding
        for layer in self.layers:
            x = layer(
                x,
                ctx,
                ctx_cos,
                ctx_sin,
                block_cos,
                block_sin,
                input_pos,
                valid_ctx_len,
            )

        return self.norm(x)

    model.forward = types.MethodType(_cuda_dflash_forward, model)


def dflash_cuda_source_transformations(model: nn.Module) -> None:
    """Apply CUDA source transformations to a DFlashDraftModel in place.

    Mirrors ``dflash_mlx_source_transformations``: swaps each layer's
    ``DFlashAttention`` for ``CUDADFlashAttention`` (flat KV cache with block
    scratch + length-aware ``torch.ops.triton.sdpa``) and replaces the layer and
    model forwards to precompute RoPE and thread ``input_pos`` through. Keeps
    ``dflash_model.py`` fully backend-agnostic.

    Args:
        model: ``DFlashDraftModel`` instance to transform in place.
    """
    block_size = model.config.block_size
    n_swapped = 0
    for layer in model.layers:
        layer.self_attn = CUDADFlashAttention.from_attention(
            layer.self_attn, block_size
        )
        _replace_dflash_cuda_layer_forward(layer)
        n_swapped += 1
    _replace_dflash_cuda_model_forward(model)
    print(
        f"[muse_glimmer cuda] DFlash: swapped {n_swapped} attention layers with "
        f"CUDADFlashAttention (flat KV cache + length-aware triton.sdpa)"
    )


def materialize_dflash_runtime_buffers(
    model: nn.Module, dtype: torch.dtype, device: str = "cpu"
) -> None:
    """Materialize meta-device DFlash KV caches after checkpoint loading.

    ``CUDADFlashAttention`` creates its KV caches on the meta device; this fills
    them with real zero buffers of the activation dtype, after the attention swap
    and quantized-tensor conversion.
    """
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        parent.register_buffer(
            parts[-1],
            torch.zeros(buf.shape, dtype=dtype, device=device),
            persistent=False,
        )


def _project_qkvo(self, h: torch.Tensor):
    if self.fuse_qko:
        qko = self.qko_proj(h)
        q_end = self.q_dim
        k_end = q_end + self.kv_dim
        xq = qko[..., :q_end]
        xk = qko[..., q_end:k_end]
        og = qko[..., k_end:] if self.use_o_gate else None
        xv = self.v_proj(h)
    elif self.fuse_qkv:
        qkv = self.qkv_proj(h)
        q_end = self.q_dim
        k_end = q_end + self.kv_dim
        v_end = k_end + self.kv_dim
        xq = qkv[..., :q_end]
        xk = qkv[..., q_end:k_end]
        xv = qkv[..., k_end:v_end]
        og = qkv[..., v_end:] if self.use_o_gate else None
    else:
        xq = self.q_proj(h)
        xk = self.k_proj(h)
        xv = self.v_proj(h)
        og = self.og_proj(h) if self.use_o_gate else None
    return xq, xk, xv, og


def _turboquant_attention_forward(
    self,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for ``MuseGlimmerAttention.forward`` on global/NoPE layers.

    Mirrors the default forward but (1) applies NO RoPE (these are global NoPE
    layers) and (2) routes attention through ``torch.ops.triton.tq4_sdpa`` over
    a ``TurboQuantKVCache``.

    NOTE: ``attn_mask`` is unused — the global mask is standard causal, so it is
    reconstructed in the kernel (``mask_is_causal=True``) to save data transfer.
    """
    B, T, _ = x.shape

    h = self.qkv_proj_norm(x)
    xq, xk, xv, og = _project_qkvo(self, h)

    xq = xq.view(B, T, self.n_heads, self.head_dim)
    xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
    xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

    # Scaleless QK-norm (query scaling absorbed into attn_scale below).
    if self.q_norm is not None:
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

    # (B, H, T, D) for the KV cache / SDPA.
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    # NoPE: global layers apply no positional encoding (no RoPE here).

    # Compress + write. tq4_sdpa decompresses per tile, so the full
    # uncompressed K/V is never materialized.
    k_packed, k_norms, v_packed, v_norms = self.kv_cache.update(input_pos, xk, xv)

    # Number of valid (filled) KV positions = input_pos[0] + T. Bounds the
    # tq4_sdpa KV loop to the actual context instead of the full pre-allocated
    # buffer (max_seq_len), making attention O(context) instead of
    # O(max_seq_len). Kept as a GPU scalar (no ``.item()``) so the bound is
    # captured by the decode CUDA graph. Decode: T=1 -> input_pos+1; prefill
    # chunk: T -> chunk_end.
    kv_len = input_pos[0] + input_pos.shape[0]

    # scale=self.attn_scale absorbs the muP query scaling and 1/sqrt(D),
    # overriding tq4_sdpa's default 1/sqrt(D).
    y = torch.ops.triton.tq4_sdpa(
        xq,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        self.kv_cache.centroids,
        self.kv_cache.rotation,
        None,  # reconstruct the causal mask in the kernel to save data transfer
        False,  # is_causal: needs L_q==L_kv; causal comes from mask_is_causal
        self.attn_scale,
        kv_len,
        True,  # mask_is_causal: muse_glimmer global-attention mask is standard causal
    )

    y = y.transpose(1, 2).contiguous()

    # Output gate: sigmoid(OG) * attn_output.
    if self.use_o_gate and og is not None:
        og = og.view(B, T, self.n_heads, self.head_dim)
        y = torch.sigmoid(og) * y

    y = y.reshape(B, T, -1)
    return self.o_proj(y)


def _lenaware_attention_forward(
    self,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Drop-in ``MuseGlimmerAttention.forward`` for global/NoPE layers on the
    non-TurboQuant CUDA path that bounds SDPA to the valid context length.

    Identical to the default forward (plain bf16 ``FlatKVCache``, NO RoPE,
    output gate) except the final ``F.scaled_dot_product_attention`` is replaced
    with ``torch.ops.triton.sdpa(..., kv_len=...)``. Passing ``kv_len`` bounds
    the KV loop to the filled context instead of the full pre-allocated buffer
    (``max_seq_len``), making decode O(context) instead of O(max_seq_len) — and
    routes L_q==1 decode through the length-aware split-K flash-decoding kernel.
    Sliding-window layers are not patched (they already use a bounded ring
    buffer).

    The global mask is standard causal, so reconstruct it analytically inside
    SDPA from ``kv_len`` and avoid materializing/reading the dense mask.
    """
    B, T, _ = x.shape

    h = self.qkv_proj_norm(x)
    xq, xk, xv, og = _project_qkvo(self, h)

    xq = xq.view(B, T, self.n_heads, self.head_dim)
    xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
    xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

    # Scaleless QK-norm (query scaling absorbed into attn_scale below).
    if self.q_norm is not None:
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

    # (B, H, T, D) for the KV cache / SDPA.
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    # NoPE: global layers apply no positional encoding (no RoPE here).

    # Update cache and read back the full (pre-allocated) bf16 K/V buffers.
    k, v = self.kv_cache.update(input_pos, xk, xv)

    # Number of valid (filled) KV positions = input_pos[0] + T. Bounds the sdpa
    # KV loop to the actual context (O(context) instead of O(max_seq_len)). Kept
    # as a GPU scalar so the bound is captured by the decode CUDA graph.
    kv_len = input_pos[0] + input_pos.shape[0]

    # scale=self.attn_scale absorbs the muP query scaling and 1/sqrt(D).
    # enable_gqa=True lets the kernel handle the 16:1 head ratio without
    # materializing expanded K/V. With kv_len, is_causal uses bottom-right
    # alignment for chunked prefill and decode.
    y = torch.ops.triton.sdpa(
        xq,
        k,
        v,
        None,
        0.0,  # dropout_p
        True,
        self.attn_scale,
        True,  # enable_gqa
        kv_len,
    )

    y = y.transpose(1, 2).contiguous()

    # Output gate: sigmoid(OG) * attn_output.
    if self.use_o_gate and og is not None:
        og = og.view(B, T, self.n_heads, self.head_dim)
        y = torch.sigmoid(og) * y

    y = y.reshape(B, T, -1)
    return self.o_proj(y)


def cuda_source_transformations(
    model: nn.Module,
    *,
    use_turboquant: bool = False,
) -> None:
    """Apply CUDA source transformations to a Muse Glimmer model in place.

    Always bounds global-attention SDPA to the valid context via a runtime
    ``kv_len`` (O(context) decode). Optionally also swaps the global KV caches
    for TurboQuant TQ4. Sliding-window layers are untouched in both cases (they
    already use a bounded ring buffer).

    Args:
        model: ``MuseGlimmerModel`` instance to transform.
        use_turboquant: When True, swap the global (NoPE) layers' KV caches for
            ``TurboQuantKVCache`` (TQ4, ~3.8x cache memory savings) and route
            their attention through ``torch.ops.triton.tq4_sdpa``. When False,
            keep the bf16 ``FlatKVCache`` but route global attention through the
            length-aware ``torch.ops.triton.sdpa``.
    """
    if not use_turboquant:
        # Non-TurboQuant path: keep the bf16 FlatKVCache but bound global
        # attention to the valid context via a runtime kv_len scalar (routes
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
            f"[muse_glimmer cuda] length-aware SDPA: bounded {n_bounded} global-attention "
            "layers to runtime kv_len (O(context) attention); "
            "in-kernel causal mask (dense mask dropped)"
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
        f"[muse_glimmer cuda] TurboQuant: swapped {n_swapped} global-attention "
        f"KV caches with TurboQuantKVCache (TQ4)"
    )


def _embed_text_forward(
    self,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Exported ``embed_text`` entry: token embedding lookup, no sampling.

    Returns bf16 ``(B, T, dim)`` embeddings for the runner to (optionally) splice
    vision features into before calling the embeddings-input forward entry.
    """
    return self.embed_text(tokens)


def _forward_from_embeddings_sampling(
    self,
    inputs_embeds: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Embeddings-input prefill entry that samples one token on-device.

    Runs the decoder stack over pre-computed ``inputs_embeds`` (built by the
    runner from ``embed_text`` plus any image splice) and returns a single
    Gumbel-max token id ``(B, 1)`` from the last position. Only the last
    position's logits are computed, so ``lm_head`` runs on ``(B, 1, H)``.
    """
    x = self._run_blocks(inputs_embeds, input_pos)
    x = self.output_norm(x)
    last = self._soft_cap(self.lm_head(x[:, -1, :]))
    return sample(last, temperature)


def _decode_from_embedding_sampling(
    self,
    input_embedding: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Single-embedding decode entry that samples one token on-device."""
    x = self._run_blocks(input_embedding, input_pos)
    x = self.output_norm(x)
    last = self._soft_cap(self.lm_head(x[:, -1, :]))
    return sample(last, temperature)


def add_on_device_sampler(model: nn.Module) -> None:
    """Bind the three on-device-sampling entry points to a Muse Glimmer model in place.

    Exposes bound methods used by ``export.py`` via a per-method
    ``forward``-swap:

    * ``embed_text_forward(tokens) -> (B, T, dim)`` bf16 embeddings (no sampling)
    * ``forward_from_embeddings(inputs_embeds, input_pos, temperature)``
    * ``decode_from_embedding(input_embedding, input_pos, temperature)``

    Splitting ``embed_text`` from both decoder entry points gives CUDA and MLX
    the same method contract and lets the runner splice vision features before
    prefill. The NLL-validation path exports the sampler-free model methods.

    Args:
        model: ``MuseGlimmerModel`` instance to transform.
    """
    model.embed_text_forward = types.MethodType(_embed_text_forward, model)
    model.forward_from_embeddings = types.MethodType(
        _forward_from_embeddings_sampling, model
    )
    model.decode_from_embedding = types.MethodType(
        _decode_from_embedding_sampling, model
    )


# Vision encoder: optional FP32 GEMM (CUDA only)
# The shared vision model remains backend-neutral; CUDA rewrites are applied
# only during export.


@triton.jit
def _bf16_fp32_matmul_kernel(
    x,
    weight,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = weight + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton_op("triton::bf16_fp32_matmul", mutates_args={})
def bf16_fp32_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T`` with bf16 operands and F32 output storage."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("bf16_fp32_matmul requires bf16 inputs")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected x[M,K] and weight[N,K]")

    m, _ = x.shape
    n = weight.shape[0]
    if not x.is_cuda:
        return torch.mm(x.float(), weight.float().t())

    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    wrap_triton(_bf16_fp32_matmul_kernel)[grid](
        x,
        weight,
        output,
        m,
        n,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=3,
    )
    return output


@bf16_fp32_matmul.register_fake
def _bf16_fp32_matmul_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device
    )


def _f32_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``nn.Linear.forward`` with bf16 operands and an F32-accumulated output."""
    shape = x.shape[:-1]
    y = bf16_fp32_matmul(
        x.reshape(-1, self.in_features).to(torch.bfloat16),
        self.weight.to(torch.bfloat16),
    )
    if self.bias is not None:
        y = y + self.bias.to(torch.float32)
    return y.view(*shape, self.out_features)


def _make_fp32_encoder_forward(start: int, end: int, out_dtype: torch.dtype):
    """Encoder ``forward`` that runs blocks [start, end] in F32.

    Mirrors the stock ``MuseGlimmerVisionEncoder.forward`` but adds the dtype
    transitions around the F32 block range. Kept as a closure so ``start`` /
    ``end`` / ``out_dtype`` are trace-time constants and none of this state has
    to exist on the shared module.
    """

    def _forward(
        self,
        patches: torch.Tensor,
        pos_emb: torch.Tensor,
        cos_2d: torch.Tensor,
        sin_2d: torch.Tensor,
        sparse_perm: torch.Tensor,
        inv_perm: torch.Tensor,
        global_mask: torch.Tensor,
        sparse_mask: torch.Tensor,
        pixel_perm: torch.Tensor,
    ) -> torch.Tensor:
        w_dtype = self.conv1_linear.weight.dtype
        x = self.conv1_linear(patches.to(w_dtype))
        x = x + pos_emb.to(w_dtype)
        x = self.ln_pre(x)

        x = x.index_select(1, sparse_perm)
        cos_p = cos_2d.index_select(0, sparse_perm)
        sin_p = sin_2d.index_select(0, sparse_perm)
        cos_b = cos_p.unsqueeze(0).unsqueeze(2)
        sin_b = sin_p.unsqueeze(0).unsqueeze(2)

        for i, block in enumerate(self.blocks):
            if i == start:
                x = x.to(torch.float32)
                cos_b = cos_b.to(torch.float32)
                sin_b = sin_b.to(torch.float32)
            elif i == end + 1:
                x = x.to(out_dtype)
                cos_b = cos_b.to(out_dtype)
                sin_b = sin_b.to(out_dtype)
            mask = global_mask if self._is_global[i] else sparse_mask
            x = block(x, cos_b, sin_b, mask)
        if end == len(self.blocks) - 1:
            x = x.to(out_dtype)

        x = x.index_select(1, inv_perm)
        x = self.ln_post(x)

        x = x.index_select(1, pixel_perm)
        b, p, d = x.shape
        f = self.downsample_factor
        x_ds = x.view(b, p // (f * f), (f * f) * d)

        x_ds = self.adapter_fc(x_ds)
        x_ds = F.gelu(self.adapter_proj(x_ds))
        x_ds = self.vision_proj(x_ds)
        x_ds = self.perception_emb_norm(x_ds)
        return x_ds.to(w_dtype)

    return _forward


def vision_cuda_source_transformations(
    vision_model: nn.Module,
    *,
    start: int = 0,
    end: int = 34,
    use_triton_mm: bool = True,
) -> None:
    """Run vision blocks [start, end] in F32 on the CUDA export path.

    Two sub-modes, matching ``--vision-fp32-mm``:

    - ``use_triton_mm=True`` ("triton"): keep bf16 weights and swap each
      ``nn.Linear.forward`` in range for an F32-output Triton GEMM.
    - ``use_triton_mm=False`` ("native"): cast the modules themselves to F32.

    Must run AFTER the checkpoint is loaded (weights are assigned by FQN, and
    CUDA has already run ``convert_quantized_tensors_for_cuda``) and BEFORE
    ``torch.export``. Wraps ``forward`` rather than swapping module classes so
    module identity and ``state_dict`` keys stay stable.
    """
    n_blocks = len(vision_model.blocks)
    if not (0 <= start <= end < n_blocks):
        raise ValueError(f"invalid F32 block range [{start}, {end}]")

    out_dtype = vision_model.conv1_linear.weight.dtype
    vision_model.forward = types.MethodType(
        _make_fp32_encoder_forward(start, end, out_dtype), vision_model
    )

    if not use_triton_mm:
        if start == 0:
            vision_model.conv1_linear.to(torch.float32)
        vision_model.ln_pre.to(torch.float32)
        for block in vision_model.blocks[start : end + 1]:
            block.to(torch.float32)
        print(
            f"[muse_glimmer cuda] vision: native F32 blocks [{start}, {end}] "
            f"of {n_blocks}"
        )
        return

    n_swapped = 0
    if start == 0:
        vision_model.conv1_linear.forward = types.MethodType(
            _f32_linear_forward, vision_model.conv1_linear
        )
        n_swapped += 1
        vision_model.ln_pre.to(torch.float32)
    for i in range(start, end + 1):
        block = vision_model.blocks[i]
        block.ln_1.to(torch.float32)
        block.ln_2.to(torch.float32)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.forward = types.MethodType(_f32_linear_forward, module)
                n_swapped += 1
    print(
        f"[muse_glimmer cuda] vision: F32-output Triton GEMM on {n_swapped} linears "
        f"in blocks [{start}, {end}] of {n_blocks}"
    )
