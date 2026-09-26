# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX source transformations for Muse Glimmer.

The transformations replace attention, RoPE, and KV-cache operations with MLX
operators while preserving sliding-window and global-attention behavior.
"""

import inspect
import types

import executorch.backends.mlx.custom_ops  # noqa: F401 — registers mlx:: ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.mlx.llm.cache import (
    KVCache as MLXKVCache,
    RingBufferKVCache as MLXRingKVCache,
)
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    build_dflash_swa_mask,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerAttention


class MLXMuseGlimmerAttention(nn.Module):
    """Muse Glimmer attention backed by MLX operators and KV caches."""

    @classmethod
    def from_attention(
        cls,
        attn: "MuseGlimmerAttention",
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
        max_write_len: int | None = None,
    ) -> "MLXMuseGlimmerAttention":
        m = cls.__new__(cls)
        nn.Module.__init__(m)

        m.n_heads = attn.n_heads
        m.n_kv_heads = attn.n_kv_heads
        m.head_dim = attn.head_dim
        m.q_dim = attn.q_dim
        m.kv_dim = attn.kv_dim
        m.fuse_qkv = attn.fuse_qkv
        m.use_rope = attn.use_rope
        m.is_sliding = attn.is_sliding
        m.window_size = attn.window_size
        m.use_o_gate = attn.use_o_gate
        m.attn_scale = attn.attn_scale
        m.rope_theta = attn.rope_theta

        m.qkv_proj_norm = attn.qkv_proj_norm
        if attn.fuse_qkv:
            m.qkv_proj = attn.qkv_proj
        else:
            m.q_proj = attn.q_proj
            m.k_proj = attn.k_proj
            m.v_proj = attn.v_proj
            if attn.use_o_gate:
                m.og_proj = attn.og_proj
        m.o_proj = attn.o_proj
        m.q_norm = attn.q_norm
        m.k_norm = attn.k_norm

        if attn.is_sliding:
            ring_kwargs = {
                "max_batch_size": 1,
                "max_context_length": attn.window_size,
                "n_heads": attn.n_kv_heads,
                "head_dim": attn.head_dim,
                "dtype": dtype,
            }
            if max_write_len is not None and _ring_supports_mwl():
                ring_kwargs["max_write_len"] = max_write_len
            m.kv_cache = MLXRingKVCache(**ring_kwargs)
        else:
            m.kv_cache = MLXKVCache(
                max_batch_size=1,
                max_context_length=max_seq_len,
                n_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                enable_dynamic_shape=True,
                dtype=dtype,
            )

        return m

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        start_pos = input_pos[0].item()

        h = self.qkv_proj_norm(x)
        if self.fuse_qkv:
            qkv = self.qkv_proj(h)
            q_end = self.q_dim
            k_end = q_end + self.kv_dim
            v_end = k_end + self.kv_dim
            xq = qkv[..., :q_end].view(B, T, self.n_heads, self.head_dim)
            xk = qkv[..., q_end:k_end].view(B, T, self.n_kv_heads, self.head_dim)
            xv = qkv[..., k_end:v_end].view(B, T, self.n_kv_heads, self.head_dim)
            og = qkv[..., v_end:] if self.use_o_gate else None
        else:
            xq = self.q_proj(h).view(B, T, self.n_heads, self.head_dim)
            xk = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
            xv = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
            og = self.og_proj(h) if self.use_o_gate else None

        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if self.use_rope:
            q = torch.ops.mlx.rope(
                q, self.head_dim, start_pos, True, self.rope_theta, 1.0, None
            )
            k = torch.ops.mlx.rope(
                k, self.head_dim, start_pos, True, self.rope_theta, 1.0, None
            )

        k_cache, v_cache = self.kv_cache.update(start_pos, k, v)

        if self.is_sliding:
            sdpa_mask = self.kv_cache.create_sliding_window_mask(start_pos, T)
            y = torch.ops.mlx.custom_sdpa(
                q,
                k_cache,
                v_cache,
                start_pos=self.kv_cache.buffer_size - T,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            y = torch.ops.mlx.custom_sdpa(
                q,
                k_cache,
                v_cache,
                start_pos=start_pos,
                dropout_p=0.0,
                is_causal=True,
                scale=self.attn_scale,
            )

        y = y.transpose(1, 2).contiguous()

        if self.use_o_gate and og is not None:
            og = og.view(B, T, self.n_heads, self.head_dim)
            y = torch.sigmoid(og) * y

        y = y.reshape(B, T, -1)
        return self.o_proj(y)


def _ring_supports_mwl() -> bool:
    return "max_write_len" in inspect.signature(MLXRingKVCache.__init__).parameters


def _replace_layer_forward(layer: nn.Module) -> None:
    """Replace MuseGlimmerDecoderLayer's forward to drop the mask parameters."""

    def _mlx_layer_forward(
        self, x: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        attn_out = self.self_attn(x, input_pos)
        x = x + self.post_attn_norm(attn_out)

        ffn_out = self.mlp(x)
        x = x + self.post_ffn_norm(ffn_out)
        return x

    layer.forward = types.MethodType(_mlx_layer_forward, layer)


def _replace_model_forward(model: nn.Module) -> None:
    """Install sampler-free MLX entry points.

    MLX samples on the host, so prefill returns softcapped last-token logits
    ``(B, V)``. Splitting embedding (``mlx_embed_text``) from the decoder stack
    (``mlx_prefill_forward``, embeddings-input) is what lets the C++ runner
    splice image soft-tokens into the prompt embeddings before prefill (the MLX
    vision path). The token-input ``forward`` is kept as a text-only convenience
    and stays equivalent to ``mlx_embed_text`` followed by
    ``mlx_prefill_forward``.

    Each MLX attention builds its own mask via ``custom_sdpa``, so there is no
    ``_build_masks`` call and no per-layer mask arguments.
    """

    def _mlx_embed_text(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(tokens)
        if self.normalize_tok_embeddings:
            x = self.tok_norm(x)
        return x.to(self.activation_dtype)

    def _mlx_logits_from_embeds(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, input_pos)

        x = self.output_norm(x)
        last = self.lm_head(x[:, -1, :]).float()
        mult = self.output_multiplier.float()
        cap = self.logit_softcap.float()
        return torch.tanh(last * mult / cap) * cap

    def _mlx_prefill_forward(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        return self.mlx_logits_from_embeds(inputs_embeds, input_pos)

    def _mlx_model_forward(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        return self.mlx_prefill_forward(self.mlx_embed_text(tokens), input_pos)

    model.mlx_embed_text = types.MethodType(_mlx_embed_text, model)
    model.mlx_logits_from_embeds = types.MethodType(_mlx_logits_from_embeds, model)
    model.mlx_prefill_forward = types.MethodType(_mlx_prefill_forward, model)
    model.forward = types.MethodType(_mlx_model_forward, model)


def _apply_mlx_transforms(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    max_write_len: int | None = None,
) -> None:
    """Shared MLX transforms: swap attention modules, patch layer forwards, fuse norms."""
    config = model.config

    # Record the activation compute dtype so the patched forwards cast to it.
    model.activation_dtype = dtype

    for layer in model.layers:
        layer.self_attn = MLXMuseGlimmerAttention.from_attention(
            layer.self_attn,
            max_seq_len=config.max_seq_len,
            dtype=dtype,
            max_write_len=max_write_len,
        )
        _replace_layer_forward(layer)


def mlx_source_transformations(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    max_write_len: int | None = None,
) -> None:
    """Apply MLX source transformations to a Muse Glimmer model in-place.

    Self-contained MLX adaptation. After calling this, the model exposes
    sampler-free, mask-free entry points that return ``(B, V)`` last-token
    logits (host-side sampling):

    * ``forward(tokens, input_pos)`` — text-only convenience.
    * ``mlx_embed_text(tokens)`` — token embedding lookup (+ tok_norm), cast to
      the activation dtype.
    * ``mlx_prefill_forward(inputs_embeds, input_pos)`` — embeddings-input
      prefill (lets the runner splice image soft-tokens before the decoder
      stack; used by the MLX vision path).

    Args:
        model: MuseGlimmerModel to transform in place.
        dtype: dtype for KV cache buffers (bf16 by default).
        max_write_len: largest single write to the sliding-window ring buffer
            (i.e. the max prefill chunk). When set, the ring buffer is sized to
            ``window + max_write_len - 1`` instead of ``2 * window``, saving
            memory for chunked prefill. Ignored if the installed
            ``RingBufferKVCache`` predates the ``max_write_len`` parameter.
    """
    _apply_mlx_transforms(model, dtype, max_write_len)
    _replace_model_forward(model)


# DFlash: target model with hidden state tapping


def _replace_model_forward_with_hidden_tapping(
    model: nn.Module, target_layer_ids: list[int]
) -> None:
    """Install MLX DFlash token- and embeddings-input target entry points.

    ``target_forward_from_embeddings`` owns the target traversal and returns
    full-sequence ``(logits, hidden)`` tensors. The token-input ``forward``
    remains backward compatible and delegates after embedding the tokens.
    """
    target_set = set(target_layer_ids)
    model._dflash_target_layer_ids = target_layer_ids

    def _mlx_embed_text(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(tokens)
        if self.normalize_tok_embeddings:
            x = self.tok_norm(x)
        return x.to(self.activation_dtype)

    def _mlx_target_forward_from_embeddings(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = inputs_embeds.to(self.activation_dtype)
        tapped = []
        for i, layer in enumerate(self.layers):
            if i in target_set:
                tapped.append(x)
            x = layer(x, input_pos)

        x = self.output_norm(x)
        all_logits = self.lm_head(x).float()
        mult = self.output_multiplier.float()
        cap = self.logit_softcap.float()
        logits = torch.tanh(all_logits * mult / cap) * cap
        return logits, torch.cat(tapped, dim=-1)

    def _mlx_model_forward_with_hidden(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target_forward_from_embeddings(self.embed_text(tokens), input_pos)

    model.embed_text = types.MethodType(_mlx_embed_text, model)
    model.target_forward_from_embeddings = types.MethodType(
        _mlx_target_forward_from_embeddings, model
    )
    model.forward = types.MethodType(_mlx_model_forward_with_hidden, model)


def mlx_source_transformations_dflash_target(
    model: nn.Module,
    target_layer_ids: list[int],
    dtype: torch.dtype = torch.bfloat16,
    max_write_len: int | None = None,
) -> None:
    """Apply MLX source transformations for the DFlash target model.

    Same as ``mlx_source_transformations`` but installs a shared embeddings-input
    target traversal that also returns concatenated hidden states from
    ``target_layer_ids``.

    After calling this, the model exposes:
        ``forward(tokens, input_pos) -> (logits [B, T, V], hidden [B, T, N*D])``
        ``embed_text(tokens) -> inputs_embeds [B, T, D]``
        ``target_forward_from_embeddings(inputs_embeds, input_pos)`` with the
        same full-sequence logits and hidden outputs as ``forward``.
    """
    _apply_mlx_transforms(model, dtype, max_write_len)
    _replace_model_forward_with_hidden_tapping(model, target_layer_ids)


# DFlash: draft model MLX source transformations


class MLXDFlashAttention(nn.Module):
    """MLX-optimized DFlash cross-attention with KV cache for context.

    Context K/V (from projected target hidden states) are cached across calls.
    Only new target hidden states are projected each call. Block token K/V are
    computed fresh every call.
    """

    @classmethod
    def from_attention(
        cls,
        attn,
        max_context_length: int,
        dtype: torch.dtype = torch.bfloat16,
        is_sliding: bool = False,
        window_size: int | None = None,
        max_write_len: int | None = None,
    ) -> "MLXDFlashAttention":
        m = cls.__new__(cls)
        nn.Module.__init__(m)

        m.n_heads = attn.n_heads
        m.n_kv_heads = attn.n_kv_heads
        m.head_dim = attn.head_dim
        m.attn_scale = attn.attn_scale
        m.rope_theta = attn.rope_theta
        m.is_sliding = is_sliding

        m.q_proj = attn.q_proj
        m.k_proj = attn.k_proj
        m.v_proj = attn.v_proj
        m.o_proj = attn.o_proj
        m.q_norm = attn.q_norm
        m.k_norm = attn.k_norm

        if is_sliding:
            ring_kwargs = {
                "max_batch_size": 1,
                "max_context_length": window_size,
                "n_heads": attn.n_kv_heads,
                "head_dim": attn.head_dim,
                "dtype": dtype,
            }
            if max_write_len is not None and _ring_supports_mwl():
                ring_kwargs["max_write_len"] = max_write_len
            m.kv_cache = MLXRingKVCache(**ring_kwargs)
        else:
            m.kv_cache = MLXKVCache(
                max_batch_size=1,
                max_context_length=max_context_length,
                n_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                enable_dynamic_shape=True,
                dtype=dtype,
            )

        return m

    def forward(
        self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        n_new = target_hidden.shape[1]
        ctx_start = input_pos[0].item()
        total_ctx = ctx_start + n_new

        # Project new context K/V, apply k_norm + RoPE, write to cache
        k_new = self.k_proj(target_hidden).view(
            B, n_new, self.n_kv_heads, self.head_dim
        )
        v_new = self.v_proj(target_hidden).view(
            B, n_new, self.n_kv_heads, self.head_dim
        )
        k_new = self.k_norm(k_new)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)
        k_new = torch.ops.mlx.rope(
            k_new, self.head_dim, ctx_start, False, self.rope_theta, 1.0, None
        )
        k_ctx, v_ctx = self.kv_cache.update(ctx_start, k_new, v_new)

        if self.is_sliding:
            attn_mask = build_dflash_swa_mask(
                total_ctx,
                T,
                self.kv_cache.buffer_size,
                self.kv_cache.window_size,
            )
        else:
            torch._check(ctx_start >= 0)
            torch._check(total_ctx <= self.kv_cache.max_context_length)
            k_ctx = k_ctx[:, :, :total_ctx, :]
            v_ctx = v_ctx[:, :, :total_ctx, :]
            attn_mask = None

        # Block K/V — computed fresh each call
        k_blk = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v_blk = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        k_blk = self.k_norm(k_blk)
        k_blk = k_blk.transpose(1, 2)
        v_blk = v_blk.transpose(1, 2)
        k_blk = torch.ops.mlx.rope(
            k_blk, self.head_dim, total_ctx, False, self.rope_theta, 1.0, None
        )

        # Q from block tokens
        xq = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xq = self.q_norm(xq)
        q = xq.transpose(1, 2)
        q = torch.ops.mlx.rope(
            q, self.head_dim, total_ctx, False, self.rope_theta, 1.0, None
        )

        # SDPA over cached context + block
        k = torch.cat([k_ctx, k_blk], dim=2)
        v = torch.cat([v_ctx, v_blk], dim=2)

        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.attn_scale, is_causal=False
        )

        y = y.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(y)


def _replace_dflash_layer_forward(layer: nn.Module) -> None:
    """Replace DFlashDecoderLayer forward to take input_pos instead of cos/sin."""

    def _mlx_dflash_layer_forward(
        self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), target_hidden, input_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    layer.forward = types.MethodType(_mlx_dflash_layer_forward, layer)


def _replace_dflash_model_forward(model: nn.Module) -> None:
    """Replace DFlashDraftModel forward to pass input_pos to layers instead of cos/sin.

    After this, the model accepts only new target hidden states per call.
    Context K/V caching is handled by the per-layer MLXDFlashAttention modules.
    """

    def _mlx_dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.hidden_norm(self.fc(target_hidden))

        x = noise_embedding
        for layer in self.layers:
            x = layer(x, ctx, input_pos)

        return self.norm(x)

    model.forward = types.MethodType(_mlx_dflash_forward, model)


def dflash_mlx_source_transformations(
    model: nn.Module,
    max_context_length: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    sliding_window: int | None = None,
    sliding_window_pattern: list[bool] | None = None,
    max_write_len: int | None = None,
) -> nn.Module:
    """Apply MLX source transformations to a DFlashDraftModel in-place.

    Swaps each layer's DFlashAttention with MLXDFlashAttention (which has a
    per-layer KV cache for context K/V) and replaces layer and model forwards
    to pass input_pos instead of cos/sin.
    """
    for i, layer in enumerate(model.layers):
        is_sliding = (
            sliding_window_pattern[i] if sliding_window_pattern is not None else False
        )
        layer.self_attn = MLXDFlashAttention.from_attention(
            layer.self_attn,
            max_context_length,
            dtype,
            is_sliding=is_sliding,
            window_size=sliding_window if is_sliding else None,
            max_write_len=max_write_len if is_sliding else None,
        )
        _replace_dflash_layer_forward(layer)

    _replace_dflash_model_forward(model)
    return model
