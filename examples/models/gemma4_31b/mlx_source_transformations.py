# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLX source transformations for Gemma 4 31B-IT.

Replaces the generic PyTorch ops in the model with MLX custom ops that lower
to optimized Metal kernels:

- ``torch.ops.mlx.rope`` for rotary position embeddings
- ``torch.ops.mlx.kv_cache_update`` for KV cache scatter (via MLX cache modules)
- ``torch.ops.mlx.custom_sdpa`` for scaled dot-product attention with GQA

Applied at export time before ``torch.export`` — the model code in ``model.py``
stays backend-agnostic.
"""

import executorch.backends.mlx.custom_ops  # noqa: F401 — registers mlx:: ops
import torch
import torch.nn as nn
from executorch.backends.mlx.llm.cache import (
    KVCache as MLXKVCache,
    RingBufferKVCache as MLXRingKVCache,
)
from executorch.backends.mlx.llm.turboquant_cache import (
    TurboQuantKVCache as MLXTurboQuantKVCache,
)


def _replace_attention_forward(attn: nn.Module) -> None:
    """Replace a Gemma4Attention's forward with one that uses MLX custom ops."""
    import types

    def _mlx_forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        start_pos = input_pos[0].item()

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        raw_k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.k_eq_v:
            raw_v = raw_k
        else:
            raw_v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(raw_k)
        v = self.v_norm(raw_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE via mlx::rope.
        if self.is_sliding:
            q = torch.ops.mlx.rope(
                q, self.head_dim, start_pos, False, self.rope_theta, 1.0, None
            )
            k = torch.ops.mlx.rope(
                k, self.head_dim, start_pos, False, self.rope_theta, 1.0, None
            )
        else:
            # Full-attention layers use proportional partial RoPE: only
            # rotary_dim out of head_dim dimensions are rotated. Pass
            # dims=rotary_dim and the non-zero frequencies as 1D freqs.
            # MLX computes inv_freq = 1/freqs internally.
            rotary_dim = int(self.head_dim * self.partial_rotary)
            rotary_inv_freq = self.inv_freq[: rotary_dim // 2]
            mlx_freqs = 1.0 / rotary_inv_freq
            q = torch.ops.mlx.rope(q, rotary_dim, start_pos, False, 0.0, 1.0, mlx_freqs)
            k = torch.ops.mlx.rope(k, rotary_dim, start_pos, False, 0.0, 1.0, mlx_freqs)

        if getattr(self, "is_turboquant", False):
            self.kv_cache.update(start_pos, k, v)
            y = self.kv_cache.sdpa(q, start_pos, scale=self.scaling)
        else:
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
                    scale=self.scaling,
                )
            else:
                y = torch.ops.mlx.custom_sdpa(
                    q,
                    k_cache,
                    v_cache,
                    start_pos=start_pos,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=self.scaling,
                )

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)

    attn.forward = types.MethodType(_mlx_forward, attn)


def _replace_layer_forward(layer: nn.Module) -> None:
    """Replace Gemma4DecoderLayer's forward to remove mask parameters."""
    import types

    def _mlx_layer_forward(
        self, x: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, input_pos)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h

        return x * self.layer_scalar

    layer.forward = types.MethodType(_mlx_layer_forward, layer)


def _replace_model_forward(model: nn.Module) -> None:
    """Replace the top-level Gemma4_31B forward with a sampler-free, mask-free
    ``(tokens, input_pos) → (B, 1, V)`` variant.

    MLX samples on the host, so the on-device sampler and temperature input
    are dropped.  Each MLX attention builds its own mask via ``custom_sdpa``,
    so ``_build_masks`` and the per-layer mask arguments are removed.
    """
    import types

    def _mlx_model_forward(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        x = self.embed_tokens(tokens) * self.embed_normalizer
        for layer in self.layers:
            x = layer(x, input_pos)
        x = self.norm(x)
        last = self.lm_head(x[:, -1, :]).float()
        cap = self.logit_softcap.float()
        return torch.tanh(last / cap) * cap

    model.forward = types.MethodType(_mlx_model_forward, model)


def mlx_source_transformations(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    use_turboquant: bool = False,
) -> None:
    """Apply MLX source transformations to a Gemma 4 31B model in-place.

    Self-contained MLX adaptation. After calling this, the model has
    signature ``(tokens, input_pos) → (B, 1, V)`` logits — no temperature,
    no sampler, no attention masks.

    - Replaces KV caches with MLX-optimized versions using ``mlx.kv_cache_update``
    - Rewrites attention forward to use ``mlx.rope`` and ``mlx.custom_sdpa``
    - Rewrites layer forward to drop mask parameters (each attention builds
      its own mask via ``custom_sdpa``)
    - Rewrites model forward to drop the sampler and ``_build_masks``

    Args:
        model: Gemma4_31B model to transform in place.
        dtype: dtype for KV cache buffers (bf16 by default).
        use_turboquant: If True, swap full-attention layers' KV caches
            for ``MLXTurboQuantKVCache`` (~3.8× cache memory savings).
            Sliding-window layers are unaffected.
    """
    config = model.config

    for layer in model.layers:
        attn = layer.self_attn

        if attn.is_sliding:
            attn.kv_cache = MLXRingKVCache(
                max_batch_size=1,
                max_context_length=config.sliding_window,
                n_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                dtype=dtype,
            )
            attn.is_turboquant = False
        elif use_turboquant:
            attn.kv_cache = MLXTurboQuantKVCache(
                max_batch_size=1,
                max_context_length=config.max_seq_len,
                n_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                enable_dynamic_shape=True,
                dtype=dtype,
            )
            attn.is_turboquant = True
        else:
            attn.kv_cache = MLXKVCache(
                max_batch_size=1,
                max_context_length=config.max_seq_len,
                n_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                enable_dynamic_shape=True,
                dtype=dtype,
            )
            attn.is_turboquant = False

        _replace_attention_forward(attn)
        _replace_layer_forward(layer)

    _replace_model_forward(model)
