#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX-optimized attention for ExecutorTorch's Llama attention registry.

Registers an "mlx" attention type that uses mlx::kv_cache_update and
mlx::custom_sdpa for efficient execution on Apple Silicon.

Usage:
    import executorch.backends.apple.mlx.examples.et_attention  # noqa: F401

    model_args = ModelArgs(attention_type="mlx", ...)
    transformer = construct_transformer(model_args)
"""

from typing import Any, Optional, Tuple, TYPE_CHECKING

import executorch.backends.apple.mlx.custom_ops as _mlx_custom_ops  # noqa: F401

import torch
import torch.nn as nn
from executorch.backends.apple.mlx.examples.cache import ETKVCache
from executorch.examples.models.llama.attention import (
    Attention,
    ForwardOptions,
    register_attention,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope

if TYPE_CHECKING:
    from executorch.examples.models.llama.attention import AttentionMHA


@register_attention("mlx")
class MLXAttentionMHA(Attention):
    """
    MLX-optimized attention using mlx::kv_cache_update and mlx::custom_sdpa.

    Supports MHA, GQA, KV caching, and optional QK normalization.
    Follows the same interface as AttentionMHA.
    """

    def __init__(
        self,
        args: ModelArgs,
        layer_id: int,
        rope: Rope,
        **_kwargs: Any,
    ):
        super().__init__()
        if not args.use_kv_cache:
            raise ValueError("MLXAttention requires use_kv_cache=True")

        self.use_kv_cache = True
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.max_context_len = args.max_context_len
        self.dim = args.dim
        self.attention_qkv_bias = args.attention_qkv_bias
        self.use_qk_norm = args.use_qk_norm
        self.qk_norm_before_rope = args.qk_norm_before_rope
        self.enable_dynamic_shape = args.enable_dynamic_shape

        if self.use_qk_norm:
            self.q_norm_fn = RMSNorm(self.head_dim, eps=args.norm_eps)
            self.k_norm_fn = RMSNorm(self.head_dim, eps=args.norm_eps)

        self.wq = nn.Linear(
            self.dim, self.n_heads * self.head_dim, bias=self.attention_qkv_bias
        )
        self.wk = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias
        )
        self.wv = nn.Linear(
            self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.layer_id = layer_id
        self.rope = rope
        self.rope_base = rope.params.rope_freq_base
        self.use_fused_rope = self._can_use_fused_rope(rope.params)
        self.rope_traditional = not rope.params.use_hf_rope
        self.rope_dims = int(self.head_dim * rope.params.partial_rotary_factor)

        self.kv_cache = ETKVCache(
            max_batch_size=args.max_batch_size,
            max_context_length=args.max_context_len,
            n_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=args.enable_dynamic_shape,
        )

    @staticmethod
    def _can_use_fused_rope(params: ModelArgs) -> bool:
        if params.no_rope_layer_interval is not None:
            return False
        return True

    @classmethod
    def from_attention_mha(
        cls, other: "AttentionMHA", dtype: Optional[torch.dtype] = None
    ) -> "MLXAttentionMHA":
        """
        Create an MLXAttentionMHA from an existing AttentionMHA.

        Shares weight references (wq, wk, wv, wo, rope, norm) and creates
        a fresh ETKVCache.
        """
        from executorch.examples.models.llama.attention import AttentionMHA

        assert isinstance(other, AttentionMHA)

        instance = cls.__new__(cls)
        Attention.__init__(instance)

        # Copy all config attributes
        instance.use_kv_cache = True
        instance.n_heads = other.n_heads
        instance.n_kv_heads = other.n_kv_heads
        instance.n_local_heads = other.n_local_heads
        instance.n_local_kv_heads = other.n_local_kv_heads
        instance.n_rep = other.n_rep
        instance.head_dim = other.head_dim
        instance.max_batch_size = other.max_batch_size
        instance.max_context_len = other.max_context_len
        instance.dim = other.dim
        instance.attention_qkv_bias = other.attention_qkv_bias
        instance.use_qk_norm = other.use_qk_norm
        instance.qk_norm_before_rope = other.qk_norm_before_rope
        instance.enable_dynamic_shape = other.enable_dynamic_shape

        # Share weight references
        instance.wq = other.wq
        instance.wk = other.wk
        instance.wv = other.wv
        instance.wo = other.wo
        instance.layer_id = other.layer_id
        instance.rope = other.rope
        instance.rope_base = other.rope.params.rope_freq_base
        instance.use_fused_rope = cls._can_use_fused_rope(other.rope.params)
        instance.rope_traditional = not other.rope.params.use_hf_rope
        instance.rope_dims = int(
            instance.head_dim * other.rope.params.partial_rotary_factor
        )

        if other.use_qk_norm:
            instance.q_norm_fn = other.q_norm_fn
            instance.k_norm_fn = other.k_norm_fn

        # Create fresh MLX KV cache
        cache_dtype = dtype if dtype is not None else torch.float32
        if hasattr(other, "kv_cache") and hasattr(other.kv_cache, "k_cache"):
            cache_dtype = dtype if dtype is not None else other.kv_cache.k_cache.dtype
        instance.kv_cache = ETKVCache(
            max_batch_size=other.max_batch_size,
            max_context_length=other.max_context_len,
            n_heads=instance.n_kv_heads,
            head_dim=instance.head_dim,
            enable_dynamic_shape=other.enable_dynamic_shape,
            dtype=cache_dtype,
        )

        return instance

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        **kwargs: ForwardOptions,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        input_pos = kwargs.get("input_pos")
        assert input_pos is not None
        bsz, seqlen, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if "start_pos" in kwargs:
            start_pos = kwargs["start_pos"]
        else:
            start_pos = input_pos[0].item()

        if self.use_fused_rope:
            # Transpose to BHSD first (mlx::rope expects BHSD)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            q = torch.ops.mlx.rope(
                q,
                self.rope_dims,
                start_pos,
                self.rope_traditional,
                self.rope_base,
                1.0,
                None,
            )
            k = torch.ops.mlx.rope(
                k,
                self.rope_dims,
                start_pos,
                self.rope_traditional,
                self.rope_base,
                1.0,
                None,
            )
        else:
            # Fallback: upstream rope (handles scaled rope, partial rotary, etc.)
            q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        k, v = self.kv_cache.update(start_pos, k, v)

        output = torch.ops.mlx.custom_sdpa(
            q,
            k,
            v,
            start_pos=start_pos,
            is_causal=True,
            scale=self.head_dim**-0.5,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), None
