#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX-optimized attention for ExecuTorch's Llama attention registry.

Registers an "mlx" attention type that uses mlx::kv_cache_update and
mlx::custom_sdpa for efficient execution on Apple Silicon.

Usage:
    import executorch.backends.mlx.llm.et_attention  # noqa: F401

    model_args = ModelArgs(attention_type="mlx", ...)
    transformer = construct_transformer(model_args)
"""

from typing import Any, Optional, Tuple, TYPE_CHECKING

import executorch.backends.mlx.custom_ops as _mlx_custom_ops  # noqa: F401
import torch
import torch.nn as nn
from executorch.backends.mlx.llm.cache import KVCache, RingBufferKVCache
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


def _check_mlx_unsupported(
    *,
    use_q_gate: bool,
    is_kv_shared_layer: bool,
    use_attn_o_norm: bool,
    use_attn_o_gate: bool,
    scale_query_by: float,
) -> None:
    """Raise if any attention option has no MLX implementation.

    Shared by __init__ and from_attention_mha so the two paths cannot drift.
    """
    unsupported = []
    if use_q_gate:
        unsupported.append("use_q_gate")
    if is_kv_shared_layer:
        unsupported.append("is_kv_shared_layer (YOCO)")
    if use_attn_o_norm:
        unsupported.append("use_attn_o_norm")
    if use_attn_o_gate:
        unsupported.append("use_attn_o_gate")
    if scale_query_by != 1.0:
        unsupported.append(f"scale_query_by={scale_query_by}")
    if unsupported:
        raise NotImplementedError(f"MLXAttentionMHA does not support: {unsupported}")


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

        # Fail loudly on unsupported options
        num_kv_shared = args.num_kv_shared_layers
        _check_mlx_unsupported(
            use_q_gate=args.use_q_gate,
            is_kv_shared_layer=num_kv_shared > 0
            and layer_id >= args.n_layers - num_kv_shared,
            use_attn_o_norm=args.use_attn_o_norm,
            use_attn_o_gate=args.use_attn_o_gate,
            scale_query_by=args.scale_query_by,
        )

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

        # Output gate support (Spark-X2.5 uses headwise_attn_output_gate)
        self.headwise_attn_output_gate = args.headwise_attn_output_gate

        # Sliding window support
        self.is_sliding = (
            args.layer_types is not None
            and layer_id < len(args.layer_types)
            and args.layer_types[layer_id] == "sliding_attention"
        )
        # Only set sliding_window for sliding layers (consistent with from_attention_mha)
        self.sliding_window = args.sliding_window if self.is_sliding else None

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

        # Output gate weight (Spark-X2.5: headwise gate, shape [dim, n_local_heads])
        if self.headwise_attn_output_gate:
            self.og = nn.Linear(self.dim, self.n_local_heads, bias=False)

        self.layer_id = layer_id
        self.rope = rope
        self.rope_base = rope.params.rope_freq_base
        self.use_fused_rope = self._can_use_fused_rope(rope.params)
        self.rope_traditional = not rope.params.use_hf_rope
        self.rope_dims = int(self.head_dim * rope.params.partial_rotary_factor)

        # Sliding-window layers use a ring buffer, but only when the context can
        # outgrow the window; otherwise a plain cache is smaller and equivalent.
        if self._uses_ring_buffer(args.max_context_len, self.sliding_window):
            self.kv_cache = RingBufferKVCache(
                max_batch_size=args.max_batch_size,
                max_context_length=self.sliding_window,
                n_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                max_write_len=min(args.max_seq_len, self.sliding_window),
            )
        else:
            self.kv_cache = KVCache(
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

    @staticmethod
    def _uses_ring_buffer(max_context_len: int, sliding_window: Optional[int]) -> bool:
        return sliding_window is not None and max_context_len > sliding_window

    @classmethod
    def from_attention_mha(
        cls, other: "AttentionMHA", dtype: Optional[torch.dtype] = None
    ) -> "MLXAttentionMHA":
        """
        Create an MLXAttentionMHA from an existing AttentionMHA.

        Shares weight references (wq, wk, wv, wo, rope, norm, og) and creates
        a fresh KVCache (RingBufferKVCache for sliding-window layers).
        """
        from executorch.examples.models.llama.attention import AttentionMHA

        assert isinstance(other, AttentionMHA)
        if not other.use_kv_cache:
            raise ValueError("MLXAttention requires use_kv_cache=True")

        # Fail loudly on unsupported AttentionMHA options
        _check_mlx_unsupported(
            use_q_gate=other.use_q_gate,
            is_kv_shared_layer=other.is_kv_shared_layer,
            use_attn_o_norm=other.use_attn_o_norm,
            use_attn_o_gate=other.use_attn_o_gate,
            scale_query_by=other.scale_query_by,
        )

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

        # Copy output gate fields (Spark-X2.5 uses headwise_attn_output_gate)
        # use_attn_o_gate and use_attn_o_norm are checked in unsupported list above
        instance.headwise_attn_output_gate = other.headwise_attn_output_gate

        # Copy sliding window fields
        instance.is_sliding = other.is_sliding
        instance.sliding_window = other.sliding_window

        # Share weight references
        instance.wq = other.wq
        instance.wk = other.wk
        instance.wv = other.wv
        instance.wo = other.wo
        instance.layer_id = other.layer_id
        instance.rope = other.rope

        # Copy RoPE params from the source layer's rope object
        # (each AttentionMHA layer has its own rope with correct params for its layer type)
        instance.rope_base = other.rope.params.rope_freq_base
        instance.use_fused_rope = cls._can_use_fused_rope(other.rope.params)
        instance.rope_traditional = not other.rope.params.use_hf_rope
        instance.rope_dims = int(
            instance.head_dim * other.rope.params.partial_rotary_factor
        )

        if other.use_qk_norm:
            instance.q_norm_fn = other.q_norm_fn
            instance.k_norm_fn = other.k_norm_fn

        # Share output gate weight
        if instance.headwise_attn_output_gate:
            instance.og = other.og

        # Create fresh MLX KV cache (ring buffer for sliding layers whose
        # context can outgrow the window)
        cache_dtype = dtype if dtype is not None else other.kv_cache.k_cache.dtype

        if cls._uses_ring_buffer(other.max_context_len, instance.sliding_window):
            instance.kv_cache = RingBufferKVCache(
                max_batch_size=other.max_batch_size,
                max_context_length=instance.sliding_window,
                n_heads=instance.n_kv_heads,
                head_dim=instance.head_dim,
                dtype=cache_dtype,
                max_write_len=min(other.max_seq_len, instance.sliding_window),
            )
        else:
            instance.kv_cache = KVCache(
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

        # Sliding-window layers use ring-buffer mask; full layers use causal
        if isinstance(self.kv_cache, RingBufferKVCache):
            sdpa_mask = self.kv_cache.create_sliding_window_mask(start_pos, seqlen)
            # seqlen <= max_write_len <= window keeps this non-negative
            sdpa_start = self.kv_cache.buffer_size - seqlen
            output = torch.ops.mlx.custom_sdpa(
                q,
                k,
                v,
                start_pos=sdpa_start,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.head_dim**-0.5,
            )
        else:
            output = torch.ops.mlx.custom_sdpa(
                q,
                k,
                v,
                start_pos=start_pos,
                is_causal=True,
                scale=self.head_dim**-0.5,
            )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Apply headwise output gate (Spark-X2.5: per-head sigmoid gate)
        if self.headwise_attn_output_gate:
            output_4d = output.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            og = self.og(x).unsqueeze(-1).to(output_4d.dtype)
            output_4d = torch.sigmoid(og) * output_4d
            output = output_4d.reshape(bsz, seqlen, -1)

        return self.wo(output), None
