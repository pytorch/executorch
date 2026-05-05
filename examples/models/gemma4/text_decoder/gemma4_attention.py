# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Attention module.

Features:
- Per-layer head_dim (global_head_dim=512 for full attention, head_dim=256 for sliding)
- Partial RoPE for full attention layers (only first 25% of dims)
- V-norm (RMSNorm without weight)
- QK-norm applied before RoPE
- MQA with num_key_value_heads=1
- KV cache for autoregressive generation
- YOCO support for KV sharing
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .gemma4_config import Gemma4Config
from .gemma4_norm import RMSNorm, RMSNormNoWeight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (HuggingFace style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K (HuggingFace style).

    Args:
        xq: Query tensor of shape [batch, num_heads, seq_len, rotary_dim]
        xk: Key tensor of shape [batch, num_kv_heads, seq_len, rotary_dim]
        freqs_cos: Cosine frequencies [seq_len, rotary_dim]
        freqs_sin: Sine frequencies [seq_len, rotary_dim]

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    xq_out = (xq.float() * freqs_cos) + (rotate_half(xq.float()) * freqs_sin)
    xk_out = (xk.float() * freqs_cos) + (rotate_half(xk.float()) * freqs_sin)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_single(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a single tensor (Q only)."""
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    x_out = (x.float() * freqs_cos) + (rotate_half(x.float()) * freqs_sin)

    return x_out.type_as(x)


class Gemma4KVCache(nn.Module):
    """Key-Value cache for autoregressive generation.

    Args:
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        num_kv_heads: Number of key-value heads
        head_dim: Dimension per head
        dtype: Data type for cache tensors
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        use_index_copy: bool = False,
    ):
        super().__init__()
        self.use_index_copy = use_index_copy
        cache_shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V values.

        Returns:
            Tuple of (full_k, full_v) - returns entire cache
        """
        if self.use_index_copy:
            self.k_cache.index_copy_(2, input_pos, k_val)
            self.v_cache.index_copy_(2, input_pos, v_val)
        else:
            seq_len = k_val.size(2)
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos >= 0)
            self.k_cache.narrow(2, start_pos, seq_len).copy_(k_val)
            self.v_cache.narrow(2, start_pos, seq_len).copy_(v_val)

        return self.k_cache, self.v_cache


class Gemma4Attention(nn.Module):
    """Gemma 4 attention with per-layer head_dim, partial RoPE, and MQA.

    Args:
        config: Gemma4Config with model parameters
        layer_idx: Index of this layer (for attention type selection)
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.get_head_dim(layer_idx)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.is_sliding = config.is_sliding_attention(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.rope_theta = config.get_rope_theta(layer_idx)

        # YOCO
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)
        self.kv_shared_layer_index = config.get_kv_shared_layer_index(layer_idx)
        self.is_kv_donor_layer = config.is_kv_donor_layer(layer_idx)

        # Gemma 4 uses scaling=1.0 since QK-norm handles normalization
        # Same convention as Gemma 3N (not Gemma 3 which uses query_pre_attn_scalar**-0.5)
        self.scaling = 1.0

        # QKV projections sized by per-layer head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # QKV norms sized by per-layer head_dim
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoWeight(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: for full attention, only rotate partial_rotary_factor of dims
        # but produce full head_dim cos/sin (zero-padded) to match HF's rotate_half pairing
        if not self.is_sliding:
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        else:
            self.rotary_dim = self.head_dim

        # RoPE: store only inv_freq; cos/sin computed on the fly per forward.
        # Partial RoPE pads with zeros for non-rotated dims so rotate_half pairs correctly.
        rope_angles = self.rotary_dim // 2
        inv_freq_rotated = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2).float() / self.head_dim)
        )
        nope_angles = self.head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)])
        else:
            inv_freq = inv_freq_rotated
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # KV cache — skip allocation for shared layers (they use donor's KV)
        self.use_index_copy = config.use_index_copy_for_kv_cache
        self.kv_cache: Optional[Gemma4KVCache] = None
        if config.use_kv_cache and not self.is_kv_shared_layer:
            self.kv_cache = Gemma4KVCache(
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_index_copy=self.use_index_copy,
            )

        self.use_custom_sdpa = config.use_custom_sdpa
        if self.use_custom_sdpa:
            from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        # Mask buffers — shared across layers by Gemma4TextModel._share_masks()
        # to avoid duplicating identical [max_seq_len x max_seq_len] tensors.
        # Initialized here as fallback for standalone usage.
        self.register_buffer("causal_mask", None, persistent=False)
        self.register_buffer("sliding_window_mask", None, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for MQA/GQA."""
        if self.num_key_value_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1)
        return x.reshape(batch, self.num_heads, seq_len, head_dim)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE. For partial rotation, cos/sin are zero-padded to head_dim
        so rotate_half pairs dims correctly (matching HF)."""
        return apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    def _apply_rope_single(
        self,
        q: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to Q only (for shared KV layers)."""
        return apply_rotary_emb_single(q, freqs_cos, freqs_sin)

    def _get_rope_freqs(
        self,
        input_pos: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin from inv_freq for the current positions."""
        pos = input_pos if input_pos is not None else torch.arange(seq_len)
        freqs = torch.outer(pos.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

    def _slice_mask(
        self,
        base_mask: torch.Tensor,
        input_pos: torch.Tensor,
        seq_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        """Slice a [max_seq_len, max_seq_len] mask to current query positions x cache."""
        if self.use_index_copy:
            return torch.index_select(base_mask, 0, input_pos).narrow(1, 0, kv_len)
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos >= 0)
        return base_mask.narrow(0, start_pos, seq_len).narrow(1, 0, kv_len)

    def _build_attn_mask(
        self,
        input_pos: Optional[torch.Tensor],
        seq_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        """Combined causal + sliding-window mask for the current step."""
        using_cached_kv = (
            (self.kv_cache is not None or self.is_kv_shared_layer)
            and input_pos is not None
            and kv_len > seq_len
        )
        if using_cached_kv:
            mask = self._slice_mask(self.causal_mask, input_pos, seq_len, kv_len)
        else:
            mask = self.causal_mask[:seq_len, :seq_len]

        if self.sliding_window is not None and self.sliding_window_mask is not None:
            if using_cached_kv:
                sw_mask = self._slice_mask(
                    self.sliding_window_mask, input_pos, seq_len, kv_len
                )
            else:
                sw_mask = self.sliding_window_mask[:seq_len, :seq_len]
            mask = mask + sw_mask
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for attention.

        Args:
            hidden_states: Input of shape [batch, seq_len, hidden_size]
            input_pos: Current position(s) for KV cache update
            mask: Optional attention mask
            shared_kv: Optional tuple of (k, v) from donor layer for YOCO

        Returns:
            Tuple of:
            - Output tensor of shape [batch, seq_len, hidden_size]
            - Optional tuple of (k, v) to share with later layers (only for donor layers)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q projection
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        # For KV shared layers, use shared K/V from donor layer
        if self.is_kv_shared_layer and shared_kv is not None:
            k, v = shared_kv
            freqs_cos, freqs_sin = self._get_rope_freqs(input_pos, seq_len)
            q = self._apply_rope_single(q, freqs_cos, freqs_sin)
        else:
            # Compute K, V projections
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )

            # Apply KV norms
            k = self.k_norm(k)
            v = self.v_norm(v)

            # Get RoPE frequencies
            freqs_cos, freqs_sin = self._get_rope_freqs(input_pos, seq_len)

            # Apply RoPE (partial for full attention, full for sliding)
            q, k = self._apply_rope(q, k, freqs_cos, freqs_sin)

        if (
            self.kv_cache is not None
            and input_pos is not None
            and not self.is_kv_shared_layer
        ):
            k, v = self.kv_cache.update(input_pos, k, v)

        # Lazy dequant for INT8 KV cache: do it once here, before both
        # the donor-share path (cross-decoder layers can't see scales) and
        # the custom_sdpa branch. Using basic torch ops keeps it inside
        # the XNNPACK partition (no quantized_decomposed graph break).
        if (
            isinstance(self.kv_cache, Gemma4QuantizedKVCache)
            and not self.kv_cache.return_float_values
        ):
            k = k.to(torch.float32) * self.kv_cache.k_cache_scales
            v = v.to(torch.float32) * self.kv_cache.v_cache_scales

        kv_to_share: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.is_kv_donor_layer:
            kv_to_share = (k, v)

        if self.use_custom_sdpa and input_pos is not None:
            # Custom SDPA handles GQA/MQA natively (skips 8x KV expansion)
            # and tiles attention so the [seq x seq] matrix never materializes.
            kv_len = k.size(2)
            start_pos = 0 if self.use_index_copy else input_pos[0].item()
            attn_mask = self._build_attn_mask(input_pos, seq_len, kv_len)

            # custom_sdpa expects [bs, seq_len, n_heads, head_dim]
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)

            # custom_sdpa positional args: (q, k, v, start_pos, attn_mask, dropout, is_causal, scale).
            # The op schema has a typo (`drpout_p`); avoid kwargs.
            attn_output = torch.ops.llama.custom_sdpa(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                start_pos,
                attn_mask,
                0.0,
                False,
                self.scaling,
            )
            attn_output = attn_output.view(batch_size, seq_len, -1)
        else:
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            if mask is None:
                mask = self._build_attn_mask(input_pos, seq_len, k.size(2))

            attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                q
            )
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, kv_to_share


class Gemma4QuantizedKVCache(nn.Module):
    """INT8 Quantized Key-Value cache for Gemma4.

    Stores K and V tensors as int8 with symmetric per-token quantization.
    Uses simple torch ops (abs/amax, div, mul) that XNNPACK can fuse,
    avoiding quantized_decomposed ops that break graph partitioning.

    When return_float_values=False, returns raw INT8 K/V + exposes scales
    as attributes. The attention module does lazy inline dequant with
    simple ops right before custom_sdpa, keeping everything in one
    XNNPACK partition.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        use_index_copy: bool = False,
        return_float_values: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.use_index_copy = use_index_copy
        self.return_float_values = return_float_values

        cache_shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        scale_shape = (max_batch_size, num_kv_heads, max_seq_len, 1)

        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=torch.int8))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=torch.int8))
        self.register_buffer(
            "k_cache_scales", torch.ones(scale_shape, dtype=torch.float32)
        )
        self.register_buffer(
            "v_cache_scales", torch.ones(scale_shape, dtype=torch.float32)
        )

    def _quantize(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric per-token quantization using basic torch ops."""
        amax = value.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = amax / 127.0
        quantized = (value / scales).round().clamp(-128, 127).to(torch.int8)
        return quantized, scales

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V values.

        Returns:
            If return_float_values=True: (full_k, full_v) as dequantized float.
            If return_float_values=False: (full_k_int8, full_v_int8) — caller
            dequantizes lazily using k_cache_scales/v_cache_scales attributes.
        """
        quantized_k, k_scales = self._quantize(k_val)
        quantized_v, v_scales = self._quantize(v_val)

        if self.use_index_copy:
            self.k_cache.index_copy_(2, input_pos, quantized_k)
            self.v_cache.index_copy_(2, input_pos, quantized_v)
            self.k_cache_scales.index_copy_(2, input_pos, k_scales)
            self.v_cache_scales.index_copy_(2, input_pos, v_scales)
        else:
            seq_len = k_val.size(2)
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos >= 0)
            self.k_cache.narrow(2, start_pos, seq_len).copy_(quantized_k)
            self.v_cache.narrow(2, start_pos, seq_len).copy_(quantized_v)
            self.k_cache_scales.narrow(2, start_pos, seq_len).copy_(k_scales)
            self.v_cache_scales.narrow(2, start_pos, seq_len).copy_(v_scales)

        if not self.return_float_values:
            return self.k_cache, self.v_cache

        # Legacy path: full dequant + overwrite current pos with float original
        k_out = (self.k_cache.to(torch.float32) * self.k_cache_scales).to(self.dtype)
        v_out = (self.v_cache.to(torch.float32) * self.v_cache_scales).to(self.dtype)

        if self.use_index_copy:
            k_out.index_copy_(2, input_pos, k_val)
            v_out.index_copy_(2, input_pos, v_val)
        else:
            k_out.narrow(2, start_pos, seq_len).copy_(k_val)
            v_out.narrow(2, start_pos, seq_len).copy_(v_val)

        return k_out, v_out

    @classmethod
    def from_float(
        cls, kv_cache: Gemma4KVCache, return_float_values: bool = True
    ) -> "Gemma4QuantizedKVCache":
        """Create quantized KV cache from float KV cache."""
        max_batch_size, num_kv_heads, max_seq_len, head_dim = kv_cache.k_cache.shape
        dtype = kv_cache.k_cache.dtype
        return cls(
            max_batch_size,
            max_seq_len,
            num_kv_heads,
            head_dim,
            dtype,
            use_index_copy=kv_cache.use_index_copy,
            return_float_values=return_float_values,
        )


def replace_kv_cache_with_quantized_kv_cache(
    model: nn.Module,
    use_custom_sdpa: bool = False,
) -> nn.Module:
    """Replace Gemma4KVCache with Gemma4QuantizedKVCache in the model.

    When use_custom_sdpa=True, the quantized cache returns raw INT8 tensors
    for use with custom_quantized_sdpa (avoids full-cache dequant overhead).
    """
    return_float = not use_custom_sdpa
    return _replace_kv_cache_with_quantized_kv_cache(model, return_float)


def _replace_kv_cache_with_quantized_kv_cache(
    module: nn.Module,
    return_float_values: bool = True,
) -> nn.Module:
    """Recursively replace Gemma4KVCache with Gemma4QuantizedKVCache."""
    for name, child in module.named_children():
        if isinstance(child, Gemma4KVCache):
            setattr(
                module,
                name,
                Gemma4QuantizedKVCache.from_float(
                    child, return_float_values=return_float_values
                ),
            )
        else:
            _replace_kv_cache_with_quantized_kv_cache(child, return_float_values)
    return module
