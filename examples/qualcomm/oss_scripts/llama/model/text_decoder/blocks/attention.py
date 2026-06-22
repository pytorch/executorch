# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import Dict, List, Optional, Tuple, Type

import scipy
import torch
import torch.nn as nn

from executorch.examples.models.llama.model_args import ModelArgs

from ..rope import ROPE_REGISTRY


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_attention(model_architecture: str):
    """Register an attention class by model_architecture, mirroring
    FEED_FORWARD_REGISTRY: LlamaAttention is the default fallback (not
    registered), while gemma4 registers its own class, which internally derives
    self vs cross (KV-shared) behaviour from config.is_kv_shared_layer(layer_idx)."""

    def decorator(cls: Type[nn.Module]):
        ATTENTION_REGISTRY[model_architecture] = cls
        return cls

    return decorator


class LlamaAttention(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.use_kv_cache = config.use_kv_cache
        self.output_new_cache_only = output_new_cache_only
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_before_rope = config.qk_norm_before_rope
        # If None, assume each layer uses rope
        self.use_rope = (
            config.no_rope_layer_interval is None
            or (layer_idx + 1) % config.no_rope_layer_interval
        )

        if self.use_qk_norm:
            q_norm_dim = self.head_dim
            k_norm_dim = self.head_dim
            self.q_norm_fn = torch.nn.RMSNorm(q_norm_dim, eps=config.norm_eps)
            self.k_norm_fn = torch.nn.RMSNorm(k_norm_dim, eps=config.norm_eps)

        if config.partial_rotary_factor < 1:
            self.apply_rope_emb = ROPE_REGISTRY["partial"]
        else:
            self.apply_rope_emb = ROPE_REGISTRY["default"]

        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=config.attention_qkv_bias,
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_qkv_bias,
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_qkv_bias,
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = (
            float(self.head_dim) ** 0.5
            if config.attention_multiplier is None
            else 1.0 / config.attention_multiplier
        )

        # gemma 2 uses soft-capping on attention and logits
        self.attn_logit_softcapping = config.attn_logit_softcapping

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(self.head_dim, dtype=float)
                    / math.sqrt(self.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

    def prepare_attention_conv(self):
        self.wq_conv = nn.Conv2d(
            self.dim,
            self.n_heads * self.head_dim,
            1,
            bias=self.config.attention_qkv_bias,
        )
        self.wk_conv = nn.Conv2d(
            self.dim,
            self.n_kv_heads * self.head_dim,
            1,
            bias=self.config.attention_qkv_bias,
        )
        self.wv_conv = nn.Conv2d(
            self.dim,
            self.n_kv_heads * self.head_dim,
            1,
            bias=self.config.attention_qkv_bias,
        )
        self.wo_conv = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_attention_conv

        self.wq_conv.weight.data.copy_(self.wq.weight[:, :, None, None])
        if self.wq_conv.bias is not None:
            self.wq_conv.bias.data.copy_(self.wq.bias)
        self.wk_conv.weight.data.copy_(self.wk.weight[:, :, None, None])
        if self.wk_conv.bias is not None:
            self.wk_conv.bias.data.copy_(self.wk.bias)
        self.wv_conv.weight.data.copy_(self.wv.weight[:, :, None, None])
        if self.wv_conv.bias is not None:
            self.wv_conv.bias.data.copy_(self.wv.bias)
        self.wo_conv.weight.data.copy_(self.wo.weight[:, :, None, None])

        del self.wq
        del self.wk
        del self.wv
        del self.wo

    def forward_attention_conv(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)

        q = self.wq_conv(hidden_states)
        k = self.wk_conv(hidden_states)
        v = self.wv_conv(hidden_states)
        q = q.permute(0, 3, 1, 2).squeeze(-1)
        k = k.permute(0, 3, 1, 2).squeeze(-1)
        v = v.permute(0, 3, 1, 2).squeeze(-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        kh, vh = None, None
        # kv cache mode
        if self.use_kv_cache:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        # batch_prefill mode
        else:
            kh = k
            vh = v

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2)
        y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
        y = self.wo_conv(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        if self.output_new_cache_only:
            return y, [k], [v]

        return y, [kh], [vh]

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        kh, vh = None, None
        # kv cache mode
        if self.use_kv_cache:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        # batch_prefill mode
        else:
            kh = k
            vh = v

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        # gemma2-2b
        if self.attn_logit_softcapping is not None:
            attn = attn / self.attn_logit_softcapping
            attn = torch.tanh(attn)
            attn = attn * self.attn_logit_softcapping
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        y = self.wo(y)

        if self.output_new_cache_only:
            return y, [k], [v]

        return y, [kh], [vh]


@register_attention("gemma4")
class Gemma4Attention(nn.Module):
    """Gemma4 attention. Self layers compute their own K/V (and donor layers
    expose them for YOCO sharing); cross layers (KV-shared) reuse a donor
    layer's K/V and produce no new cache. All layers keep the K/V projections
    (checkpoint parity); config.is_kv_shared_layer(layer_idx) only changes the
    forward path, not the parameter set."""

    def __init__(
        self,
        layer_idx: int,
        config: ModelArgs,
        output_new_cache_only: bool = True,
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.use_kv_cache = config.use_kv_cache
        self.output_new_cache_only = output_new_cache_only
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)

        # Full-attention layers use global_head_dim; sliding layers use head_dim.
        self.head_dim = config.get_head_dim(layer_idx)

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Gemma4 always applies QK-norm (before rope) and V-norm. KV-shared
        # (cross) layers keep the K/V projections for checkpoint parity even
        # though their forward reuses a donor's K/V instead.
        self.q_norm_fn = torch.nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm_fn = torch.nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.v_norm_fn = torch.nn.RMSNorm(
            self.head_dim, eps=config.norm_eps, elementwise_affine=False
        )

        # Sliding layers apply full rope; full-attention layers use partial rope.
        if config.is_sliding_attention(layer_idx):
            self.apply_rope_emb = ROPE_REGISTRY["default"]
        else:
            self.apply_rope_emb = ROPE_REGISTRY["partial"]

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        # Gemma 4 uses scale=1.0; QK-norm handles normalisation.
        self.scale = 1.0

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(self.head_dim, dtype=float)
                    / math.sqrt(self.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

    def prepare_attention_conv(self):
        self.wq_conv = nn.Conv2d(self.dim, self.n_heads * self.head_dim, 1, bias=False)
        self.wo_conv = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)
        self.wk_conv = nn.Conv2d(
            self.dim, self.n_kv_heads * self.head_dim, 1, bias=False
        )
        self.wv_conv = nn.Conv2d(
            self.dim, self.n_kv_heads * self.head_dim, 1, bias=False
        )
        self.wq_conv.weight.data.copy_(self.wq.weight[:, :, None, None])
        self.wo_conv.weight.data.copy_(self.wo.weight[:, :, None, None])
        self.wk_conv.weight.data.copy_(self.wk.weight[:, :, None, None])
        self.wv_conv.weight.data.copy_(self.wv.weight[:, :, None, None])
        del self.wq
        del self.wo
        del self.wk
        del self.wv

        self.forward_no_conv = self.forward
        self.forward = self.forward_attention_conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: Optional[torch.Tensor] = None,
        v_caches: Optional[torch.Tensor] = None,
        donor_k: Optional[torch.Tensor] = None,
        donor_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.wq(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm_fn(q)
        q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)

        if self.is_kv_shared_layer:
            kh = repeat_kv(donor_k, self.num_key_value_groups)
            vh = repeat_kv(donor_v, self.num_key_value_groups)
            attn = q @ kh
            attn = attn / self.scale
            if self.enable_masked_softmax:
                attn_min = torch.amin(attn, dim=-1, keepdim=True)
                minus_value = -20
                attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
            else:
                attn = attn + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh
            y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
            return self.wo(y), None, None

        k = self.wk(hidden_states)
        v = self.wv(hidden_states)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm_fn(k)
        v = self.v_norm_fn(v)
        k = self.apply_rope_emb(k, freqs_cos, freqs_sin)
        if getattr(self.config, "enable_r3", False):
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        if k_caches is not None:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        else:
            kh = k
            vh = v

        new_k = k if self.output_new_cache_only else kh
        new_v = v if self.output_new_cache_only else vh

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)
        attn = q @ kh
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(y), new_k, new_v

    def forward_attention_conv(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: Optional[torch.Tensor] = None,
        v_caches: Optional[torch.Tensor] = None,
        donor_k: Optional[torch.Tensor] = None,
        donor_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)

        q = self.wq_conv(hidden_states)
        q = q.permute(0, 3, 1, 2).squeeze(-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm_fn(q)
        q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)

        if self.is_kv_shared_layer:
            kh = repeat_kv(donor_k, self.num_key_value_groups)
            vh = repeat_kv(donor_v, self.num_key_value_groups)
            attn = q @ kh
            attn = attn / self.scale
            if self.enable_masked_softmax:
                attn_min = torch.amin(attn, dim=-1, keepdim=True)
                minus_value = -20
                attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
            else:
                attn = attn + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh
            y = y.transpose(1, 2)
            y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
            y = self.wo_conv(y)
            y = y.transpose(1, 3)
            y = y.reshape(bsz, seq_len, -1)
            return y, None, None

        k = self.wk_conv(hidden_states)
        v = self.wv_conv(hidden_states)
        k = k.permute(0, 3, 1, 2).squeeze(-1)
        v = v.permute(0, 3, 1, 2).squeeze(-1)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm_fn(k)
        v = self.v_norm_fn(v)
        k = self.apply_rope_emb(k, freqs_cos, freqs_sin)
        if getattr(self.config, "enable_r3", False):
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        if k_caches is not None:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        else:
            kh = k
            vh = v

        new_k = k if self.output_new_cache_only else kh
        new_v = v if self.output_new_cache_only else vh

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)
        attn = q @ kh
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2)
        y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
        y = self.wo_conv(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)
        return y, new_k, new_v
