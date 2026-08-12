# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash draft model for Qwen3. 

Predicts a block of future tokens in parallel using hidden states from the target model. 
It reuses HuggingFace Qwen3 modules while adapting attention for DFlash's block/context pattern. 
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    Qwen3Attention,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    target_layer_ids: Tuple[int, ...]
    block_size: int = 16
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    layer_types: Tuple[str, ...] = field(default_factory=tuple)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    embed_scale: float = 1.0  # Qwen3/Llama use 1.0


def _to_qwen3_config(config: DFlashConfig) -> Qwen3Config:
    """Buld a Qwen3Config from the DFlash checkpoint configuration. This allows the draft to reuse HuggingFace's Qwen3 modules."""
    rope_parameters = dict(config.rope_scaling or {})
    rope_parameters["rope_type"] = rope_parameters.pop(
        "type", rope_parameters.get("rope_type", "default")
    )
    rope_parameters["rope_theta"] = config.rope_theta

    qwen3_config = Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_parameters=rope_parameters,
        attention_bias=False,
        layer_types=list(config.layer_types) or None,
        sliding_window=config.sliding_window,
    )
    qwen3_config._attn_implementation = "mlx"  # Match the target's attention dispatch
    return qwen3_config


class DFlashQwen3Attention(Qwen3Attention):
    """DFlash attention where block queries attend to both target context and the block. Reuses Qwen3 projections, RoPE, normalization, and attention dispatch."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False  # The proposal block uses bidirectional attention

    def forward(
        self,
        x: torch.Tensor,
        x_ctx: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache=None,
        cache_position=None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q_shape = (B, L, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)

        if cache is None:
            # The uncached path reprojects the full target context each round.
            # It serves as the reference behavior for the cached implementation.
            S = x_ctx.shape[1]
            kv_shape = (B, S + L, -1, self.head_dim)
            kv_input = torch.cat([x_ctx, x], dim=1)
            key_states = self.k_norm(self.k_proj(kv_input).view(kv_shape)).transpose(
                1, 2
            )
            value_states = self.v_proj(kv_input).view(kv_shape).transpose(1, 2)
            total_len = cos.shape[1]
            q_cos = cos.narrow(1, total_len - L, L)
            q_sin = sin.narrow(1, total_len - L, L)
            query_states, _ = apply_rotary_pos_emb(
                query_states, query_states, q_cos, q_sin
            )
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
            attention_mask = None
        else:
            # Only newly confirmed context is projected and added to the cache.
            # The valid cache range is narrowed before attention to avoid computing over padded positions, keeping cached attention efficient as the context grows.
            new_ctx_len = x_ctx.shape[1]
            new_shape = (B, new_ctx_len, -1, self.head_dim)
            new_k = self.k_norm(self.k_proj(x_ctx).view(new_shape)).transpose(1, 2)
            new_v = self.v_proj(x_ctx).view(new_shape).transpose(1, 2)

            ctx_cos = cos.narrow(1, 0, new_ctx_len)
            ctx_sin = sin.narrow(1, 0, new_ctx_len)
            _, new_k = apply_rotary_pos_emb(new_k, new_k, ctx_cos, ctx_sin)
            blk_cos = cos.narrow(1, new_ctx_len, L)
            blk_sin = sin.narrow(1, new_ctx_len, L)
            query_states, _ = apply_rotary_pos_emb(
                query_states, query_states, blk_cos, blk_sin
            )

            full_k, full_v = cache.write(self.layer_idx, new_k, new_v, cache_position)

            blk_shape = (B, L, -1, self.head_dim)
            blk_k = self.k_norm(self.k_proj(x).view(blk_shape)).transpose(1, 2)
            blk_v = self.v_proj(x).view(blk_shape).transpose(1, 2)
            _, blk_k = apply_rotary_pos_emb(blk_k, blk_k, blk_cos, blk_sin)

            # Use export-safe checks when slicing the cache to its valid length.
            valid_len = cache.valid_len_after(cache_position).item()
            torch._check(valid_len >= 1)
            torch._check(valid_len <= full_k.shape[2])
            ctx_k = full_k.narrow(2, 0, valid_len)
            ctx_v = full_v.narrow(2, 0, valid_len)
            key_states = torch.cat([ctx_k, blk_k], dim=2)
            value_states = torch.cat([ctx_v, blk_v], dim=2)
            attention_mask = None

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=None,
        )
        attn_output = attn_output.reshape(B, L, -1).contiguous()
        return self.o_proj(attn_output)


class DFlashQwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer adapted for DFlash's block/context attention pattern.
    Standard Qwen3 normalization and MLP modules are reused unchanged."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashQwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, x, x_ctx, cos, sin, cache=None, cache_position=None):
        x = x + self.self_attn(
            self.input_layernorm(x),
            x_ctx,
            cos,
            sin,
            cache=cache,
            cache_position=cache_position,
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.qwen3_config = _to_qwen3_config(config)
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(self.qwen3_config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(self.qwen3_config)
        # Keep embedded and LM head inside the draft so the exported model is self-contained.
        # Their weights are copied from the target during export.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens, target_hidden, cache=None, cache_position=None):
        # Uncached mode receives the full target context; cached mode receives only newly confirmed States
        # Positions are built from the target context when uncached, or from the cached position followed by the proposal block when cached.
        block_len = tokens.shape[1]
        h = self.embed_tokens(tokens) * self.config.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))

        if cache is None:
            ctx_len = target_hidden.shape[1]
            position_ids = torch.arange(
                ctx_len + block_len, device=tokens.device
            ).unsqueeze(0)
        else:
            block_start = cache_position[-1] + 1
            block_positions = block_start + torch.arange(
                block_len, device=tokens.device
            )
            position_ids = torch.cat([cache_position, block_positions]).unsqueeze(0)

        cos, sin = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            h = layer(h, h_ctx, cos, sin, cache=cache, cache_position=cache_position)
        h = self.norm(h)

        logits = self.lm_head(h[:, 1:, :])  # drop the known first token
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits


def load_dflash_config(checkpoint_dir) -> "DFlashConfig":
    """Load Qwen3 architecture parameters and DFlash-specific settings from config.json.
    This includes the tapped target layers, mask token, and other draft configuration.
    """
    import json
    from pathlib import Path

    cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    dcfg = cfg["dflash_config"]
    return DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        target_layer_ids=tuple(dcfg["target_layer_ids"]),
        block_size=cfg["block_size"],
        mask_token_id=dcfg["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=tuple(
            cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
        ),
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
        embed_scale=cfg.get("embed_scale", dcfg.get("embed_scale", 1.0)),
    )
