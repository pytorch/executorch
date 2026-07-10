"""PyTorch implementation of the DFlash draft model for ExecuTorch export.

This model is the lightweight "draft" network used in DFlash speculative
decoding. Instead of generating one token at a time like the target LLM, it
predicts an entire block of future tokens in parallel. To do this, it takes:
    - proposal tokens (the draft block, beginning with the last accepted token),
    - hidden states extracted from the target model (Phase 1), and
    - position IDs for the draft block.

The target hidden states are first projected into the draft model's hidden
space, then every draft transformer layer attends to both the projected target
context and the proposal tokens. The result is a fast approximation of what
the target model is likely to generate next.

The implementation is model-agnostic. Architectural differences such as RoPE,
sliding-window attention, embedding scale, and logit softcapping come from
DFlashConfig, allowing the same code to export draft models for Qwen3, Gemma,
Llama, and other standard-attention architectures.

For ExecuTorch export, the draft model owns its own embedding and LM head
weights (copied from the target during export) and returns final draft logits
directly rather than intermediate hidden states.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


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
    # Some models scale token embeddings before entering transformer.
    # Qwen3/Llama use 1.0, while Gemma scales by sqrt(hidden_size).
    embed_scale: float = 1.0


def _rope_inv_freq(config: DFlashConfig) -> torch.Tensor:
    # Build the RoPE frequencies expected by draft checkpoint.
    # Different model families use different RoPE scaling strategies, so this is driven entirely from the checkpoint config rather than hardcoded.
    dim = config.head_dim
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    scaling = config.rope_scaling or {}
    if scaling.get("rope_type", scaling.get("type")) == "linear":
        inv_freq = inv_freq / float(scaling["factor"])
    return inv_freq


class DFlashRotaryEmbedding(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.register_buffer("inv_freq", _rope_inv_freq(config), persistent=False)

    def forward(self, position_ids: torch.Tensor):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)  # [B, S, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Repeat KV heads when the model has fewer KV heads than attention heads.
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


def apply_rotary_pos_emb(q, k, cos, sin):
    # The proposal block only contains the current draft tokens, so queries rotate over those positions only.
    # Keys contain both the target context and proposal block, so they rotate over the full combined sequence.
    q_len = q.shape[-2]
    cq, sq = cos[:, None, -q_len:, :], sin[:, None, -q_len:, :]
    ck, sk = cos[:, None, :, :], sin[:, None, :, :]
    return (q * cq) + (rotate_half(q) * sq), (k * ck) + (rotate_half(k) * sk)


class DFlashRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashAttention(nn.Module):
    """Proposal tokens generate the queries. These represent the positions whose contents we are trying to predict."""

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        h, hd = config.hidden_size, config.head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = hd
        self.scaling = hd**-0.5
        self.n_rep = self.n_heads // self.n_kv
        lt = config.layer_types
        self.is_sliding = bool(lt) and lt[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(h, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, h, bias=False)
        self.q_norm = DFlashRMSNorm(hd, config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(hd, config.rms_norm_eps)

    def forward(self, x, x_ctx, cos, sin):
        """Keys and values come frmo both the projected target context and the proposal block itself. This lets the draft attend to what the target model already understands while also allowing predictions within the proposal block to interact with one another."""
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        q = self.q_norm(
            self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        ).transpose(1, 2)
        k = torch.cat([self.k_proj(x_ctx), self.k_proj(x)], dim=1).view(
            B, S + L, self.n_kv, self.head_dim
        )
        v = torch.cat([self.v_proj(x_ctx), self.v_proj(x)], dim=1).view(
            B, S + L, self.n_kv, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
        mask = self._sliding_mask(L, S, q.device, q.dtype) if self.is_sliding else None
        # Each proposal position attends over the combined context to build a richer representation before predicting its token.
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=self.scaling
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))

    def _sliding_mask(self, L, S, device, dtype):
        """Restrict attention to the configured sliding window for models that use sliding-window attention, like Gemma."""
        total = S + L
        q_pos = torch.arange(S, total, device=device)[:, None]
        k_pos = torch.arange(total, device=device)[None, :]
        allowed = (k_pos <= q_pos) & (k_pos > q_pos - self.sliding_window)
        return torch.where(allowed, 0.0, float("-inf")).to(dtype)[None, None]


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, x, x_ctx, cos, sin):
        # Standard transformer decoder block:
        # RMSNorm --> Attention --> Residual
        # RMSNorm --> MLP --> Residual
        x = x + self.self_attn(self.input_layernorm(x), x_ctx, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DFlashRotaryEmbedding(config)
        # The draft owns its own embedding and LM head weights.
        # During export these are copied from the target model, making the draft .pte self-contained.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens, target_hidden, position_ids):
        # Embed the proposal block (last accepted token + masked future positions).
        h = self.embed_tokens(tokens) * self.config.embed_scale
        # Translate the concatenated target hidden states into the draft model's hidden space.
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        # Positional information for both the proposal block and target context.
        cos, sin = self.rotary_emb(position_ids)
        for layer in self.layers:
            h = layer(h, h_ctx, cos, sin)
        h = self.norm(h)
        # Only return predictions for the future positions.
        logits = self.lm_head(h[:, 1:, :])
        # logits_start=1: drop the known first token
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits


def load_dflash_config(checkpoint_dir) -> "DFlashConfig":
    """Load the architecture needed to reconstruct a DFlash draft model.

    The checkpoint config describes both underlying transformer architecture (hidden size, attention heads, RoPE, etc.) and the DFlash-specific settings such as the tapped target layers and mask token.
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
    )
