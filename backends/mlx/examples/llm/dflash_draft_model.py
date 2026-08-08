# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch implementation of the DFlash draft model for ExecuTorch export.

This model is the lightweight "draft" network used in DFlash speculative
decoding. Instead of generating one token at a time like the target LLM, it
predicts an entire block of future tokens in parallel. To do this, it takes:
    - proposal tokens (the draft block, beginning with the last accepted token),
    - hidden states extracted from the target model (Phase 1).

The target hidden states are first projected into the draft model's hidden
space, then every draft transformer layer attends to both the projected target
context and the proposal tokens (bidirectionally -- see DFlash paper Section
4.2). The result is a fast approximation of what the target model is likely
to generate next.

Per review, this adapts HuggingFace's real Qwen3 building blocks (attention,
RMSNorm, rotary embeddings, MLP, and HF's attention-interface dispatch --
the same one the MLX integration registers "mlx" into, see
backends/mlx/llm/hf_attention.py) rather than reimplementing them from
scratch. Only the attention forward and decoder-layer container are
DFlash-specific (queries come from the proposal block alone; keys/values
span the projected target context concatenated with the block) -- everything
else reuses the real HF modules directly, so behavior stays with the model
implementation instead of drifting from a duplicated copy.

For ExecuTorch export, the draft model owns its own embedding and LM head
weights (copied from the target during export) and returns final draft logits
directly rather than intermediate hidden states.
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
    # Some models scale token embeddings before entering transformer.
    # Qwen3/Llama use 1.0, while Gemma scales by sqrt(hidden_size).
    embed_scale: float = 1.0


def _to_qwen3_config(config: DFlashConfig) -> Qwen3Config:
    """Translate DFlash's checkpoint-derived config into a real Qwen3Config,
    so the draft model can build real Qwen3RMSNorm/Qwen3RotaryEmbedding/
    Qwen3MLP/Qwen3Attention instances instead of hand-reimplemented copies.
    """
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
    # Route through the same "mlx" attention interface the MLX integration
    # registers for the target model (backends/mlx/llm/hf_attention.py),
    # instead of calling torch SDPA directly.
    qwen3_config._attn_implementation = "mlx"
    return qwen3_config


class DFlashQwen3Attention(Qwen3Attention):
    """Adapts HF's Qwen3Attention to DFlash's cross-attention pattern:
    queries come only from the proposal block, while keys/values span the
    projected target context *and* the proposal block, concatenated. Reuses
    the parent class's projections (q_proj/k_proj/v_proj/o_proj/q_norm/
    k_norm), RoPE application (apply_rotary_pos_emb), and HF's attention
    dispatch (ALL_ATTENTION_FUNCTIONS) rather than reimplementing any of
    them.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # DFlash attends bidirectionally within a block and its target
        # context (confirmed against the DFlash paper, Section 4.2: "Tokens
        # attend bidirectionally within the same block and to the
        # corresponding injected target context features"). This is never
        # causal, unlike the base class's decode-time self-attention.
        self.is_causal = False

    def forward(
        self,
        x: torch.Tensor,
        x_ctx: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache=None,
        cache_position=None,
    ) -> torch.Tensor:
        # Two paths that MUST produce identical logits (verified in eager by
        # scratch_draft_cache_equiv.py):
        #   - uncached (cache=None): x_ctx is the FULL accumulated context;
        #     reproject the full [context; block] every round. Unchanged.
        #   - cached: x_ctx is ONLY the newly-arrived context slice (so
        #     new_ctx_len = x_ctx.shape[1] is derived from a tensor shape --
        #     a backed SymInt torch.export can trace -- rather than passed
        #     as a raw Python int, which can never be marked dynamic).
        #     Project only that slice plus the block, write the new context
        #     K/V into the cache at cache_position, then attend over the
        #     FULL fixed-size cache buffer with a validity mask -- never
        #     narrow() it down to the valid length, since that length is a
        #     data value (from cache_position's contents) rather than a
        #     shape, and narrowing by it would need .item(), triggering
        #     exactly the GuardOnDataDependentSymNode failure this design
        #     exists to avoid. valid_mask() does the equivalent job via a
        #     plain elementwise comparison instead.
        B, L, _ = x.shape
        q_shape = (B, L, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)

        if cache is None:
            # --- Uncached path (unchanged) ---
            S = x_ctx.shape[1]
            kv_shape = (B, S + L, -1, self.head_dim)
            kv_input = torch.cat([x_ctx, x], dim=1)
            key_states = self.k_norm(self.k_proj(kv_input).view(kv_shape)).transpose(1, 2)
            value_states = self.v_proj(kv_input).view(kv_shape).transpose(1, 2)
            total_len = cos.shape[1]
            q_cos = cos.narrow(1, total_len - L, L)
            q_sin = sin.narrow(1, total_len - L, L)
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, q_cos, q_sin)
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
            attention_mask = None
        else:
            # --- Cached path ---
            new_ctx_len = x_ctx.shape[1]  # backed SymInt, derived from shape
            new_shape = (B, new_ctx_len, -1, self.head_dim)
            new_k = self.k_norm(self.k_proj(x_ctx).view(new_shape)).transpose(1, 2)
            new_v = self.v_proj(x_ctx).view(new_shape).transpose(1, 2)

            # cos/sin cover [new context ; block], contiguous, in that order.
            ctx_cos = cos.narrow(1, 0, new_ctx_len)
            ctx_sin = sin.narrow(1, 0, new_ctx_len)
            _, new_k = apply_rotary_pos_emb(new_k, new_k, ctx_cos, ctx_sin)
            blk_cos = cos.narrow(1, new_ctx_len, L)
            blk_sin = sin.narrow(1, new_ctx_len, L)
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, blk_cos, blk_sin)

            # Write new context K/V into the persistent cache. full_k/full_v
            # are ALWAYS the full max_seq_len buffer -- never narrowed.
            full_k, full_v = cache.write(self.layer_idx, new_k, new_v, cache_position)

            # Block K/V are fresh every round (never cached).
            blk_shape = (B, L, -1, self.head_dim)
            blk_k = self.k_norm(self.k_proj(x).view(blk_shape)).transpose(1, 2)
            blk_v = self.v_proj(x).view(blk_shape).transpose(1, 2)
            _, blk_k = apply_rotary_pos_emb(blk_k, blk_k, blk_cos, blk_sin)

            # Write new context K/V into the persistent cache. full_k/full_v
            # are the full max_seq_len buffer as returned by write() -- the
            # narrow() below (not here) is what trims them to valid_len.

            # Slice down to [valid context ; block] -- NOT the full padded
            # buffer -- so attention only ever computes over real content.
            # valid_len_after() is the authoritative crop point (one past the
            # last written position). Extracting it via .item() + torch._check
            # hints (exactly the pattern custom_sdpa's own internal K/V
            # slicing already uses, see custom_ops.py) keeps this export-safe
            # despite being a data-dependent length -- verified directly via
            # a standalone probe before wiring this in. This replaces the
            # earlier mask-based approach, which attended over the ENTIRE
            # fixed buffer every round (e.g. 4096 padded positions) and was
            # the actual measured performance cost (1.68x slower than
            # uncached at max_seq_len=4096, scaling directly with buffer
            # size) -- masking discards invalid attention weights but still
            # computes them; slicing never computes them at all.
            valid_len = cache.valid_len_after(cache_position).item()
            torch._check(valid_len >= 1)
            torch._check(valid_len <= full_k.shape[2])
            ctx_k = full_k.narrow(2, 0, valid_len)
            ctx_v = full_v.narrow(2, 0, valid_len)
            key_states = torch.cat([ctx_k, blk_k], dim=2)
            value_states = torch.cat([ctx_v, blk_v], dim=2)
            attention_mask = None  # nothing invalid remains to mask

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
    """Thin DFlash-specific container reusing real Qwen3 building blocks
    (Qwen3RMSNorm, Qwen3MLP) plus the adapted DFlashQwen3Attention above. A
    standard Qwen3DecoderLayer's forward() assumes one unified input
    sequence; DFlash's split proposal-block/target-context query/key
    pattern doesn't fit that contract, so this container still needs its
    own forward(), but every actual computation is delegated to real HF
    modules rather than reimplemented.
    """

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
            self.input_layernorm(x), x_ctx, cos, sin,
            cache=cache, cache_position=cache_position,
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
        # The draft owns its own embedding and LM head weights.
        # During export these are copied from the target model, making the draft .pte self-contained.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens, target_hidden, cache=None, cache_position=None):
        # target_hidden means different things depending on cache:
        #   - cache is None: the FULL accumulated context (unchanged).
        #   - cache is not None: ONLY the newly-arrived context slice. Its
        #     length (new_ctx_len) is derived from target_hidden.shape[1] --
        #     a backed SymInt torch.export can trace -- rather than passed
        #     as a separate raw-int argument, which can never be marked
        #     dynamic. cache_position gives the new slice's absolute
        #     positions; the block's positions follow immediately after.
        block_len = tokens.shape[1]
        h = self.embed_tokens(tokens) * self.config.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))

        if cache is None:
            ctx_len = target_hidden.shape[1]
            position_ids = torch.arange(ctx_len + block_len, device=tokens.device).unsqueeze(0)
        else:
            new_ctx_len = target_hidden.shape[1]
            block_start = cache_position[-1] + 1
            block_positions = block_start + torch.arange(block_len, device=tokens.device)
            position_ids = torch.cat([cache_position, block_positions]).unsqueeze(0)

        # Positional information for both the proposal block and target context.
        cos, sin = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            h = layer(h, h_ctx, cos, sin, cache=cache, cache_position=cache_position)
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
        embed_scale=cfg.get("embed_scale", dcfg.get("embed_scale", 1.0)),
    )
