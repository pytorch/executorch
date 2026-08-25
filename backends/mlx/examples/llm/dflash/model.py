# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash draft model.

Predicts a block of future tokens in parallel using hidden states from the target
model. The draft stack is built by subclassing the target architecture's own
HuggingFace attention and decoder layer, resolved from the checkpoint's
``model_type``, so no architecture is hardcoded here.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch

from executorch.backends.mlx.examples.llm.dflash.arch import resolve_arch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Modules the DFlash decoder-layer forward below actually consumes. An architecture
# whose decoder layer carries additional weights (Gemma3's pre/post feedforward
# norms, for example) would have them silently dropped, so it is rejected instead.
_SUPPORTED_LAYER_CHILDREN = frozenset(
    {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
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
    model_type: str
    block_size: int = 16
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    layer_types: Tuple[str, ...] = field(default_factory=tuple)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    embed_scale: float = 1.0  # Qwen3/Llama use 1.0


def to_hf_config(config: DFlashConfig):
    """Build the architecture's HuggingFace config from the DFlash checkpoint configuration.
    This lets the draft reuse that architecture's stock modules.
    """
    rope_parameters = dict(config.rope_scaling or {})
    rope_parameters["rope_type"] = rope_parameters.pop(
        "type", rope_parameters.get("rope_type", "default")
    )
    rope_parameters["rope_theta"] = config.rope_theta

    hf_config_cls = resolve_arch(config.model_type).config_cls
    try:
        hf_config = hf_config_cls(
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
    except Exception as e:
        raise ValueError(
            f"model_type {config.model_type!r} is not supported by DFlash: "
            f"{hf_config_cls.__name__} rejected the draft checkpoint's parameters "
            f"({type(e).__name__}: {e})."
        ) from e
    hf_config._attn_implementation = "mlx"  # Match the target's attention dispatch
    return hf_config


def _norm_or_identity(module, name: str, x: torch.Tensor) -> torch.Tensor:
    """Apply an optional per-head QK norm. Qwen3 and Gemma3 have these; Llama does not."""
    norm = getattr(module, name, None)
    return x if norm is None else norm(x)


@lru_cache(maxsize=None)
def dflash_attention_cls(model_type: str):
    """Build the DFlash attention class for an architecture.

    Subclasses the architecture's own attention so projections, optional QK norms
    and the attention dispatch are inherited, keeping the state-dict layout
    identical to the stock module.
    """
    arch = resolve_arch(model_type)
    apply_rotary_pos_emb = arch.apply_rotary_pos_emb
    eager_attention_forward = arch.eager_attention_forward

    class DFlashAttention(arch.attention_cls):
        """DFlash attention where block queries attend to both target context and the block."""

        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.is_causal = False  # The proposal block uses bidirectional attention

        @classmethod
        def adapt(cls, attention: nn.Module) -> None:
            """Convert an already-built stock attention module in place.

            Safe because this subclass adds no parameters or buffers of its own,
            so the module's state is already complete.
            """
            attention.__class__ = cls
            attention.is_causal = False

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
            query_states = _norm_or_identity(
                self, "q_norm", self.q_proj(x).view(q_shape)
            ).transpose(1, 2)

            if cache is None:
                # The uncached path reprojects the full target context each round.
                # It serves as the reference behavior for the cached implementation.
                S = x_ctx.shape[1]
                kv_shape = (B, S + L, -1, self.head_dim)
                kv_input = torch.cat([x_ctx, x], dim=1)
                key_states = _norm_or_identity(
                    self, "k_norm", self.k_proj(kv_input).view(kv_shape)
                ).transpose(1, 2)
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
                new_k = _norm_or_identity(
                    self, "k_norm", self.k_proj(x_ctx).view(new_shape)
                ).transpose(1, 2)
                new_v = self.v_proj(x_ctx).view(new_shape).transpose(1, 2)

                ctx_cos = cos.narrow(1, 0, new_ctx_len)
                ctx_sin = sin.narrow(1, 0, new_ctx_len)
                _, new_k = apply_rotary_pos_emb(new_k, new_k, ctx_cos, ctx_sin)
                blk_cos = cos.narrow(1, new_ctx_len, L)
                blk_sin = sin.narrow(1, new_ctx_len, L)
                query_states, _ = apply_rotary_pos_emb(
                    query_states, query_states, blk_cos, blk_sin
                )

                full_k, full_v = cache.write(
                    self.layer_idx, new_k, new_v, cache_position
                )

                blk_shape = (B, L, -1, self.head_dim)
                blk_k = _norm_or_identity(
                    self, "k_norm", self.k_proj(x).view(blk_shape)
                ).transpose(1, 2)
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

    DFlashAttention.__name__ = f"DFlash{arch.attention_cls.__name__}"
    DFlashAttention.__qualname__ = DFlashAttention.__name__
    return DFlashAttention


def _reject_unsupported_layer(layer: nn.Module, model_type: str) -> None:
    extra = [
        name
        for name, child in layer.named_children()
        if name not in _SUPPORTED_LAYER_CHILDREN
        and any(p.numel() for p in child.parameters())
    ]
    if extra:
        raise ValueError(
            f"model_type {model_type!r} is not supported by DFlash: its decoder layer "
            f"has weight-carrying submodules the DFlash forward does not apply "
            f"({', '.join(sorted(extra))}). Supporting it requires extending "
            f"the DFlash decoder-layer forward."
        )


@lru_cache(maxsize=None)
def dflash_decoder_layer_cls(model_type: str):
    """Build the DFlash decoder-layer class for an architecture.

    Subclassing the architecture's own layer inherits its MLP and norms, so only
    attention and the forward's block/context signature differ.
    """
    arch = resolve_arch(model_type)
    attention_cls = dflash_attention_cls(model_type)

    class DFlashDecoderLayer(arch.decoder_layer_cls):
        """Decoder layer adapted for DFlash's block/context attention pattern."""

        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            _reject_unsupported_layer(self, model_type)
            # Adapt the attention the parent already built rather than replacing it,
            # so no module is constructed and thrown away.
            attention_cls.adapt(self.self_attn)

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

    DFlashDecoderLayer.__name__ = f"DFlash{arch.decoder_layer_cls.__name__}"
    DFlashDecoderLayer.__qualname__ = DFlashDecoderLayer.__name__
    return DFlashDecoderLayer


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.hf_config = to_hf_config(config)
        arch = resolve_arch(config.model_type)
        layer_cls = dflash_decoder_layer_cls(config.model_type)

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [layer_cls(self.hf_config, i) for i in range(config.num_hidden_layers)]
        )
        # Take the norm class from a built layer so the architecture's own norm is
        # reused without a second name-based lookup.
        norm_cls = type(self.layers[0].input_layernorm)
        self.hidden_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = arch.rotary_embedding_cls(self.hf_config)
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
    """Load architecture parameters and DFlash-specific settings from config.json.
    This includes the tapped target layers, mask token, and other draft configuration.
    """
    import json
    from pathlib import Path

    cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    dcfg = cfg["dflash_config"]
    if "model_type" not in cfg:
        raise ValueError(
            f"{checkpoint_dir}/config.json has no 'model_type', so the draft "
            f"architecture cannot be resolved."
        )
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
        model_type=cfg["model_type"],
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
