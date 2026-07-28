"""Gemma4-31B DFlash hidden-state export wrapper. 

Gemma4-31B uses its own forward() implementation instead of the generic export_llm_hf.py path, so this patches Gemma4_31B.forward to also return hidden states from the configured target layers. 
"""

import types
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
    MLXKVCache,
    MLXRingKVCache,
    MLXTurboQuantKVCache,
    _replace_attention_forward,
    _replace_layer_forward,
)
from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig


class Gemma4_31BWithHidden(Gemma4_31B):

    def __init__(self, config: Gemma4_31BConfig, layer_ids: Sequence[int] = ()):
        super().__init__(config)
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.dflash_layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        tokens: torch.LongTensor,
        input_pos: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns full-sequence logits and hidden states for DFlash block verification. 
        Unlike the base implementation, which only returns the last token logits for single-token decoding, DFlash returns logits for every position in the drafted block so run_dflash.py can perform greedy verification and first-mismatch acceptance. 
        """
        x = self.embed_tokens(tokens) * self.embed_normalizer
        sliding_mask, full_mask = self._build_masks(input_pos)

        layer_id_set = set(self.dflash_layer_ids)
        captured = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, sliding_mask, full_mask)
            if i in layer_id_set:
                captured[i] = x

        missing = layer_id_set - captured.keys()
        if missing:
            raise ValueError(
                f"dflash_layer_ids {sorted(missing)} not reached: "
                f"model only has {len(self.layers)} layers"
            )
        hidden = torch.cat([captured[i] for i in self.dflash_layer_ids], dim=-1)

        x = self.norm(x)
        logits = self.lm_head(x).float()
        cap = self.logit_softcap.float()
        logits = torch.tanh(logits / cap) * cap
        return logits, hidden


def _replace_dflash_model_forward(model: nn.Module) -> None:
    """Installs a DFlash-aware top-level forward after MLX op rewrites are applied.
    The stock mlx_source_transformations' _replace_model_forward overwrites the model's forward entirely, discarding Gemma4_31BWithHidden.forward, so this reinstalls the same hidden-capturing logic on top of the MLX-optimized layers.
    """

    def _mlx_dflash_model_forward(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ):
        x = self.embed_tokens(tokens) * self.embed_normalizer

        layer_id_set = set(self.dflash_layer_ids)
        captured = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos)
            if i in layer_id_set:
                captured[i] = x

        missing = layer_id_set - captured.keys()
        if missing:
            raise ValueError(
                f"dflash_layer_ids {sorted(missing)} not reached: "
                f"model only has {len(self.layers)} layers"
            )
        hidden = torch.cat([captured[i] for i in self.dflash_layer_ids], dim=-1)

        x = self.norm(x)
        logits = self.lm_head(x).float()
        cap = self.logit_softcap.float()
        logits = torch.tanh(logits / cap) * cap
        return logits, hidden

    model.forward = types.MethodType(_mlx_dflash_model_forward, model)


def dflash_mlx_source_transformations(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    use_turboquant: bool = False,
    max_write_len: Optional[int] = None,
) -> None:
    """DFlash-aware variant of mlx_source_transformations for a Gemma4_31BWithHidden model.
    Reuses the same per-layer attention/KV-cache rewrites as the stock transform, but installs the hidden-capturing forward above instead of the stock last-token-only one.
    """
    if not hasattr(model, "dflash_layer_ids"):
        raise TypeError(
            "dflash_mlx_source_transformations requires a model with "
            "dflash_layer_ids (e.g. Gemma4_31BWithHidden), got "
            f"{type(model).__name__}"
        )

    config = model.config

    for layer in model.layers:
        attn = layer.self_attn

        if attn.is_sliding:
            sliding_write_len = (
                min(max_write_len, config.sliding_window)
                if max_write_len is not None
                else None
            )
            attn.kv_cache = MLXRingKVCache(
                max_batch_size=1,
                max_context_length=config.sliding_window,
                n_heads=attn.n_kv_heads,
                head_dim=attn.head_dim,
                dtype=dtype,
                max_write_len=sliding_write_len,
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

    _replace_dflash_model_forward(model)