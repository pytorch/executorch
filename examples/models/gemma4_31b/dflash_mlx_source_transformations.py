# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash-aware MLX source transformations for Gemma4-31B.

mlx_source_transformations.py's mlx_source_transformations() ends by calling
_replace_model_forward(model), which unconditionally overwrites the top-level
forward with a (tokens, input_pos) -> (B, 1, V) last-token-only, single-output
variant -- it does a full `types.MethodType` replacement, not a wrapper, so
whatever forward the model had before (including Gemma4_31BWithHidden's
full-sequence, two-output forward) is discarded entirely and never called at
export time. That silently turned our DFlash target export back into a
last-token-only, hidden-state-free model with no error at any stage.

This module reuses everything else from mlx_source_transformations.py
unchanged (KV cache attachment, per-layer attention/layer forward rewrites
via mlx.rope / mlx.custom_sdpa -- both output-shape-agnostic, operate within
a layer) and only replaces the final _replace_model_forward(model) call with
a DFlash-aware top-level forward that preserves full-sequence logits and
hidden-state capture at model.dflash_layer_ids.
"""

import types

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
    MLXKVCache,
    MLXRingKVCache,
    MLXTurboQuantKVCache,
    _replace_attention_forward,
    _replace_layer_forward,
)


def _replace_dflash_model_forward(model: nn.Module) -> None:
    """Replace the top-level forward with a DFlash-aware, MLX-optimized variant.

    Signature: (tokens, input_pos) -> (logits, hidden) where logits is
    (B, T, V) over every position (not last-token-only) and hidden is
    (B, T, len(dflash_layer_ids) * hidden_size) -- matching
    Gemma4_31BWithHidden.forward's contract, but using the per-layer
    MLX-optimized attention/layer forwards installed by
    dflash_mlx_source_transformations instead of the original PyTorch ops.
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
                f"dflash_layer_ids {sorted(missing)} not reached -- "
                f"model only has {len(self.layers)} layers"
            )
        hidden = torch.cat([captured[i] for i in self.dflash_layer_ids], dim=-1)

        x = self.norm(x)
        logits = self.lm_head(x).float()  # (B, T, V) -- NOT x[:, -1, :]
        cap = self.logit_softcap.float()
        logits = torch.tanh(logits / cap) * cap
        return logits, hidden

    model.forward = types.MethodType(_mlx_dflash_model_forward, model)


def dflash_mlx_source_transformations(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    use_turboquant: bool = False,
    max_write_len: int | None = None,
) -> None:
    """Apply MLX source transformations to a Gemma4_31BWithHidden model in-place.

    Identical to mlx_source_transformations.mlx_source_transformations()
    except the final step installs a DFlash-aware top-level forward
    (full-sequence logits + hidden-state capture) instead of the stock
    last-token-only, single-output one. See module docstring for why this
    duplication is necessary rather than composing with the original.

    model must be a Gemma4_31BWithHidden instance (needs .dflash_layer_ids).
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
