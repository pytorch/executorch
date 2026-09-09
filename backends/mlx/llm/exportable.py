#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF exportable wrappers with optional hidden-state tapping.

Centralizes creation of TorchExportableModuleWithStaticCache / HybridCache
and their hidden-tapping variants. This avoids duplicating the
sliding_window vs tap_layers branching in every example script.

Both static and hybrid wrappers are supported:
- static: models with no sliding_window (e.g. Llama)
- hybrid: models with sliding_window (e.g. Gemma, Qwen)

The MLX cache installation (HFStaticCache / RingBuffer) is also factored
into `install_mlx_cache` which works regardless of wrapper type because it
checks for `static_cache` vs `cache` attribute.
"""

import logging
from typing import List, Optional, Sequence

import torch
from transformers.integrations.executorch import (
    TorchExportableModuleWithHybridCache,
    TorchExportableModuleWithStaticCache,
)

logger = logging.getLogger(__name__)


class _HiddenTapMixin:
    """Shared tapping logic - expects self.layer_ids and self.model to exist."""

    def _tap_hidden(self, outs):
        # hidden_states[0] is embedding output, so layer i output is at i+1
        captured = [outs.hidden_states[i + 1] for i in self.layer_ids]
        return torch.cat(captured, dim=-1)


class TorchExportableModuleWithStaticCacheAndHidden(
    _HiddenTapMixin, TorchExportableModuleWithStaticCache
):
    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        layer_ids: Sequence[int] = (),
    ):
        super().__init__(
            model, batch_size=batch_size, max_cache_len=max_cache_len, device=device
        )
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.static_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        hidden = self._tap_hidden(outs)
        if hasattr(outs, "logits"):
            return outs.logits, hidden
        return outs.last_hidden_state, hidden


class TorchExportableModuleWithHybridCacheAndHidden(
    _HiddenTapMixin, TorchExportableModuleWithHybridCache
):
    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        layer_ids: Sequence[int] = (),
    ):
        super().__init__(
            model, batch_size=batch_size, max_cache_len=max_cache_len, device=device
        )
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
        )
        hidden = self._tap_hidden(outs)
        if hasattr(outs, "logits"):
            return outs.logits, hidden
        return outs.last_hidden_state, hidden


def create_hf_exportable(
    model,
    max_cache_len: int,
    tap_layers: Optional[Sequence[int]] = None,
    batch_size: int = 1,
):
    """Factory: picks static vs hybrid and hidden-tapping vs plain.

    Args:
        model: HF CausalLM
        max_cache_len: cache capacity
        tap_layers: optional layer indices to tap and concat as second output
        batch_size: batch size for cache init

    Returns:
        An exportable module with .model attribute pointing to HF model
        and .static_cache or .cache attribute depending on type.
    """
    text_config = model.config.get_text_config()
    sliding_window = getattr(text_config, "sliding_window", None)

    if sliding_window is not None:
        if tap_layers is not None:
            logger.info(
                f"Creating TorchExportableModuleWithHybridCacheAndHidden with taps {list(tap_layers)}..."
            )
            return TorchExportableModuleWithHybridCacheAndHidden(
                model=model,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                layer_ids=tap_layers,
            )
        logger.info("Creating TorchExportableModuleWithHybridCache wrapper...")
        return TorchExportableModuleWithHybridCache(
            model=model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
        )
    else:
        if tap_layers is not None:
            logger.info(
                f"Creating TorchExportableModuleWithStaticCacheAndHidden with taps {list(tap_layers)}..."
            )
            return TorchExportableModuleWithStaticCacheAndHidden(
                model=model,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                layer_ids=tap_layers,
            )
        logger.info("Creating TorchExportableModuleWithStaticCache wrapper...")
        return TorchExportableModuleWithStaticCache(
            model=model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
        )


def install_mlx_cache(
    exportable,
    config,
    max_batch_size: int = 1,
    max_cache_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    prefill_chunk_size: Optional[int] = None,
):
    """Install MLX KV cache (linear or ring-buffer) regardless of wrapper type.

    For sliding-window models, prefill_chunk_size is used as max_write_len to size
    the ring buffer as window + chunk -1, avoiding over-allocation.
    """
    text_config = config.get_text_config()
    sliding_window = getattr(text_config, "sliding_window", None)

    if sliding_window is not None:
        from executorch.backends.mlx.llm.source_transformation import (
            replace_hf_cache_with_mlx_ring_buffer,
        )

        logger.info(
            f"Replacing HuggingFace HybridCache with MLX ring buffers "
            f"(window {sliding_window}, cache length {max_cache_len}, "
            f"prefill_chunk_size {prefill_chunk_size})..."
        )
        replace_hf_cache_with_mlx_ring_buffer(
            exportable,
            config,
            max_batch_size=max_batch_size,
            window_size=sliding_window,
            max_cache_len=max_cache_len,
            dtype=dtype,
            max_write_len=prefill_chunk_size,
        )
    else:
        from executorch.backends.mlx.llm.source_transformation import (
            replace_hf_cache_with_mlx,
        )

        logger.info(
            f"Replacing HuggingFace StaticCache with HFStaticCache "
            f"(cache length {max_cache_len})..."
        )
        replace_hf_cache_with_mlx(
            exportable,
            config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype,
        )
