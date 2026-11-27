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
from enum import IntEnum
from typing import List, Optional, Sequence, Union

import torch
from transformers.integrations.executorch import (
    TorchExportableModuleWithHybridCache,
    TorchExportableModuleWithStaticCache,
)

logger = logging.getLogger(__name__)


class LogitsToKeepMode(IntEnum):
    FULL = 0
    LAST = 1
    SELECTED = 2

    @classmethod
    def from_value(cls, value: Union["LogitsToKeepMode", str, int]):
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError as error:
                raise ValueError(f"Unsupported logits-to-keep mode: {value}") from error
        return cls(value)


class _LogitsToKeepMixin:
    logits_to_keep_mode: LogitsToKeepMode

    def _resolve_logits_to_keep(
        self, logits_to_keep: Optional[torch.LongTensor]
    ) -> Union[int, torch.LongTensor]:
        if self.logits_to_keep_mode == LogitsToKeepMode.SELECTED:
            if logits_to_keep is None:
                raise ValueError("selected logits-to-keep requires an index tensor")
            if logits_to_keep.dtype != torch.int64 or logits_to_keep.dim() != 1:
                raise ValueError("logits_to_keep must be an int64[K] tensor")
            return logits_to_keep
        return int(self.logits_to_keep_mode)

    def _logits_to_keep_kwargs(
        self, logits_to_keep: Optional[torch.LongTensor]
    ) -> dict:
        if self.logits_to_keep_mode == LogitsToKeepMode.FULL:
            return {}
        return {"logits_to_keep": self._resolve_logits_to_keep(logits_to_keep)}

    def _sync_cache_position(self, cache, cache_position) -> None:
        if cache_position is None or not hasattr(cache, "layers"):
            return
        for layer in cache.layers:
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length.copy_(cache_position[0])


class _HiddenTapMixin:
    """Shared tapping logic - expects self.layer_ids and self.model to exist."""

    def _tap_hidden(self, outs):
        # hidden_states[0] is embedding output, so layer i output is at i+1
        captured = [outs.hidden_states[i + 1] for i in self.layer_ids]
        return torch.cat(captured, dim=-1)


class TorchExportableModuleWithStaticCacheAndLogitsToKeep(
    _LogitsToKeepMixin, TorchExportableModuleWithStaticCache
):
    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        logits_to_keep_mode: LogitsToKeepMode = LogitsToKeepMode.FULL,
    ):
        super().__init__(
            model, batch_size=batch_size, max_cache_len=max_cache_len, device=device
        )
        self.logits_to_keep_mode = LogitsToKeepMode.from_value(logits_to_keep_mode)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[torch.LongTensor] = None,
    ):
        self._sync_cache_position(self.static_cache, cache_position)
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.static_cache,
            use_cache=True,
            **self._logits_to_keep_kwargs(logits_to_keep),
        ).logits


class TorchExportableModuleWithHybridCacheAndLogitsToKeep(
    _LogitsToKeepMixin, TorchExportableModuleWithHybridCache
):
    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        logits_to_keep_mode: LogitsToKeepMode = LogitsToKeepMode.FULL,
    ):
        super().__init__(
            model, batch_size=batch_size, max_cache_len=max_cache_len, device=device
        )
        self.logits_to_keep_mode = LogitsToKeepMode.from_value(logits_to_keep_mode)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[torch.LongTensor] = None,
    ):
        self._sync_cache_position(self.cache, cache_position)
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.cache,
            use_cache=True,
            **self._logits_to_keep_kwargs(logits_to_keep),
        ).logits


class TorchExportableModuleWithStaticCacheAndHidden(
    _HiddenTapMixin, _LogitsToKeepMixin, TorchExportableModuleWithStaticCache
):
    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        layer_ids: Sequence[int] = (),
        logits_to_keep_mode: LogitsToKeepMode = LogitsToKeepMode.FULL,
    ):
        super().__init__(
            model, batch_size=batch_size, max_cache_len=max_cache_len, device=device
        )
        self.logits_to_keep_mode = LogitsToKeepMode.from_value(logits_to_keep_mode)
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[torch.LongTensor] = None,
    ):
        self._sync_cache_position(self.static_cache, cache_position)
        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.static_cache,
            use_cache=True,
            output_hidden_states=True,
            **self._logits_to_keep_kwargs(logits_to_keep),
        )
        hidden = self._tap_hidden(outs)
        if hasattr(outs, "logits"):
            return outs.logits, hidden
        return outs.last_hidden_state, hidden


class TorchExportableModuleWithHybridCacheAndHidden(
    _HiddenTapMixin, _LogitsToKeepMixin, TorchExportableModuleWithHybridCache
):
    def __init__(
        self,
        model,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        layer_ids: Sequence[int] = (),
        logits_to_keep_mode: LogitsToKeepMode = LogitsToKeepMode.FULL,
    ):
        super().__init__(
            model, batch_size=batch_size, max_cache_len=max_cache_len, device=device
        )
        self.logits_to_keep_mode = LogitsToKeepMode.from_value(logits_to_keep_mode)
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[torch.LongTensor] = None,
    ):
        self._sync_cache_position(self.cache, cache_position)
        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            **self._logits_to_keep_kwargs(logits_to_keep),
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
    logits_to_keep_mode: Union[LogitsToKeepMode, str, int] = LogitsToKeepMode.FULL,
):
    """Factory: picks static vs hybrid and hidden-tapping vs plain.

    Args:
        model: HF CausalLM
        max_cache_len: cache capacity
        tap_layers: optional layer indices to tap and concat as second output
        batch_size: batch size for cache init
        logits_to_keep_mode: full, last, or selected logits selection

    Returns:
        An exportable module with .model attribute pointing to HF model
        and .static_cache or .cache attribute depending on type.
    """
    text_config = model.config.get_text_config()
    sliding_window = getattr(text_config, "sliding_window", None)
    logits_to_keep_mode = LogitsToKeepMode.from_value(logits_to_keep_mode)

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
                logits_to_keep_mode=logits_to_keep_mode,
            )
        if logits_to_keep_mode == LogitsToKeepMode.FULL:
            logger.info("Creating TorchExportableModuleWithHybridCache wrapper...")
            return TorchExportableModuleWithHybridCache(
                model=model,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
            )
        logger.info(
            f"Creating hybrid-cache wrapper with {logits_to_keep_mode.name.lower()} logits..."
        )
        return TorchExportableModuleWithHybridCacheAndLogitsToKeep(
            model=model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            logits_to_keep_mode=logits_to_keep_mode,
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
                logits_to_keep_mode=logits_to_keep_mode,
            )
        if logits_to_keep_mode == LogitsToKeepMode.FULL:
            logger.info("Creating TorchExportableModuleWithStaticCache wrapper...")
            return TorchExportableModuleWithStaticCache(
                model=model,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
            )
        logger.info(
            f"Creating static-cache wrapper with {logits_to_keep_mode.name.lower()} logits..."
        )
        return TorchExportableModuleWithStaticCacheAndLogitsToKeep(
            model=model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            logits_to_keep_mode=logits_to_keep_mode,
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
