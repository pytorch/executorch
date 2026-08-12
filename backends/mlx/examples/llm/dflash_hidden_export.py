# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic hidden-state-tapping export wrapper for DFlash. 

Wraps TorchExportableModuleWithStaticCache and enables hidden-state outputs during the forward pass. This allows standard HuggingFace causal language models using the generic export path to reuse the wrapper. The wrapper is kept separate from model-specific implementations so it can be shared across different models. The base class behavior follows transformers' TorchExportableModuleWithStaticCache implementation. 
"""

from typing import List, Optional, Sequence

import torch
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache


class TorchExportableModuleWithStaticCacheAndHidden(
    TorchExportableModuleWithStaticCache
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

        captured = [outs.hidden_states[i + 1] for i in self.layer_ids]
        hidden = torch.cat(captured, dim=-1)

        if hasattr(outs, "logits"):
            return outs.logits, hidden
        return outs.last_hidden_state, hidden
