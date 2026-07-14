# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extracting Qwen3 hidden-state for DFlash.

Same idea as examples/models/gemma4_31b/mlx_source_transformations.py --
extract layers and return them concatenated alongside logits. In Qwen3, the 
layer ids from z-lab Qwen3 DFlash draft config is [1, 9, 17, 25, 33]

Gemma 4 does this by patching its own hand-written forward(). Qwen3 goes
through the generic HF export path instead (export_llm_hf.py), which wraps
the model in transformers' TorchExportableModuleWithStaticCache before
torch.export. So we subclass that wrapper and add output_hidden_states
to its forward rather than patching Qwen3 itself.

Base class signature/behavior confirmed via:
    inspect.getsource(transformers.integrations.executorch.TorchExportableModuleWithStaticCache)
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
