# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma4-31B DFlash hidden-state export wrapper.

Gemma4-31B's forward() is hand-written (not the generic export_llm_hf.py
path), so backends/mlx/examples/llm/dflash_hidden_export.py's
TorchExportableModuleWithStaticCacheAndHidden doesn't apply here. This
patches Gemma4_31B.forward directly to also return hidden states from
the configured target layers.

target_layer_ids are 0-indexed into self.layers (captured[i] = output of
self.layers[i]), matching the Qwen3 wrapper's convention.

UNVERIFIED: Gemma4DecoderLayer applies layer_scalar before returning, so
the captured hidden already includes it. Not confirmed against HF's
modeling_gemma4.py (unavailable in current transformers) or z-lab's
training code -- if wrong, draft conditioning will be subtly off and
tau will look worse with no explicit error. Re-check before trusting
benchmarks on this path.
"""

from typing import List, Sequence, Tuple

import torch

from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4_31BConfig
from executorch.examples.models.gemma4_31b.sampler import sample


class Gemma4_31BWithHidden(Gemma4_31B):
    """Gemma4_31B variant that also returns concatenated target hidden states."""

    def __init__(self, config: Gemma4_31BConfig, layer_ids: Sequence[int] = ()):
        super().__init__(config)
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        self.dflash_layer_ids: List[int] = list(layer_ids)

    def forward(
        self,
        tokens: torch.LongTensor,
        input_pos: torch.LongTensor,
        temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                f"dflash_layer_ids {sorted(missing)} not reached -- "
                f"model only has {len(self.layers)} layers"
            )
        hidden = torch.cat([captured[i] for i in self.dflash_layer_ids], dim=-1)

        x = self.norm(x)
        last = self.lm_head(x[:, -1, :]).float()
        cap = self.logit_softcap.float()
        last = torch.tanh(last / cap) * cap
        return sample(last, temperature), hidden
