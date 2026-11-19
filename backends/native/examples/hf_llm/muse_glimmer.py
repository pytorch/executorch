# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The parts of Muse Glimmer's Hugging Face model the native export runs."""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class MuseGlimmerTextLM(nn.Module):
    """Muse Glimmer's language model and LM head, under their Hugging Face
    parameter names, without the vision tower. Text-only, so it skips the
    multimodal model's scan of input_ids for image and video tokens."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.config = model.config
        self.generation_config = model.generation_config
        self.model = nn.ModuleDict({"language_model": model.model.language_model})
        self.lm_head = model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        **_,
    ) -> CausalLMOutput:
        text_config = self.config.text_config
        hidden = self.model["language_model"](
            input_ids=input_ids, position_ids=position_ids, use_cache=False
        ).last_hidden_state
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = self.lm_head(hidden) * text_config.output_multiplier
        cap = text_config.final_logit_softcapping
        return CausalLMOutput(logits=torch.tanh(logits / cap) * cap)
