# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import WhisperModel as wm

from ..model_base import EagerModelBase


class WhisperWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).last_hidden_state


class WhisperModel(EagerModelBase):
    def __init__(self, model_name: str = "openai/whisper-base"):
        self.model = wm.from_pretrained(model_name)

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading Whisper model")
        m = WhisperWrapper(self.model).eval()
        logging.info("Loaded Whisper model")
        return m

    def get_example_inputs(self):
        return (
            torch.randn(1, 80, 3000),
            torch.tensor(
                [[self.model.config.decoder_start_token_id]], dtype=torch.long
            ),
        )
