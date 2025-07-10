# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import Wav2Vec2Model

from ..model_base import EagerModelBase


class Wav2Vec2Wrapper(torch.nn.Module):
    """Wrapper for HuggingFace Wav2Vec2 model to make it torch.export compatible"""

    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec2.eval()

    def forward(self, input_values):
        # input_values: [batch, sequence_length] - raw audio waveform
        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
        return outputs.last_hidden_state


class Wav2Vec2Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading Wav2Vec2 model from HuggingFace")
        model = Wav2Vec2Wrapper("facebook/wav2vec2-base-960h")
        model.eval()
        logging.info("Loaded Wav2Vec2 model")
        return model

    def get_example_inputs(self):
        # Raw audio input: 1 second of 16kHz audio
        input_values = torch.randn(1, 16000)
        return (input_values,)
