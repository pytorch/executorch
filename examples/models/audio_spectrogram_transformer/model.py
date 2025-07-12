# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from ..model_base import EagerModelBase


class AudioSpectrogramTransformerWrapper(torch.nn.Module):
    """Wrapper for HuggingFace Audio Spectrogram Transformer model to make it torch.export compatible"""

    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(model_name)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_values):
        # Audio classification with AST
        with torch.no_grad():
            outputs = self.model(input_values)

        # Return classification logits
        return outputs.logits


class AudioSpectrogramTransformerModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading Audio Spectrogram Transformer model from HuggingFace")
        model = AudioSpectrogramTransformerWrapper(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        model.eval()
        logging.info("Loaded Audio Spectrogram Transformer model")
        return model

    def get_example_inputs(self):
        # Example inputs for AST
        # Audio spectrogram: batch_size=1, time_steps=1024, freq_bins=128
        input_values = torch.randn(1, 1024, 128)

        return (input_values,)
