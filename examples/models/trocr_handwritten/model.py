# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import VisionEncoderDecoderModel

from ..model_base import EagerModelBase


class TrOCRWrapper(torch.nn.Module):
    """Wrapper for HuggingFace TrOCR model to make it torch.export compatible"""

    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        super().__init__()
        self.trocr = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.trocr.eval()

    def forward(self, pixel_values):
        # pixel_values: [batch, 3, height, width] - RGB image
        with torch.no_grad():
            # Generate text from image
            generated_ids = self.trocr.generate(pixel_values, max_length=50)
        return generated_ids


class TrOCRHandwrittenModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading TrOCR handwritten model from HuggingFace")
        model = TrOCRWrapper("microsoft/trocr-base-handwritten")
        model.eval()
        logging.info("Loaded TrOCR handwritten model")
        return model

    def get_example_inputs(self):
        # TrOCR input: 384x384 RGB text image
        pixel_values = torch.randn(1, 3, 384, 384)
        return (pixel_values,)
