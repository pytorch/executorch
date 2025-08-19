# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import Swin2SRForImageSuperResolution

from ..model_base import EagerModelBase


class Swin2SRWrapper(torch.nn.Module):
    """Wrapper for HuggingFace Swin2SR model to make it torch.export compatible"""

    def __init__(self, model_name="caidas/swin2SR-classical-sr-x2-64"):
        super().__init__()
        self.swin2sr = Swin2SRForImageSuperResolution.from_pretrained(model_name)
        self.swin2sr.eval()

    def forward(self, pixel_values):
        # pixel_values: [batch, 3, height, width] - RGB image
        with torch.no_grad():
            outputs = self.swin2sr(pixel_values)
        return outputs.reconstruction


class Swin2SR2xModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading Swin2SR 2x model from HuggingFace")
        model = Swin2SRWrapper("caidas/swin2SR-classical-sr-x2-64")
        model.eval()
        logging.info("Loaded Swin2SR 2x model")
        return model

    def get_example_inputs(self):
        # Swin2SR input size: 64x64 RGB image for 2x super-resolution
        tensor_size = (1, 3, 64, 64)
        return (torch.randn(tensor_size),)
