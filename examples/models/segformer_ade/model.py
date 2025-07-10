# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import SegformerForSemanticSegmentation

from ..model_base import EagerModelBase


class SegformerWrapper(torch.nn.Module):
    """Wrapper for HuggingFace SegFormer model to make it torch.export compatible"""

    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.segformer.eval()

    def forward(self, pixel_values):
        # pixel_values: [batch, 3, height, width] - RGB image
        with torch.no_grad():
            outputs = self.segformer(pixel_values)
        return outputs.logits


class SegformerADEModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading SegFormer ADE model from HuggingFace")
        model = SegformerWrapper("nvidia/segformer-b0-finetuned-ade-512-512")
        model.eval()
        logging.info("Loaded SegFormer ADE model")
        return model

    def get_example_inputs(self):
        # SegFormer standard input size: 512x512 RGB image
        tensor_size = (1, 3, 512, 512)
        return (torch.randn(tensor_size),)
