# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import DetrForObjectDetection

from ..model_base import EagerModelBase


class DetrWrapper(torch.nn.Module):
    """Wrapper for HuggingFace DETR model to make it torch.export compatible"""

    def __init__(self, model_name="facebook/detr-resnet-50"):
        super().__init__()
        self.detr = DetrForObjectDetection.from_pretrained(model_name)
        self.detr.eval()

    def forward(self, pixel_values):
        # pixel_values: [batch, 3, height, width] - RGB image
        with torch.no_grad():
            outputs = self.detr(pixel_values)
        # Return logits and boxes for object detection
        return outputs.logits, outputs.pred_boxes


class DetrResNet50Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading DETR ResNet-50 model from HuggingFace")
        model = DetrWrapper("facebook/detr-resnet-50")
        model.eval()
        logging.info("Loaded DETR ResNet-50 model")
        return model

    def get_example_inputs(self):
        # DETR standard input size: 800x800 RGB image (can handle various sizes)
        tensor_size = (1, 3, 800, 800)
        return (torch.randn(tensor_size),)
