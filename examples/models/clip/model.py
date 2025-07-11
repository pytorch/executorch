# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import CLIPModel as HFCLIPModel, CLIPProcessor

from ..model_base import EagerModelBase


class OpenCLIPWrapper(torch.nn.Module):
    """Wrapper for OpenCLIP model to make it torch.export compatible"""

    def __init__(self, model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):
        super().__init__()
        self.model = HFCLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def forward(self, pixel_values, input_ids, attention_mask):
        # Extract image and text features
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=False,
            )

        # Return image and text embeddings
        return outputs.image_embeds, outputs.text_embeds


class CLIPModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading OpenCLIP model from HuggingFace")
        model = OpenCLIPWrapper("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        model.eval()
        logging.info("Loaded OpenCLIP model")
        return model

    def get_example_inputs(self):
        # Example inputs for CLIP
        # Image: batch_size=1, channels=3, height=224, width=224
        pixel_values = torch.randn(1, 3, 224, 224)

        # Text: batch_size=1, max_length=77 (CLIP's typical context length)
        input_ids = torch.randint(0, 49408, (1, 77))  # CLIP vocab size is ~49408
        attention_mask = torch.ones(1, 77)

        return (pixel_values, input_ids, attention_mask)
