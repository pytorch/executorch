# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from examples.models.model_base import EagerModelBase
from llava.eval.run_llava import eval_model, load_images, process_images
from llava.mm_utils import get_model_name_from_path

from llava.model.builder import load_pretrained_model
from torch import nn


class EncoderModel(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.model_ = llava_model

    def forward(self, images_tensor):
        features = self.model_.get_model().get_vision_tower()(images_tensor)
        features = self.model_.get_model().mm_projector(features)
        return features


class LlavaModel(EagerModelBase):
    def __init__(self):
        model_path = "liuhaotian/llava-v1.5-7b"
        tokenizer, self.model_, self.image_processor_, context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
            )
        )
        self.device = "cpu"
        self.dtype = torch.float32
        self.model_.to(device=self.device, dtype=self.dtype)

    def get_eager_model(self):
        model = EncoderModel(self.model_)
        return model

    def get_example_inputs(self):
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        images = load_images([image_file])
        images_tensor = process_images(
            images, self.image_processor_, self.model_.config
        ).to(self.model_.device)
        return (images_tensor,)
