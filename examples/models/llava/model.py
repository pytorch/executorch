# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch
from torch import nn

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, load_images, process_images

from torch._export import capture_pre_autograd_graph
from examples.models.model_base import EagerModelBase


class EncoderModel(nn.Module):
    def __init__(self, llava_model, image_processor):
        super().__init__()
        self.model_ = llava_model
        self.image_processor_ = image_processor

    def forward(self, images):
        # image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor_,
            self.model_.config
        ).to(self.model_.device, dtype=torch.float32)

        features = self.model_.get_model().get_vision_tower()(images_tensor)
        features = self.model_.get_model().mm_projector(features)
        return features
class LlavaModel(EagerModelBase):
    def __init__(self, model_path):
        tokenizer, self.model_, self.image_processor_, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )
        self.device = 'cpu'
        self.model_.to(self.device)
        self.dtype = torch.float32

    def get_eager_model(self):

        # if self.dtype:
        #     # convert to the type of the provided checkpoint
        #     # input and output are torch.long, so signature unchanged
        #     self.model_.to(self.dtype)
        # else:
        #     # int8 quantization code has some bf16,
        #     # switch all to FP32
        #     self.model_.to(torch.float32)
        model = EncoderModel(self.model_, self.image_processor_)
        return model


    def get_example_inputs(self):
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        return load_images([image_file])

    def get_example_inputs_kvcache(self):
        cache_sizes = self.model_.get_cache_sizes()
        cache_k = torch.zeros(cache_sizes, dtype=self.dtype)
        cache_v = torch.zeros(cache_sizes, dtype=self.dtype)
        return (
            torch.tensor(
                [[1]], dtype=torch.long
            ),  # tokens, with kv cache our input token length is always just 1 token.
            torch.tensor(
                0, dtype=torch.long
            ),  # start_pos, what token of output are we on.
            cache_k,  # key caches
            cache_v,  # value caches
        )

model_path = "liuhaotian/llava-v1.5-7b"
llava_model = LlavaModel(model_path)
model = llava_model.get_eager_model()
inputs = llava_model.get_example_inputs()


features = model(inputs)

m = capture_pre_autograd_graph(model, inputs)


