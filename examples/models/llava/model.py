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
from examples.portable.utils import export_to_edge


class EncoderModel(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.model_ = llava_model

    def forward(self, images_tensor):
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
        model = EncoderModel(self.model_)
        return model


    def get_example_inputs(self):
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        images = load_images([image_file])
        images_tensor = process_images(
            images,
            self.image_processor_,
            self.model_.config
        ).to(self.model_.device, dtype=torch.float32)
        return (images_tensor,)


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
print(torch.__version__)

features = model(*inputs)

m = capture_pre_autograd_graph(model, inputs)
edge_manager = export_to_edge(m, inputs)


# image_file = "https://llava-vl.github.io/static/images/view.jpg"
# prompt = "What are the things I should be cautious about when I visit here?"
# args = type('Args', (), {
#     "model_path": model_path,
#     "model_base": None,
#     "model_name": get_model_name_from_path(model_path),
#     "query": prompt,
#     "conv_mode": None,
#     "image_file": image_file,
#     "sep": ",",
#     "temperature": 0,
#     "top_p": None,
#     "num_beams": 1,
#     "max_new_tokens": 512
# })()
#
# # vision_tower = build_vision_tower(args, )
#
# eval_model(args)
