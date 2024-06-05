# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from examples.models.model_base import EagerModelBase
from llava.eval.run_llava import load_images, process_images
from llava.mm_utils import get_model_name_from_path

from llava.model.builder import load_pretrained_model
from torch import nn

import torchvision
from torchvision.transforms import v2

from dataclasses import dataclass
from torch.export import Dim



@dataclass
class PreprocessConfig:
    crop_size: dict
    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float


class MyPreprocess(nn.Module):
    def __init__(self, config: PreprocessConfig):
        super().__init__()
        self.config = config

    def forward(self, img):
        w = max(img.shape[1], img.shape[2])
        padded = v2.functional.center_crop(img, output_size=[w, w])
        resized = torchvision.transforms._functional_tensor.resize(padded, size=[self.config.crop_size['height'], self.config.crop_size['width']])
        torch._check(resized.size(1) == self.config.crop_size['height'])
        torch._check(resized.size(2) == self.config.crop_size['width'])
        scaled = resized * self.config.rescale_factor
        normed = v2.functional.normalize(scaled, self.config.image_mean, self.config.image_std)
        return normed

class EncoderModel(nn.Module):
    def __init__(self, image_processor, llava_model):
        super().__init__()
        pre_config = PreprocessConfig(image_processor.crop_size, image_processor.image_mean, image_processor.image_std,
                                      image_processor.rescale_factor)
        self.pre_processer_ = MyPreprocess(pre_config)
        self.model_ = llava_model

    def forward(self, images_tensor):
        processed = self.pre_processer_(images_tensor)
        processed = torch.unsqueeze(processed, dim=0)
        features = self.model_.get_model().get_vision_tower()(processed)
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
        model = EncoderModel(self.image_processor_, self.model_)
        return model

    def get_example_inputs(self):
        tensor_size = (3, 1024, 2048)
        return (torch.randn(tensor_size),)

    def get_dynamic_shapes(self):
        length = Dim('length', min=8, max=4090)
        width = Dim('width', min=10, max=4092)  # 1346 <= 2*_width <= 2048
        # width = 2*_width
        dynamic_shapes = [{1: length, 2: width}]
        return dynamic_shapes

