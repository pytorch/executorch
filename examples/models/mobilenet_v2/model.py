# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchvision.models import mobilenet_v2  # @manual
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from ..model_base import EagerModelBase


class MV2Model(EagerModelBase):
    def __init__(self, use_real_input=True):
        self.use_real_input = use_real_input
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading mobilenet_v2 model")
        mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        logging.info("Loaded mobilenet_v2 model")
        return mv2

    def get_example_inputs(self):
        tensor_size = (1, 3, 224, 224)
        if self.use_real_input:
            try:
                import urllib.request

                url, filename = (
                    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
                    "dog.jpg",
                )
                urllib.request.urlretrieve(url, filename)
                from PIL import Image
                from torchvision import transforms

                input_image = Image.open(filename)
                preprocess = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                input_tensor = preprocess(input_image)
                return (input_tensor.unsqueeze(0),)
            except Exception:
                logging.warning(
                    "Failed to download real input image, using random input"
                )
        return (torch.randn(tensor_size),)


class MV2UntrainedModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        mv2 = mobilenet_v2()
        return mv2

    def get_example_inputs(self):
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
