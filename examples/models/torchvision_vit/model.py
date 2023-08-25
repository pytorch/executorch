# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from ..model_base import EagerModelBase
from torchvision import models

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


class TorchVisionViTModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading torchvision vit_b_16 model")
        vit_b_16 = models.vit_b_16(weights="IMAGENET1K_V1")
        logging.info("loaded torchvision vit_b_16 model")
        return vit_b_16

    def get_example_inputs(self):
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)
