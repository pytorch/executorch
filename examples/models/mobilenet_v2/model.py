# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import models
from ..model_base import EagerModelBase
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


class MV2Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading mobilenet_v2 model")
        mv2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        logging.info("loaded mobilenet_v2 model")
        return mv2

    def get_example_inputs(self):
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
