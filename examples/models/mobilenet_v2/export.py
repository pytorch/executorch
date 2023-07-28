# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)

# will refactor this in a separate file.
class MV2Model:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading mobilenet_v2 model")
        mv2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        logging.info("loaded mobilenet_v2 model")
        return mv2

    @staticmethod
    def get_example_inputs():
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
