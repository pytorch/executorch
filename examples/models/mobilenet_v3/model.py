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


class MV3Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading mobilenet_v3 model")
        mv3_small = models.mobilenet_v3_small(pretrained=True)
        logging.info("loaded mobilenet_v3 model")
        return mv3_small

    def get_example_inputs(self):
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
