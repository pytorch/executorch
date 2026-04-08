# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import models

from ..model_base import EagerModelBase


class SwinTModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading Swin Transformer Tiny model")
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        logging.info("Loaded Swin Transformer Tiny model")
        return model

    def get_example_inputs(self):
        return (torch.randn(1, 3, 224, 224),)
