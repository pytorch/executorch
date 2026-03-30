# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import models

from ..model_base import EagerModelBase


class EfficientNetB0Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading EfficientNet-B0 model")
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        logging.info("Loaded EfficientNet-B0 model")
        return model

    def get_example_inputs(self):
        return (torch.randn(1, 3, 224, 224),)
