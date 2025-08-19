# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchvision.models import efficientnet_b4  # @manual
from torchvision.models.efficientnet import EfficientNet_B4_Weights

from ..model_base import EagerModelBase


class EfficientNetB4Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading EfficientNet-B4 model")
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.eval()
        logging.info("Loaded EfficientNet-B4 model")
        return model

    def get_example_inputs(self):
        # EfficientNet-B4 uses 380x380 input size
        tensor_size = (1, 3, 380, 380)
        return (torch.randn(tensor_size),)
