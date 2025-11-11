# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchvision.models import (  # @manual
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)

from ..model_base import EagerModelBase


class ResNet18Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading torchvision resnet18 model")
        resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        logging.info("Loaded torchvision resnet18 model")
        return resnet18_model

    def get_example_inputs(self):
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)


class ResNet50Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading torchvision resnet50 model")
        resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        logging.info("Loaded torchvision resnet50 model")
        return resnet50_model

    def get_example_inputs(self):
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)
