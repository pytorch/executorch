# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision import models

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


class ResNet18Model:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading torchvision resnet18 model")
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        logging.info("loaded torchvision resnet18 model")
        return resnet18

    @staticmethod
    def get_example_inputs():
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)


class ResNet50Model:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading torchvision resnet50 model")
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        logging.info("loaded torchvision resnet50 model")
        return resnet50

    @staticmethod
    def get_example_inputs():
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)
