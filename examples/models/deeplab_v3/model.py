# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchvision.models.segmentation import (
    deeplabv3,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
)

from ..model_base import EagerModelBase


class DeepLabV3ResNet50Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading deeplabv3_resnet50 model")
        deeplabv3_model = deeplabv3_resnet50(
            weights=deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        logging.info("loaded deeplabv3_resnet50 model")
        return deeplabv3_model

    def get_example_inputs(self):
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)


class DeepLabV3ResNet101Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading deeplabv3_resnet101 model")
        deeplabv3_model = deeplabv3_resnet101(
            weights=deeplabv3.DeepLabV3_ResNet101_Weights.DEFAULT
        )
        logging.info("loaded deeplabv3_resnet101 model")
        return deeplabv3_model

    def get_example_inputs(self):
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)
