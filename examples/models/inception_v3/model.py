# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchvision.models import inception_v3  # @manual

from ..model_base import EagerModelBase


class InceptionV3Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading torchvision inception_v3 model")
        inception_v3_model = inception_v3(weights="IMAGENET1K_V1")
        logging.info("Loaded torchvision inception_v3 model")
        return inception_v3_model

    def get_example_inputs(self):
        input_shape = (1, 3, 224, 224)
        return (torch.randn(input_shape),)
