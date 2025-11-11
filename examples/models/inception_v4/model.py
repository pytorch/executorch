# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from timm.models import inception_v4

from ..model_base import EagerModelBase


class InceptionV4Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading inception_v4 model")
        m = inception_v4(pretrained=True)
        logging.info("Loaded inception_v4 model")
        return m

    def get_example_inputs(self):
        return (torch.randn(3, 299, 299).unsqueeze(0),)
