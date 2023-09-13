# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

# pyre-ignore
from torchvision.models import mobilenet_v2  # @manual
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from ..model_base import EagerModelBase


class MV2Model(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading mobilenet_v2 model")
        # pyre-ignore
        mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        logging.info("Loaded mobilenet_v2 model")
        class MobileNetV2Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mv2 = mv2

            def forward(self, x):
                out = self.mv2(x)
                return torch.nn.functional.softmax(out[0], dim=0)
                # probabilities = torch.nn.functional.softmax(out[0], dim=0)
                # return torch.topk(probabilities, 1)
                # return self.preds

        return MobileNetV2Wrapper()

    def get_example_inputs(self):
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
