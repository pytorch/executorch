# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from torchvision.models.segmentation import deeplabv3, deeplabv3_resnet50  # @manual


class DL3Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = deeplabv3_resnet50(
            weights=deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT
        )

    def forward(self, *args):
        return self.m(*args)["out"]


class TestDeepLabV3(unittest.TestCase):
    dl3 = DL3Wrapper()
    dl3 = dl3.eval()
    model_inputs = (torch.randn(1, 3, 224, 224),)

    def test_fp32_dl3(self):

        (
            Tester(self.dl3, self.model_inputs)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
