# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision

from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import Quantize


class TestResNet18(unittest.TestCase):
    def test_fp32_resnet18(self):
        inputs = (torch.ones(1, 3, 224, 224),)
        (
            Tester(torchvision.models.resnet18(), inputs)
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_resnet18(self):
        inputs = (torch.ones(1, 3, 224, 224),)
        (
            Tester(torchvision.models.resnet18(), inputs)
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
