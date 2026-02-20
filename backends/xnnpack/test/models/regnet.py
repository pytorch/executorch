# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision
from executorch.backends.xnnpack.test.tester import Tester


class TestInceptionV4(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    regnet = torchvision.models.regnet_y_32gf()
    model_inputs = (torch.randn(3, 299, 299).unsqueeze(0),)

    def test_fp32_regnet(self):
        (
            Tester(self.regnet, self.model_inputs)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
