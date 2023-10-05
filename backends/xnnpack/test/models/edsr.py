# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test.tester import Tester
from torchsr.models import edsr_r16f64


class TestEDSR(unittest.TestCase):
    edsr = edsr_r16f64(2, False).eval()  # noqa
    model_inputs = (torch.ones(1, 3, 224, 224),)

    def test_fp32_edsr(self):
        (
            Tester(self.edsr, self.model_inputs)
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_qs8_edsr(self):
        (
            Tester(self.edsr, self.model_inputs)
            .quantize()
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
