# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from torchsr.models import edsr_r16f64


class DynamicEDSR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = edsr_r16f64(2, False).eval()

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=True,
            antialias=False,
        )
        return self.model(x)


class TestEDSR(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    edsr = edsr_r16f64(2, False).eval()  # noqa
    model_inputs = (torch.randn(1, 3, 224, 224),)
    dynamic_shapes = (
        {
            2: torch.export.Dim("height", min=224, max=455),
            3: torch.export.Dim("width", min=224, max=455),
        },
    )

    def test_fp32_edsr(self):
        (
            Tester(self.edsr, self.model_inputs)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("T187799178: Debugging Numerical Issues with Calibration")
    def _test_qs8_edsr(self):
        (
            Tester(self.edsr, self.model_inputs)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    # TODO: Delete and only used calibrated test after T187799178
    def test_qs8_edsr_no_calibrate(self):
        (
            Tester(self.edsr, self.model_inputs)
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_edsr_dynamic(self):
        (
            Tester(DynamicEDSR(), self.model_inputs, self.dynamic_shapes)
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
