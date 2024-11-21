# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision

from executorch.backends.xnnpack.test.tester import Quantize, Tester


class TestResNet18(unittest.TestCase):
    inputs = (torch.randn(1, 3, 224, 224),)
    dynamic_shapes = (
        {
            2: torch.export.Dim("height", min=224, max=455),
            3: torch.export.Dim("width", min=224, max=455),
        },
    )

    class DynamicResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torchvision.models.resnet18()

        def forward(self, x):
            x = torch.nn.functional.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=True,
                antialias=False,
            )
            return self.model(x)

    def _test_exported_resnet(self, tester):
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_convolution_default",
                    "executorch_exir_dialects_edge__ops_aten_mean_dim",
                ]
            )
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_resnet18(self):
        self._test_exported_resnet(Tester(torchvision.models.resnet18(), self.inputs))

    @unittest.skip("T187799178: Debugging Numerical Issues with Calibration")
    def _test_qs8_resnet18(self):
        quantized_tester = Tester(torchvision.models.resnet18(), self.inputs).quantize()
        self._test_exported_resnet(quantized_tester)

    # TODO: Delete and only used calibrated test after T187799178
    def test_qs8_resnet18_no_calibration(self):
        quantized_tester = Tester(torchvision.models.resnet18(), self.inputs).quantize(
            Quantize(calibrate=False)
        )
        self._test_exported_resnet(quantized_tester)

    def test_fp32_resnet18_dynamic(self):
        self._test_exported_resnet(
            Tester(self.DynamicResNet(), self.inputs, self.dynamic_shapes)
        )

    @unittest.skip("T187799178: Debugging Numerical Issues with Calibration")
    def _test_qs8_resnet18_dynamic(self):
        self._test_exported_resnet(
            Tester(self.DynamicResNet(), self.inputs, self.dynamic_shapes).quantize()
        )

    # TODO: Delete and only used calibrated test after T187799178
    def test_qs8_resnet18_dynamic_no_calibration(self):
        self._test_exported_resnet(
            Tester(self.DynamicResNet(), self.inputs, self.dynamic_shapes).quantize(
                Quantize(calibrate=False)
            )
        )
