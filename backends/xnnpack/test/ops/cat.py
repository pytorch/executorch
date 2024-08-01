# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestCat(unittest.TestCase):
    class Cat2(torch.nn.Module):
        def forward(self, arg1, arg2):
            xs = [arg1, arg2]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat3(torch.nn.Module):
        def forward(self, arg1, arg2, arg3):
            xs = [arg1, arg2, arg3]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat4(torch.nn.Module):
        def forward(self, arg1, arg2, arg3, arg4):
            xs = [arg1, arg2, arg3, arg4]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat5(torch.nn.Module):
        def forward(self, arg1, arg2, arg3, arg4, arg5):
            xs = [arg1, arg2, arg3, arg4, arg5]
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    def _test_cat(self, module, inputs, cat_num=1, quant=False, quant_ops=2):
        tester = Tester(module, inputs)

        if quant:
            tester.quantize()

        tester.export().check_count({"torch.ops.aten.cat": 1})
        tester.dump_artifact()

        if quant:
            # Expect multiple quantize ops - one per input, cat, and add.
            tester.check_node_count(
                {
                    # Q/DQ pair for each input and quantized op. For most tests, there are
                    # two quantized ops - cat and add.
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: (
                        cat_num + quant_ops
                    )
                }
            )

        tester.to_edge_transform_and_lower()

        if quant:
            tester.check_not(["torch.ops.quantized_decomposed"])

        (
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_cat2(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            torch.randn(1, 2, 3).to(torch.float16),
            torch.randn(3, 2, 3).to(torch.float16),
        )
        self._test_cat(self.Cat2(), inputs)

    def test_fp16_cat3(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            torch.randn(1, 2, 3).to(torch.float16),
            torch.randn(3, 2, 3).to(torch.float16),
            torch.randn(2, 2, 3).to(torch.float16),
        )
        self._test_cat(self.Cat3(), inputs)

    def test_fp16_cat4(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            torch.randn(1, 2, 3).to(torch.float16),
            torch.randn(3, 2, 3).to(torch.float16),
            torch.randn(2, 2, 3).to(torch.float16),
            torch.randn(5, 2, 3).to(torch.float16),
        )
        self._test_cat(self.Cat4(), inputs)

    def test_fp32_cat2(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3))
        self._test_cat(self.Cat2(), inputs)

    def test_fp32_cat3(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3), torch.randn(2, 2, 3))
        self._test_cat(self.Cat3(), inputs)

    def test_fp32_cat4(self):
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
        )
        self._test_cat(self.Cat4(), inputs)

    def test_qs8_cat2(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3))
        self._test_cat(self.Cat2(), inputs, cat_num=2, quant=True)

    def test_qs8_cat3(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3), torch.randn(2, 2, 3))
        self._test_cat(self.Cat3(), inputs, cat_num=3, quant=True)

    def test_qs8_cat4(self):
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
        )
        self._test_cat(self.Cat4(), inputs, cat_num=4, quant=True)

    def test_fp32_cat_unsupported(self):
        """
        XNNPACK only supports concatenating up to 4 values, so it should not delegate here.
        """
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
            torch.randn(1, 2, 3),
        )
        (
            Tester(self.Cat5(), inputs)
            .export()
            .check_count({"torch.ops.aten.cat": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_cat": 1})
        )

    class CatNegativeDim(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat([x, y], -1)

    def test_fp32_cat_negative_dim(self):
        inputs = (torch.randn(3, 2, 3), torch.randn(3, 2, 1))
        self._test_cat(self.CatNegativeDim(), inputs)

    class CatNhwc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            )

        def forward(self, x, y):
            x = self.conv(x)
            z = torch.concatenate((y, x, y, x), 1)
            return z + z

    @unittest.skip("T172862540 - Runtime failure.")
    def _test_qs8_cat_nhwc(self):
        inputs = (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3))
        self._test_cat(self.CatNhwc(), inputs, quant=True, quant_ops=3)

    class CatNhwc2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            )

        def forward(self, x, y):
            x = self.conv(x)
            y = self.conv(y)
            z = torch.concatenate((y, x, y, x), 3)
            return z + z

    @unittest.skip("T172862540 - Runtime failure.")
    def _test_qs8_cat_nhwc2(self):
        inputs = (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3))
        self._test_cat(self.CatNhwc(), inputs, quant=True, quant_ops=4)
