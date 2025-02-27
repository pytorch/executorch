# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestCat(unittest.TestCase):
    class Cat(torch.nn.Module):
        def __init__(self, dim=0):
            super().__init__()
            self.dim = dim

        def forward(self, *args):
            xs = [*args]
            x = torch.cat(xs, dim=self.dim)
            return x + x  # Quantize by propagation.

    def _test_cat(self, module, inputs, cat_num=1, quant=False, quant_ops=2):
        for legacy_mode in (True, False):
            tester = Tester(module, inputs)

            if quant:
                tester.quantize()

            tester.export().check_count({"torch.ops.aten.cat": 1})

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

            if legacy_mode:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()

            if quant:
                tester.check_not(["torch.ops.quantized_decomposed"])

            (
                tester.check_count(
                    {"torch.ops.higher_order.executorch_call_delegate": 1}
                )
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
        self._test_cat(self.Cat(), inputs)

    def test_fp16_cat3(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            torch.randn(1, 2, 3).to(torch.float16),
            torch.randn(3, 2, 3).to(torch.float16),
            torch.randn(2, 2, 3).to(torch.float16),
        )
        self._test_cat(self.Cat(), inputs)

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
        self._test_cat(self.Cat(), inputs)

    def test_fp16_cat5(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            torch.randn(1, 2, 3).to(torch.float16),
            torch.randn(3, 2, 3).to(torch.float16),
            torch.randn(2, 2, 3).to(torch.float16),
            torch.randn(5, 2, 3).to(torch.float16),
            torch.randn(5, 2, 3).to(torch.float16),
        )
        self._test_cat(self.Cat(), inputs)

    def test_fp16_cat_gt_5(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        for num_inputs in range(6, 10):
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3).to(torch.float16))
            self._test_cat(self.Cat(), tuple(inputs))

    def test_fp32_cat2(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3))
        self._test_cat(self.Cat(), inputs)

    def test_fp32_cat3(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3), torch.randn(2, 2, 3))
        self._test_cat(self.Cat(), inputs)

    def test_fp32_cat4(self):
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
        )
        self._test_cat(self.Cat(), inputs)

    def test_fp32_cat5(self):
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
            torch.randn(1, 2, 3),
        )
        self._test_cat(self.Cat(), inputs)

    def test_fp32_cat_gt_5(self):
        for num_inputs in range(6, 10):
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3))
            self._test_cat(self.Cat(), tuple(inputs))

    def test_qs8_cat2(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3))
        self._test_cat(self.Cat(), inputs, cat_num=2, quant=True)

    def test_qs8_cat3(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(3, 2, 3), torch.randn(2, 2, 3))
        self._test_cat(self.Cat(), inputs, cat_num=3, quant=True)

    def test_qs8_cat4(self):
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
        )
        self._test_cat(self.Cat(), inputs, cat_num=4, quant=True)

    def test_qs8_cat5(self):
        inputs = (
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(5, 2, 3),
            torch.randn(5, 2, 3),
        )
        self._test_cat(self.Cat(), inputs, cat_num=5, quant=True)

    def test_qs8_cat_gt_5(self):
        for num_inputs in range(6, 10):
            inputs = []
            for _ in range(num_inputs):
                inputs.append(torch.randn(1, 2, 3))
            self._test_cat(self.Cat(), tuple(inputs), cat_num=num_inputs, quant=True)

    def test_qs8_cat_with_empty_tensor(self):
        inputs = (
            torch.randn(0, 2, 3),
            torch.randn(1, 2, 3),
            torch.randn(3, 2, 3),
            torch.randn(0, 2, 3),
        )
        self._test_cat(self.Cat(), inputs, cat_num=4, quant=True)

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
