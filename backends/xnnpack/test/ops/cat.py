# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestCat(unittest.TestCase):
    class Cat(torch.nn.Module):
        def forward(self, xs):
            x = torch.cat(xs)
            return x + x  # Quantize by propagation.

    class Cat2(torch.nn.Module):
        def forward(self, xs):
            return torch.cat(xs)

    def _test_cat(self, module, inputs, quant=False, quant_ops=2):
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
                        len(inputs[0]) + quant_ops
                    )
                }
            )

        (
            tester.to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_cat": 1})
            .partition()
        )

        if quant:
            tester.check_not(["torch.ops.quantized_decomposed"])

        (
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_fp16_cat2(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            (
                torch.ones(1, 2, 3).to(torch.float16),
                torch.ones(3, 2, 3).to(torch.float16),
            ),
        )
        self._test_cat(self.Cat2(), inputs)

    def test_fp16_cat3(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            (
                torch.ones(1, 2, 3).to(torch.float16),
                torch.ones(3, 2, 3).to(torch.float16),
                torch.ones(2, 2, 3).to(torch.float16),
            ),
        )
        self._test_cat(self.Cat2(), inputs)

    def test_fp16_cat4(self):
        """
        Using Clamp2 because fp16 add is done in fp32 ATM. Need to fix that first.
        """
        inputs = (
            (
                torch.ones(1, 2, 3).to(torch.float16),
                torch.ones(3, 2, 3).to(torch.float16),
                torch.ones(2, 2, 3).to(torch.float16),
                torch.ones(5, 2, 3).to(torch.float16),
            ),
        )
        self._test_cat(self.Cat2(), inputs)

    def test_fp32_cat2(self):
        inputs = ((torch.ones(1, 2, 3), torch.ones(3, 2, 3)),)
        self._test_cat(self.Cat(), inputs)

    def test_fp32_cat3(self):
        inputs = ((torch.ones(1, 2, 3), torch.ones(3, 2, 3), torch.ones(2, 2, 3)),)
        self._test_cat(self.Cat(), inputs)

    def test_fp32_cat4(self):
        inputs = (
            (
                torch.ones(1, 2, 3),
                torch.ones(3, 2, 3),
                torch.ones(2, 2, 3),
                torch.ones(5, 2, 3),
            ),
        )
        self._test_cat(self.Cat(), inputs)

    def test_qs8_cat2(self):
        inputs = ((torch.ones(1, 2, 3), torch.ones(3, 2, 3)),)
        self._test_cat(self.Cat(), inputs, quant=True)

    def test_qs8_cat3(self):
        inputs = ((torch.ones(1, 2, 3), torch.ones(3, 2, 3), torch.ones(2, 2, 3)),)
        self._test_cat(self.Cat(), inputs, quant=True)

    def test_qs8_cat4(self):
        inputs = (
            (
                torch.ones(1, 2, 3),
                torch.ones(3, 2, 3),
                torch.ones(2, 2, 3),
                torch.ones(5, 2, 3),
            ),
        )
        self._test_cat(self.Cat(), inputs, quant=True)

    def test_fp32_cat_unsupported(self):
        """
        XNNPACK only supports concatenating up to 4 values, so it should not delegate here.
        """
        inputs = (
            (
                torch.ones(1, 2, 3),
                torch.ones(3, 2, 3),
                torch.ones(2, 2, 3),
                torch.ones(5, 2, 3),
                torch.ones(1, 2, 3),
            ),
        )
        (
            Tester(self.Cat(), inputs)
            .export()
            .check_count({"torch.ops.aten.cat": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_cat": 1})
            .partition()
            .check_count({"executorch_exir_dialects_edge__ops_aten_cat": 1})
        )

    class CatNegativeDim(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat([x, y], -1)

    def test_fp32_cat_negative_dim(self):
        inputs = (torch.ones(3, 2, 3), torch.ones(3, 2, 1))
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
    def test_qs8_cat_nhwc(self):
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
    def test_qs8_cat_nhwc2(self):
        inputs = (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3))
        self._test_cat(self.CatNhwc(), inputs, quant=True, quant_ops=4)
