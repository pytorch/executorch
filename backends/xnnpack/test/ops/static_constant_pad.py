# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestStaticConstantPad(unittest.TestCase):
    class StaticConstantPad(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            pad_6 = (1, 2, 3, 4, 5, 6)
            pad_4 = (1, 2, 3, 4)
            pad_2 = (1, 2)
            a = torch.nn.functional.pad(
                input=x,
                pad=pad_6,
                mode="constant",
                value=2.3,
            )
            b = torch.nn.functional.pad(
                input=x,
                pad=pad_4,
                mode="constant",
                value=1.3,
            )
            c = torch.nn.functional.pad(
                input=x,
                pad=pad_2,
                mode="constant",
                value=2.1,
            )
            d = torch.nn.functional.pad(
                input=y,
                pad=pad_6,
                mode="constant",
                value=2.7,
            )
            e = torch.nn.functional.pad(
                input=y,
                pad=pad_4,
                mode="constant",
                value=1.9,
            )
            f = torch.nn.functional.pad(
                input=y,
                pad=pad_2,
                mode="constant",
                value=3.1,
            )
            g = torch.nn.functional.pad(
                input=z,
                pad=pad_4,
                mode="constant",
                value=2.9,
            )
            h = torch.nn.functional.pad(
                input=z,
                pad=pad_2,
                mode="constant",
                value=1.2,
            )
            return (a, b, c, d, e, f, g, h)

    def test_fp32_static_constant_pad(self):
        inputs = (
            torch.randn(size=(5, 4, 3, 2)),
            torch.randn(size=(5, 3, 2)),
            torch.randn(size=(4, 3)),
        )
        (
            Tester(self.StaticConstantPad(), inputs)
            .export()
            .check_count({"torch.ops.aten.constant_pad_nd.default": 8})
            .to_edge()
            .check_count(
                {"executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default": 8}
            )
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"]
            )
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
