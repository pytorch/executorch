# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class TestStaticConstantPad(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class NHWCStaticConstantPad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
            self.conv2 = torch.nn.Conv2d(in_channels=13, out_channels=13, kernel_size=1)

        def forward(self, x):
            a = self.conv1(x)
            pad_6 = (1, 2, 3, 4, 5, 6)
            a = torch.nn.functional.pad(
                input=a,
                pad=pad_6,
                mode="constant",
                value=3.1,
            )
            # tensorshape = [1, 13, 10, 7]
            a = self.conv2(a)

            return a

        def sample_inputs(self):
            # NCHW
            return (torch.randn(1, 2, 3, 4),)

    class StaticConstantPadFunctional(torch.nn.Module):
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

            # Pad quantizes by propagation

            return (a + a, b + b, c + c, d + d, e + e, f + f, g + g, h + h)

    class StaticConstantPad2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = torch.nn.ConstantPad2d([1, 2, 3, 4], 2.3)

        def forward(self, x):
            y = self.pad(x)
            # Pad quantizes by propagation
            z = y + y
            return z

    def _test_static_constant_pad_functional(self, inputs):
        (
            Tester(self.StaticConstantPadFunctional(), inputs)
            .export()
            .check_count({"torch.ops.aten.pad.default": 8})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class NegativePadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = torch.nn.ConstantPad2d((0, 0, -2, 2), 0.0)

        def forward(self, input):
            input = self.pad(input)
            return input

    def test_negative_pad_model_with_ints(self):
        """Test that negative padding with integer inputs falls back to PyTorch implementation as XNNPACK does not support negative padding dimensions"""
        input_tensor = torch.tensor([[4], [5], [6]])
        model = self.NegativePadModel()
        model.eval()
        model.to("cpu")

        exported_model = export(model, (input_tensor,))

        executorch_program = to_edge_transform_and_lower(
            exported_model, partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        self.assertIsNotNone(executorch_program)

    def test_negative_pad_model_with_floats(self):
        """Test that negative padding with float inputs is now rejected by XNNPACK partitioner as XNNPACK does not support negative padding dimensions"""
        input_tensor = torch.tensor([[4.0], [5.0], [6.0]])
        model = self.NegativePadModel()
        model.eval()
        model.to("cpu")

        exported_model = export(model, (input_tensor,))

        executorch_program = to_edge_transform_and_lower(
            exported_model, partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        self.assertIsNotNone(executorch_program)

    def test_fp16_static_constant_pad_functional(self):
        inputs = (
            torch.randn(size=(5, 4, 3, 2)).to(torch.float16),
            torch.randn(size=(5, 3, 2)).to(torch.float16),
            torch.randn(size=(4, 3)).to(torch.float16),
        )
        self._test_static_constant_pad_functional(inputs)

    def test_fp32_static_constant_pad_functional(self):
        inputs = (
            torch.randn(size=(5, 4, 3, 2)),
            torch.randn(size=(5, 3, 2)),
            torch.randn(size=(4, 3)),
        )
        self._test_static_constant_pad_functional(inputs)

    def test_constant_pad_nd(self):
        class ConstantPad(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                pad_6 = (1, 2, 3, 4, 5, 6)
                pad_4 = (1, 2, 3, 4)
                pad_2 = (1, 2)
                a = torch.constant_pad_nd(input=x, pad=pad_6)
                b = torch.constant_pad_nd(input=y, pad=pad_4)
                c = torch.constant_pad_nd(input=z, pad=pad_2)

                return (a + a, b + b, c + c)

        inputs = (
            torch.randn(size=(5, 4, 3, 2)),
            torch.randn(size=(5, 3, 2)),
            torch.randn(size=(4, 3)),
        )
        (
            Tester(ConstantPad(), inputs)
            .export()
            .check_count({"torch.ops.aten.constant_pad_nd.default": 3})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_static_constant_pad_functional(self):
        class Pad(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                z = torch.nn.functional.pad(
                    input=x,
                    pad=(2, 1),
                    mode="constant",
                    value=2.3,
                )
                return z + z

        inputs = (torch.randn(size=(1, 2)),)
        (
            Tester(Pad(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.pad.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_static_constant_pad_2d(self):
        inputs = (torch.randn(size=(5, 4, 3, 2)),)
        (
            Tester(self.StaticConstantPad2d(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.pad.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_static_constant_pad_nhwc(self):
        conv = self.NHWCStaticConstantPad()
        inputs = conv.sample_inputs()
        (
            Tester(conv, inputs)
            .export()
            .check_count({"torch.ops.aten.pad.default": 1})
            .dump_artifact()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default",
                    "executorch_exir_dialects_edge__ops_aten_convolution_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
