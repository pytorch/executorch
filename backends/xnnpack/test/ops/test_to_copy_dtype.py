# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.export.utils import is_fbcode


class TestToCopyDtype(unittest.TestCase):
    """fp32 <-> fp16 and fp32 <-> bf16 casts are lowered to XNNPACK convert nodes."""

    def setUp(self):
        torch._dynamo.reset()

    class Fp16Inner(torch.nn.Module):
        """fp32 I/O contract around an fp16 model: the common deployment shape."""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1).half()

        def forward(self, x):
            return self.conv(x.to(torch.float16)).to(torch.float32)

    class CastOnly(torch.nn.Module):
        def forward(self, x):
            return (x.to(torch.float16) * 2.0).to(torch.float32)

    cast_ops = [
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default",
        "executorch_exir_dialects_edge__ops_aten__to_copy_default",
    ]

    def _lower(
        self,
        module,
        inputs,
        enable_bf16=False,
        skip_dim_order=False,
        dynamic_shapes=None,
    ):
        return (
            Tester(module.eval(), inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    partitioners=[XnnpackPartitioner(enable_bf16=enable_bf16)],
                    edge_compile_config=get_xnnpack_edge_compile_config(
                        skip_dim_order=skip_dim_order
                    ),
                )
            )
        )

    def _run(self, module, inputs, enable_bf16=False, skip_dim_order=False):
        (
            self._lower(
                module, inputs, enable_bf16=enable_bf16, skip_dim_order=skip_dim_order
            )
            .check_not(self.cast_ops)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=2e-2, rtol=2e-2)
        )

    def _run_both(self, module, inputs, enable_bf16=False):
        """
        With skip_dim_order, as LLMEdgeManager exports, casts stay aten._to_copy
        instead of becoming _to_dim_order_copy.
        """
        for skip_dim_order in (False, True):
            with self.subTest(skip_dim_order=skip_dim_order):
                self._run(module, inputs, enable_bf16, skip_dim_order)

    def _check_not_partitioned(self, module, inputs, enable_bf16=False, casts=1):
        for skip_dim_order in (False, True):
            with self.subTest(skip_dim_order=skip_dim_order):
                self._lower(
                    module,
                    inputs,
                    enable_bf16=enable_bf16,
                    skip_dim_order=skip_dim_order,
                ).check_count({self.cast_ops[skip_dim_order]: casts})

    def test_fp32_io_around_fp16_conv(self):
        self._run_both(self.Fp16Inner(), (torch.randn(1, 3, 8, 8),))

    def test_cast_only(self):
        self._run_both(self.CastOnly(), (torch.randn(2, 5),))

    def test_cast_only_dynamic(self):
        batch = torch.export.Dim("batch", min=1, max=8)
        for skip_dim_order in (False, True):
            with self.subTest(skip_dim_order=skip_dim_order):
                (
                    self._lower(
                        self.CastOnly(),
                        (torch.randn(2, 5),),
                        skip_dim_order=skip_dim_order,
                        dynamic_shapes=({0: batch},),
                    )
                    .check_not(self.cast_ops)
                    .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs(num_runs=5, atol=2e-2, rtol=2e-2)
                )

    def test_not_partitioned(self):
        """
        Only fp32 <-> fp16 and fp32 <-> bf16 casts that keep the dim order are
        partitioned, and with skip_dim_order no layout-only copy is.
        """

        class ToInt(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.int32)

        class ToHalfChannelsLast(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float16, memory_format=torch.channels_last)

        for module in (ToInt(), ToHalfChannelsLast()):
            with self.subTest(type(module).__name__):
                self._check_not_partitioned(module, (torch.randn(1, 3, 4, 4),))

        class ToChannelsLast(torch.nn.Module):
            def forward(self, x):
                return x.to(memory_format=torch.channels_last)

        self._lower(
            ToChannelsLast(), (torch.randn(1, 3, 4, 4),), skip_dim_order=True
        ).check_not(["torch.ops.higher_order.executorch_call_delegate"]).check_count(
            {self.cast_ops[1]: 1}
        )

    def test_dtype_cast_to_contiguous_from_channels_last_not_partitioned(self):
        """
        With dim order, the layout-only copy is partitioned and the cast that
        changes the dim order is not. With skip_dim_order, neither is. Such a
        program only lowers there: a channels_last copy left outside the
        delegate cannot be emitted without dim order.
        """

        class ChannelsLastToHalfContiguous(torch.nn.Module):
            def forward(self, x):
                x = x.to(memory_format=torch.channels_last)
                return x.to(torch.float16, memory_format=torch.contiguous_format)

        inputs = (torch.randn(1, 3, 4, 4),)
        (
            self._lower(ChannelsLastToHalfContiguous(), inputs)
            .check_count({self.cast_ops[0]: 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=2e-2, rtol=2e-2)
        )
        self._lower(
            ChannelsLastToHalfContiguous(), inputs, skip_dim_order=True
        ).check_not(["torch.ops.higher_order.executorch_call_delegate"]).check_count(
            {self.cast_ops[1]: 2}
        )

    def test_dtype_cast_preserving_channels_last_partitioned(self):
        """
        The cast reads a channels_last tensor and keeps its dim order, so it is
        partitioned. With skip_dim_order, the layout-only copy before it stays
        outside the delegate, which only lowers there, as above.
        """

        class ChannelsLastToHalf(torch.nn.Module):
            def forward(self, x):
                return x.to(memory_format=torch.channels_last).to(torch.float16)

        inputs = (torch.randn(1, 3, 4, 4),)
        self._run(ChannelsLastToHalf(), inputs)
        self._lower(ChannelsLastToHalf(), inputs, skip_dim_order=True).check_count(
            {
                self.cast_ops[1]: 1,
                "torch.ops.higher_order.executorch_call_delegate": 1,
            }
        )

    def test_cast_of_nhwc_conv_output(self):
        """
        The cast reads the convolution's NHWC output inside the delegate, and
        its result is converted back to NCHW at the delegate output.
        """

        class ConvToHalf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3)

            def forward(self, x):
                return self.conv(x).to(torch.float16)

        for skip_dim_order in (False, True):
            with self.subTest(skip_dim_order=skip_dim_order):
                (
                    self._lower(
                        ConvToHalf(),
                        (torch.randn(1, 3, 10, 10),),
                        skip_dim_order=skip_dim_order,
                    )
                    .check_not(self.cast_ops)
                    .check_xnnpack_node_count(
                        {"XNNConv2d": 1, "XNNConvert": 1, "XNNStaticTranspose": 2}
                    )
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs(atol=2e-2, rtol=2e-2)
                )

    def test_integer_constant_cast_not_partitioned(self):
        """
        Export lifts the 2 in x.float() * 2 as an integer constant and casts it
        to float. XNNPACK does not convert integers, so that cast stays outside
        the delegate, while the cast of x is partitioned.
        """

        class UpcastTimesInt(torch.nn.Module):
            def forward(self, x):
                return x.float() * 2

        for skip_dim_order in (False, True):
            with self.subTest(skip_dim_order=skip_dim_order):
                (
                    self._lower(
                        UpcastTimesInt(),
                        (torch.randn(2, 5).to(torch.float16),),
                        skip_dim_order=skip_dim_order,
                    )
                    .check_count(
                        {
                            self.cast_ops[skip_dim_order]: 1,
                            "torch.ops.higher_order.executorch_call_delegate": 1,
                        }
                    )
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs(atol=2e-2, rtol=2e-2)
                )

    class RequantizeAroundCast(torch.nn.Module):
        def forward(self, x):
            qd = torch.ops.quantized_decomposed
            x = qd.quantize_per_tensor.default(x, 0.04, 0, -128, 127, torch.int8)
            x = qd.dequantize_per_tensor.default(x, 0.04, 0, -128, 127, torch.int8)
            x = qd.quantize_per_tensor.default(
                x.to(torch.float16), 0.1, 0, -128, 127, torch.int8
            )
            return qd.dequantize_per_tensor.default(x, 0.1, 0, -128, 127, torch.int8)

    class RequantizeAroundCastConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.requantize = TestToCopyDtype.RequantizeAroundCast()
            self.conv = torch.nn.Conv2d(4, 8, 3)

        def forward(self, x):
            return self.conv(self.requantize(x))

    def test_requantize_around_cast(self):
        """
        The q/dq around a cast stay explicit, so the requantize from a scale of
        0.04 to 0.1 is kept. Before a convolution, the layout pass copies the
        requantized value to NHWC between its own q/dq, and that copy must not
        take the cast and the requantize with it when it is removed.
        """
        cases = (
            (self.RequantizeAroundCast(), torch.randn(2, 8)),
            (self.RequantizeAroundCastConv(), torch.randn(1, 4, 8, 8)),
        )
        for module, x in cases:
            for skip_dim_order in (False, True):
                with self.subTest(type(module).__name__, skip_dim_order=skip_dim_order):
                    (
                        self._lower(module, (x,), skip_dim_order=skip_dim_order)
                        .check_count(
                            {"torch.ops.higher_order.executorch_call_delegate": 1}
                        )
                        .check_not(self.cast_ops)
                        .to_executorch()
                        .serialize()
                        .run_method_and_compare_outputs(atol=1e-3, rtol=1e-3)
                    )

    class Bf16Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 8).to(torch.bfloat16)

        def forward(self, x):
            return self.linear(x.to(torch.bfloat16)).to(torch.float32)

    class Bf16Upcast(torch.nn.Module):
        """A bf16 model that computes a step in fp32 and casts back."""

        def forward(self, x):
            return (x.to(torch.float32) * 2.0).to(torch.bfloat16)

    class Bf16SoftmaxUpcast(torch.nn.Module):
        """Softmax computed in fp32 on a bf16 input, cast back to bf16."""

        def forward(self, x):
            return torch.softmax(x, -1, dtype=torch.float32).to(torch.bfloat16)

    class ToBf16(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.bfloat16)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_fp32_io_around_bf16_linear(self):
        self._run_both(self.Bf16Inner(), (torch.randn(4, 16),), enable_bf16=True)

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_upcast(self):
        self._run_both(
            self.Bf16Upcast(),
            (torch.randn(2, 5, dtype=torch.bfloat16),),
            enable_bf16=True,
        )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_bf16_softmax_upcast(self):
        self._run_both(
            self.Bf16SoftmaxUpcast(),
            (torch.randn(2, 16, 32, dtype=torch.bfloat16),),
            enable_bf16=True,
        )

    @unittest.skipIf(is_fbcode(), "skipping for fbcode")
    def test_fp32_to_bf16_rounds_like_pytorch(self):
        """
        XNNPACK's convert must round fp32 to bf16 exactly as PyTorch does,
        including values halfway between two bf16 numbers.
        """
        halfway = torch.tensor(
            [1 + 2**-8, 1 + 3 * 2**-8, -(1 + 3 * 2**-8), 2**-100 * (1 + 3 * 2**-8)]
        )
        x = torch.cat([torch.randn(4096), halfway, torch.tensor([3.0e38, 0.0, -0.0])])
        (
            Tester(self.ToBf16().eval(), (x,))
            .export()
            .to_edge_transform_and_lower(
                ToEdgeTransformAndLower(
                    partitioners=[XnnpackPartitioner(enable_bf16=True)]
                )
            )
            .check_not(self.cast_ops)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=(x,), atol=0, rtol=0)
        )

    def test_bf16_cast_not_partitioned_without_enable_bf16(self):
        self._check_not_partitioned(
            self.Bf16Upcast(), (torch.randn(2, 5, dtype=torch.bfloat16),), casts=2
        )

    def test_fp16_bf16_cast_not_partitioned(self):
        class HalfToBf16(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.bfloat16)

        self._check_not_partitioned(
            HalfToBf16(), (torch.randn(2, 5, dtype=torch.float16),), enable_bf16=True
        )
