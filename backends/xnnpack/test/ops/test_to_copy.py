# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

from executorch.backends.xnnpack.test.tester import Tester, ToEdgeTransformAndLower


class TestChannelsLastTaggedReshapePass(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def run_tester(self, module, inputs):
        tester = Tester(
            module.eval(),
            inputs,
        )
        tester.export().to_edge_transform_and_lower().check_not(
            ["executorch_exir_dialects_edge__ops_aten__to_copy_default"]
        ).to_executorch().serialize().run_method_and_compare_outputs()

    class ChannelLastBeforeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            y = x.to(memory_format=torch.channels_last)
            return self.linear(y)

    ChannelLastBeforeLinearModule = ChannelLastBeforeLinear()

    def test_channel_last_before_linear(self):
        self.run_tester(self.ChannelLastBeforeLinearModule, (torch.randn(1, 3, 3, 3),))

    class ContiguousBeforeConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = x.to(memory_format=torch.contiguous_format)
            return self.conv(y)

    ContiguousBeforeConvModule = ContiguousBeforeConv()

    def test_contiguous_before_conv(self):
        self.run_tester(self.ContiguousBeforeConvModule, (torch.randn(1, 3, 6, 6),))

    class DtypeAndMemoryFormatConversion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = x.to(torch.float, memory_format=torch.channels_last)
            return self.conv(y)

    DtypeAndMemoryFormatConversionModule = DtypeAndMemoryFormatConversion()

    def test_dtype_and_memory_format_conversion(self):
        self.run_tester(
            self.DtypeAndMemoryFormatConversionModule,
            (torch.randint(0, 10, (1, 3, 6, 6), dtype=torch.int32),),
        )

    class DtypeAndMemoryFormatWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            y = x.to(torch.float, memory_format=torch.channels_last)
            return self.linear(y)

    DtypeAndMemoryFormatWithLinearModule = DtypeAndMemoryFormatWithLinear()

    def test_dtype_and_memory_format_with_linear(self):
        self.run_tester(
            self.DtypeAndMemoryFormatWithLinearModule,
            (torch.randint(0, 10, (1, 3, 3, 3), dtype=torch.int16),),
        )

    class QuantizedToCopy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = self.conv(x)
            y = y.to(memory_format=torch.contiguous_format)
            return self.conv2(y)

    QuantizedToCopyModule = QuantizedToCopy()

    def test_quantized_to_copy(self):
        tester = Tester(
            self.QuantizedToCopyModule.eval(),
            (torch.randn(1, 3, 9, 9),),
        )

        tester.quantize().export().to_edge_transform_and_lower().check_not(
            [
                "executorch_exir_dialects_edge__ops_aten__to_copy_default",
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            ]
        ).to_executorch().serialize().run_method_and_compare_outputs(qtol=1)


class TestDtypeConvertToCopy(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def _run_dtype_convert(self, module, inputs, should_delegate=True):
        tester = Tester(module.eval(), inputs)
        tester = tester.export().to_edge_transform_and_lower(
            ToEdgeTransformAndLower(
                partitioners=[XnnpackPartitioner(enable_bf16=True)]
            )
        )
        if should_delegate:
            tester.check_not(
                ["executorch_exir_dialects_edge__ops_aten__to_copy_default"]
            ).check_not(
                ["executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"]
            )
        else:
            # _to_copy should remain outside the delegate.
            tester.check(
                [
                    "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
                ]
            )
        tester.to_executorch().serialize().run_method_and_compare_outputs()

    class Fp32ToFp16(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float16)

    class Fp16ToFp32(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float32)

    class Fp32ToBf16(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.bfloat16)

    class Bf16ToFp32(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float32)

    class Fp16ToBf16(torch.nn.Module):
        # Intentionally unsupported: fp16 <-> bf16 stays portable.
        def forward(self, x):
            return x.to(torch.bfloat16)

    class ParamDtypeConvert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(3, 3))

        def forward(self, x):
            # Conversion of a param/constant must not be folded into the
            # delegate; it should be constant-folded outside.
            p = self.param.to(torch.float16)
            return x + p

    class CombinedDtypeAndLayout(torch.nn.Module):
        def forward(self, x):
            # Mixed dtype + channels_last is rejected by the partitioner.
            return x.to(dtype=torch.float16, memory_format=torch.channels_last)

    def test_fp32_to_fp16(self):
        self._run_dtype_convert(
            self.Fp32ToFp16(), (torch.randn(2, 3),), should_delegate=True
        )

    def test_fp16_to_fp32(self):
        self._run_dtype_convert(
            self.Fp16ToFp32(),
            (torch.randn(2, 3).to(torch.float16),),
            should_delegate=True,
        )

    def test_fp32_to_bf16(self):
        self._run_dtype_convert(
            self.Fp32ToBf16(), (torch.randn(2, 3),), should_delegate=True
        )

    def test_bf16_to_fp32(self):
        self._run_dtype_convert(
            self.Bf16ToFp32(),
            (torch.randn(2, 3).to(torch.bfloat16),),
            should_delegate=True,
        )

    def test_fp16_to_bf16_not_delegated(self):
        # fp16 <-> bf16 is intentionally not supported via xnn_define_convert.
        self._run_dtype_convert(
            self.Fp16ToBf16(),
            (torch.randn(2, 3).to(torch.float16),),
            should_delegate=False,
        )

    def test_param_dtype_not_delegated(self):
        self._run_dtype_convert(
            self.ParamDtypeConvert(), (torch.randn(3, 3),), should_delegate=False
        )

    def test_combined_dtype_and_layout_not_delegated(self):
        self._run_dtype_convert(
            self.CombinedDtypeAndLayout(),
            (torch.randn(1, 3, 4, 4),),
            should_delegate=False,
        )
