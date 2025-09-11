# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test.tester import Tester


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
