# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test.tester import Tester


class TestCloneMemoryFormat(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def run_tester(self, module, inputs):
        tester = Tester(
            module.eval(),
            inputs,
        )
        tester.export().to_edge_transform_and_lower().check_not(
            ["executorch_exir_dialects_edge__ops_aten_clone_default"]
        ).to_executorch().serialize().run_method_and_compare_outputs()

    class ChannelLastBeforeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            y = x.clone(memory_format=torch.channels_last)
            return self.linear(y)

    ChannelLastBeforeLinearModule = ChannelLastBeforeLinear()

    def test_channel_last_before_linear(self):
        self.run_tester(self.ChannelLastBeforeLinearModule, (torch.randn(1, 3, 3, 3),))

    class ContiguousBeforeConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = x.clone(memory_format=torch.contiguous_format)
            return self.conv(y)

    ContiguousBeforeConvModule = ContiguousBeforeConv()

    def test_contiguous_before_conv(self):
        self.run_tester(self.ContiguousBeforeConvModule, (torch.randn(1, 3, 6, 6),))

    class CloneChannelsLastToContiguous(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            # Start with channels_last input
            x_channels_last = x.to(memory_format=torch.channels_last)
            # Clone to contiguous format
            y = x_channels_last.clone(memory_format=torch.contiguous_format)
            return self.conv(y)

    CloneChannelsLastToContiguousModule = CloneChannelsLastToContiguous()

    def test_clone_channels_last_to_contiguous(self):
        self.run_tester(
            self.CloneChannelsLastToContiguousModule, (torch.randn(1, 3, 6, 6),)
        )

    class CloneContiguousToChannelsLast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            # Clone contiguous input to channels_last format
            y = x.clone(memory_format=torch.channels_last)
            return self.conv(y)

    CloneContiguousToChannelsLastModule = CloneContiguousToChannelsLast()

    def test_clone_contiguous_to_channels_last(self):
        self.run_tester(
            self.CloneContiguousToChannelsLastModule, (torch.randn(1, 3, 6, 6),)
        )

    class SimpleClone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            # Simple clone without memory format (should default to contiguous)
            y = x.clone()
            return self.conv(y)

    SimpleCloneModule = SimpleClone()

    def test_simple_clone(self):
        self.run_tester(self.SimpleCloneModule, (torch.randn(1, 3, 6, 6),))

    class QuantizedClone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = self.conv(x)
            y = y.clone(memory_format=torch.contiguous_format)
            return self.conv2(y)

    QuantizedCloneModule = QuantizedClone()

    def test_quantized_clone(self):
        tester = Tester(
            self.QuantizedCloneModule.eval(),
            (torch.randn(1, 3, 9, 9),),
        )

        tester.quantize().export().to_edge_transform_and_lower().check_not(
            [
                "executorch_exir_dialects_edge__ops_aten_clone_default",
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            ]
        ).to_executorch().serialize().run_method_and_compare_outputs(qtol=1)

    class ChainedClone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            # Chain multiple clones with different memory formats
            y = x.clone(memory_format=torch.channels_last)
            z = y.clone(memory_format=torch.contiguous_format)
            return self.conv(z)

    ChainedCloneModule = ChainedClone()

    def test_chained_clone(self):
        self.run_tester(self.ChainedCloneModule, (torch.randn(1, 3, 6, 6),))
