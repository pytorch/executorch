# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.channels_last_tagged_reshape_pass import (
    ChannelsLastTaggedReshapePass,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack._passes.remove_redundant_copy_pass import (
    RemoveRedundantCopyPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass


class TestChannelsLastTaggedReshapePass(unittest.TestCase):
    PassStage = RunPasses(
        [
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ChannelsLastTaggedReshapePass,
            RemoveRedundantCopyPass,
        ]
    )

    def setUp(self):
        torch._dynamo.reset()

    def run_tester(self, module, inputs):
        tester = Tester(
            module.eval(),
            inputs,
        )
        tester.export().to_edge_transform_and_lower().to_executorch().serialize().run_method_and_compare_outputs()

    class ChannelsLastToContiguous(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.linear1 = torch.nn.Linear(4, 3)

        def forward(self, x):
            y = self.linear1(x)
            y = y.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.contiguous_format)
            y = y.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.contiguous_format)
            y = y.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.contiguous_format)
            return self.conv1(y)

    ChannelsLastToContiguousModule = ChannelsLastToContiguous()

    class ContiguousToChannelsLast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.linear1 = torch.nn.Linear(4, 3)

        def forward(self, x):
            y = self.linear1(x)
            y = y.to(memory_format=torch.contiguous_format)
            y = y.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.contiguous_format)
            y = y.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.contiguous_format)
            y = y.to(memory_format=torch.channels_last)

            return self.conv1(y)

    ContiguousToChannelsLastModule = ContiguousToChannelsLast()

    class ImplicitRedundantOpRemoval(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            y = x.to(memory_format=torch.channels_last)
            y = self.upsample(y)
            y = y.to(memory_format=torch.contiguous_format)
            return self.conv(y)

    ImplicitRedundantOpRemovalModule = ImplicitRedundantOpRemoval()

    class QuantizableRedundantCopyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)

            x = x.to(memory_format=torch.contiguous_format)

            x = self.conv2(x)
            return x

    QuantizableRedundantCopyModule = QuantizableRedundantCopyModel()

    class ComplexQuantizableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(16, 8, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)

            x = x.to(memory_format=torch.contiguous_format)
            x = x.to(memory_format=torch.channels_last)
            x = x.to(memory_format=torch.contiguous_format)

            x = self.conv2(x)

            x = x.to(memory_format=torch.channels_last)
            x = x.to(memory_format=torch.contiguous_format)

            x = self.conv3(x)
            return x

    ComplexQuantizableModelModule = ComplexQuantizableModel()

    def test_implicit_redundant_op_removal(self):
        (
            Tester(self.ImplicitRedundantOpRemovalModule, (torch.randn(1, 3, 3, 3),))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten__to_copy_default": 2,
                }
            )
            .run_method_and_compare_outputs()
        )

    def test_quantized_redundant_copy_removal(self):
        (
            Tester(
                self.QuantizableRedundantCopyModule,
                (torch.randn(1, 3, 32, 32).to(memory_format=torch.channels_last),),
            )
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten__to_copy_default": 1,
                }
            )
            .run_method_and_compare_outputs(qtol=1)
        )

    def test_complex_quantized_redundant_copy_removal(self):
        (
            Tester(
                self.ComplexQuantizableModelModule,
                (torch.randn(1, 3, 32, 32).to(memory_format=torch.channels_last),),
            )
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten__to_copy_default": 1,
                }
            )
            .run_method_and_compare_outputs(qtol=1)
        )
