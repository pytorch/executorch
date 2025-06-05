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
from executorch.backends.xnnpack._passes.remove_redundant_ops_pass import (
    RemoveRedundantOpsPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass


class TestChannelsLastTaggedReshapePass(unittest.TestCase):
    PassStage = RunPasses(
        [
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ChannelsLastTaggedReshapePass,
            RemoveRedundantOpsPass,
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

    def test_redundant_to_copy_op_removal(self):
        (
            Tester(self.ChannelsLastToContiguousModule, (torch.randn(1, 3, 6, 4),))
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

    def test_redundant_to_copy_op_removal_2(self):
        (
            Tester(self.ContiguousToChannelsLastModule, (torch.randn(1, 3, 6, 4),))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten__to_copy_default": 1,
                }
            )
            .run_method_and_compare_outputs()
        )
