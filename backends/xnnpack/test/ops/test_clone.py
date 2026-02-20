# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestClone(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Clone(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = torch.clone(x)
            return z

    class CloneWithMemoryFormat(torch.nn.Module):
        def __init__(self, memory_format):
            super().__init__()
            self.memory_format = memory_format

        def forward(self, x):
            z = torch.clone(x, memory_format=self.memory_format)
            return z

    def _test_clone_partitioned(self, inputs):
        """Test that dim-order preserving clones are partitioned (removed)"""
        (
            Tester(self.Clone(), inputs)
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
            .dump_artifact()
            .to_edge_transform_and_lower()
            .dump_artifact()
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_clone(self):
        """Test FP16 clone - should be partitioned"""
        inputs = (torch.randn(2, 3, 4, 5).to(torch.float16),)
        self._test_clone_partitioned(inputs)

    def test_fp32_clone(self):
        """Test FP32 clone - should be partitioned"""
        inputs = (torch.randn(2, 3, 4, 5),)
        self._test_clone_partitioned(inputs)

    def test_fp32_clone_2d(self):
        """Test FP32 clone with 2D tensor - should be partitioned"""
        inputs = (torch.randn(10, 20),)
        self._test_clone_partitioned(inputs)

    def test_fp32_clone_3d(self):
        """Test FP32 clone with 3D tensor - should be partitioned"""
        inputs = (torch.randn(2, 3, 4),)
        self._test_clone_partitioned(inputs)

    def test_fp32_clone_with_contiguous_format(self):
        """Test FP32 clone with contiguous memory format - should be partitioned"""
        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(self.CloneWithMemoryFormat(torch.contiguous_format), inputs)
            .export()
            .to_edge_transform_and_lower()
            .dump_artifact()
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_clone_with_channels_last_not_partitioned(self):
        """Test FP32 clone with channels_last memory format - should NOT be partitioned"""
        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(self.CloneWithMemoryFormat(torch.channels_last), inputs)
            .export()
            .to_edge_transform_and_lower()
            # Clone with channels_last changes dim order, so should NOT be delegated
            .check(
                [
                    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_clone_channels_last_to_contiguous_not_partitioned(self):
        """Test clone from channels_last to contiguous - should NOT be partitioned"""

        class CloneChannelsLastToContiguous(torch.nn.Module):
            def forward(self, x):
                # Start with channels_last input
                y = x.to(memory_format=torch.channels_last)
                # Clone back to contiguous (changes dim order)
                z = torch.clone(y, memory_format=torch.contiguous_format)
                return z

        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(CloneChannelsLastToContiguous(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .dump_artifact()
            # Clone that changes dim order should NOT be delegated
            .check(
                [
                    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
