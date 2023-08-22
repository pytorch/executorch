# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackQuantizedPartitioner2,
)
from executorch.backends.xnnpack.test.tester import Partition, Tester


class TestXNNPACKSub(unittest.TestCase):
    class SubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x - y
            z = z - y
            z = x - z
            w = z - z
            z = z - w
            return z

    def test_sub(self):
        sub_module = self.SubModule()
        model_inputs = (torch.randn(2, 3), torch.randn(2, 3))

        (
            Tester(sub_module, model_inputs)
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 5})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_sub_Tensor": 5})
            .partition()
            .check_count({"torch.ops.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_sub_Tensor"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_sub_quantized(self):
        sub_module = self.SubModule()
        model_inputs = (torch.randn(2, 3), torch.randn(2, 3))

        (
            Tester(sub_module, model_inputs)
            .quantize()
            .check(["torch.ops.quantized_decomposed"])
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 5})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_sub_Tensor": 5})
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner2))
            .check_count({"torch.ops.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_sub_Tensor"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    # Skipping since annotate patterns for sub are missing
    @unittest.expectedFailure
    def test_sub_quantized_pt2e(self):
        sub_module = self.SubModule()
        model_inputs = (torch.randn(2, 3), torch.randn(2, 3))

        (
            Tester(sub_module, model_inputs)
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 5})
            .quantize2()
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_sub_Tensor": 5})
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner2))
            .check_count({"torch.ops.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_sub_Tensor"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
