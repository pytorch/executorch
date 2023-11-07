# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackQuantizedPartitioner,
)
from executorch.backends.xnnpack.test.tester import Partition, Tester


class TestAdd(unittest.TestCase):
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x + y
            z = z + x
            z = z + x
            z = z + z
            return z

    def test_fp32_add(self):
        inputs = (torch.ones(1), torch.ones(1))
        (
            Tester(self.Add(), inputs)
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 4})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_qs8_add(self):
        inputs = (torch.ones(1), torch.ones(1))
        (
            Tester(self.Add(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 4})
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner))
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    class AddRelu(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            return torch.nn.functional.relu(z)

    def test_fp32_add_relu(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.AddRelu(), inputs)
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_count({"torch.ops.aten.relu.default": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 1})
            .check_count({"executorch_exir_dialects_edge__ops_aten_relu_default": 1})
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_qs8_add_relu(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.AddRelu(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_count({"torch.ops.aten.relu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 1})
            .check_count({"executorch_exir_dialects_edge__ops_aten_relu_default": 1})
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner))
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
