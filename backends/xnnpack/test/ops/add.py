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


class TestXNNPACKAdd(unittest.TestCase):
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x + y
            z = z + x
            z = z + x
            z = z + z
            return z

    def test_add(self):
        """
        This test is the simplest test by manually lowering some submodules, we can use paritioner for auto detecting lowerable parts
        """
        add_module = self.AddModule()
        model_inputs = (torch.ones(1), torch.ones(1))

        (
            Tester(add_module, model_inputs)
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 4})
            .partition()
            .check_count({"torch.ops.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_add_quantized(self):
        add_module = self.AddModule()
        model_inputs = (torch.ones(1), torch.ones(1))

        (
            Tester(add_module, model_inputs)
            .quantize()
            .check(["torch.ops.quantized_decomposed"])
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 4})
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner2))
            .check_count({"torch.ops.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_add_quantized_pt2e(self):
        add_module = self.AddModule()
        model_inputs = (torch.ones(1), torch.ones(1))

        (
            Tester(add_module, model_inputs)
            .quantize2()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_add_Tensor": 4})
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner2))
            .check_count({"torch.ops.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
