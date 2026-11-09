# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import unittest

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig


class SumDimIntList(torch.nn.Module):
    def __init__(self, keep_dims=True) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, [2, 3], keepdim=self.keep_dims)


class TestSumDimIntList(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.sum.dim_IntList": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_sum_dim_IntList"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(atol=0.005, rtol=0.005)
        )

    def test_fp32_sum_dim_with_keep_dims(self):
        inputs = (torch.randn(1, 16, 8, 8),)
        self._test(SumDimIntList(), inputs)

    def test_fp32_sum_dim_without_keep_dims(self):
        inputs = (torch.randn(1, 16, 8, 8),)
        self._test(SumDimIntList(keep_dims=False), inputs)
