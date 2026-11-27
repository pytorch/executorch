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


class MaxDim(torch.nn.Module):
    def __init__(self, dim: int, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=self.dim, keepdim=self.keepdim)[0]


class TestMaxDim(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.max.dim": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_max_dim"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_max_dim_last_with_keepdim(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MaxDim(dim=-1, keepdim=True), inputs)

    def test_fp32_max_dim_last_without_keepdim(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MaxDim(dim=-1, keepdim=False), inputs)

    def test_fp32_max_dim_middle(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MaxDim(dim=1, keepdim=False), inputs)

    def test_fp32_max_dim_first(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MaxDim(dim=0, keepdim=False), inputs)
