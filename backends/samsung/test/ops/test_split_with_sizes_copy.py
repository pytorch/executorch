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


class Split(torch.nn.Module):
    def __init__(self, split_sizes=1, dim=0) -> None:
        super().__init__()
        self.split_sizes = split_sizes
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.split(x, self.split_sizes, dim=self.dim)


class TestSplit(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )

        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_split_with_sizes_default"]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs()
        )

    def test_fp32_split_default(self):
        inputs = (torch.randn(6, 6),)
        self._test(Split(3), inputs)

    def test_fp32_split_chunk(self):
        inputs = (torch.randn(6, 6),)
        self._test(Split([3, 3]), inputs)

    def test_fp32_split_dim1(self):
        inputs = (torch.randn(6, 6),)
        self._test(Split([3, 2, 1], dim=1), inputs)
