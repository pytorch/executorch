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


class Index(torch.nn.Module):
    def __init__(self, indices: tuple[torch.Tensor]) -> None:
        super().__init__()
        self.indices = indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.indices]


class TestIndex(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.index.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_index_Tensor"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs()
        )

    def test_fp32_index_on_axis0(self):
        indices = torch.tensor([2, 0], dtype=torch.int32)
        inputs = (torch.randn(4, 16, 8, 8),)
        self._test(Index(indices), inputs)

    def test_fp32_index_on_axis1(self):
        indices = (slice(None), torch.tensor([0, 3, 2, 11, 8], dtype=torch.int32))
        inputs = (torch.randn(4, 16, 8, 8),)
        self._test(Index(indices), inputs)

    def test_fp32_index_on_axis2(self):
        indices = (
            slice(None),
            slice(None),
            torch.tensor([1, 2, 5, 6], dtype=torch.int32),
        )
        inputs = (torch.randn(4, 16, 8, 8),)
        self._test(Index(indices), inputs)

    def test_fp32_index_on_axis3(self):
        indices = (
            slice(None),
            slice(None),
            slice(None),
            torch.tensor([0, 3, 6, 4], dtype=torch.int32),
        )
        inputs = (torch.randn(4, 16, 8, 8),)
        self._test(Index(indices), inputs)
