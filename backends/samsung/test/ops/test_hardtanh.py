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


class Hardtanh(torch.nn.Module):
    def __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.module = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TestHardtanh(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.hardtanh.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_hardtanh(self):
        inputs = (torch.randn(1, 3, 16, 16),)
        self._test(Hardtanh(min_val=0, max_val=1), inputs)
