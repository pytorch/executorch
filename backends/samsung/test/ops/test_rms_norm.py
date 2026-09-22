# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import unittest
from typing import List

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig


class RMSNorm(torch.nn.Module):
    def __init__(self, normalize_shape: List[int]) -> None:
        super().__init__()
        self.normalize_shape = normalize_shape
        self.module = torch.nn.RMSNorm(self.normalize_shape, eps=1e-5)
        self.module.weight = torch.nn.Parameter(torch.ones(self.normalize_shape) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TestRMSNorm(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )

        (
            tester.export()
            .check_count({"torch.ops.aten.rms_norm.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_rms_norm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs()
        )

    def test_fp32_rms_norm(self):
        normalize_shape = [196]
        inputs = (torch.randn(1, *normalize_shape),)
        self._test(RMSNorm(normalize_shape), inputs)
