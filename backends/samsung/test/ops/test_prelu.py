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


class PReLU(torch.nn.Module):
    def __init__(self, input_channel=3, per_channel=False) -> None:
        super().__init__()
        self.module = torch.nn.PReLU(input_channel if per_channel else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TestPReLU(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.prelu.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_prelu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_prelu(self):
        inputs = (torch.randn(1, 3, 56, 56),)
        self._test(PReLU(per_channel=False), inputs)

    def test_fp32_prelu_per_channel(self):
        inputs = (torch.randn(1, 3, 56, 56),)
        self._test(PReLU(input_channel=3, per_channel=True), inputs)
