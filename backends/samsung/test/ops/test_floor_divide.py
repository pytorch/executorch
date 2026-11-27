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


class FloorDivide(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(x, y)


class TestFloorDivide(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.floor_divide.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_floor_divide_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_floor_divide(self):
        inputs = (
            torch.randint(1, 100, (1, 3, 8, 8)),
            torch.randint(1, 10, (1, 3, 8, 8)),
        )
        self._test(FloorDivide(), inputs)

    def test_fp32_floor_divide_broadcast(self):
        inputs = (
            torch.randint(1, 100, (1, 3, 8, 8)),
            torch.randint(1, 10, (1, 1, 8, 8)),
        )
        self._test(FloorDivide(), inputs)
