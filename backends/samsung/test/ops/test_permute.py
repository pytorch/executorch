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


class Permute(torch.nn.Module):
    def __init__(self, order=None) -> None:
        super().__init__()
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.order)


class TestPermute(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module, inputs,
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
                .check_count({"torch.ops.aten.permute.default": 1})
                .to_edge_transform_and_lower()
                .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
        )

    def test_fp32_permute_0231(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(Permute(order=[0, 2, 3, 1]), inputs)

    def test_fp32_permute_0312(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(Permute(order=[0, 3, 1, 2]), inputs)

    def test_fp32_permute_0321(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(Permute(order=[0, 3, 2, 1]), inputs)
