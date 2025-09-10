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


class BatchMatmul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        input_1 = torch.randn(2, 16, 56)
        input_2 = torch.randn(2, 56, 32)
        return (input_1, input_2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, y)


class TestBatchMatmul(unittest.TestCase):
    def _test(self, module: torch.nn.Module):
        inputs = module.get_example_inputs()
        tester = SamsungTester(
            module, inputs, [gen_samsung_backend_compile_spec("E9955")]
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.bmm.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_bmm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @unittest.skip("Temporarily disable it because TOT codes not well prepared.")
    def test_fp32_bmm(self):
        self._test(BatchMatmul())
