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


class Clamp(torch.nn.Module):
    def __init__(
        self,
        minimum=0.0,
        maximum=0.0,
    ) -> None:
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.minimum, self.maximum)


class TestClamp(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module, inputs, [gen_samsung_backend_compile_spec("E9955")]
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.clamp.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_clamp(self):
        inputs = (torch.randn(1, 16, 8, 8),)
        self._test(Clamp(minimum=0, maximum=2.0), inputs)
