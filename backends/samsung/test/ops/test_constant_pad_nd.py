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


class ConstantPadND(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 0, 1, 1)
        return torch.nn.functional.pad(x, pad, mode="constant", value=0)


class TestConstantPadND(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module, inputs, [gen_samsung_backend_compile_spec("E9955")]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_constant_pad_nd(self):
        inputs = (torch.randn(1, 6, 8, 16),)
        self._test(ConstantPadND(), inputs)
