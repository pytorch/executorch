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



class SelectCopy(torch.nn.Module):
    def __init__(self, axis, index) -> None:
        super().__init__()
        self.axis = axis
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.select(x, self.axis, self.index)


class TestSelectCopy(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module, inputs,
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
                .check_count({"torch.ops.aten.select.int": 1})
                .to_edge_transform_and_lower()
                .check_not(["executorch_exir_dialects_edge__ops_aten_select_copy_int"])
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
        )

    def test_fp32_select_on_axis1(self):
        inputs = (torch.randn([1, 4, 16, 16]),)
        self._test(SelectCopy(axis=1, index=0), inputs)

    def test_fp32_concat_on_axis3(self):
        inputs = (torch.randn([1, 4, 16, 16]),)
        self._test(SelectCopy(axis=3, index=6), inputs)