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


class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size=2,
        stride=1,
        padding=0,
        dilation=1,
    ) -> None:
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=False,
            ceil_mode=False,
        ).to(torch.float)

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        input_1 = torch.randn(1, 16, 24, 24)
        return (input_1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool(x)


class TestMaxPool2d(unittest.TestCase):
    def _test(self, module: torch.nn.Module):
        tester = SamsungTester(
            module,
            module.get_example_inputs(),
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.max_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_max_pool2d_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def test_fp32_max_pool2d(self):
        self._test(MaxPool2d())

    def test_fp32_max_pool2d_with_padding(self):
        self._test(MaxPool2d(padding=1))

    def test_fp32_max_pool2d_with_kernel_size(self):
        self._test(MaxPool2d(kernel_size=4))

    def test_fp32_max_pool2d_with_dilation(self):
        self._test(MaxPool2d(dilation=2))
