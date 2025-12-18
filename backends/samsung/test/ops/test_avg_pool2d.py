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


class AvgPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size=2,
        stride=1,
        padding=0,
    ) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=False,
            ceil_mode=False,
        ).to(torch.float)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool(x)


class TestAvgPool2d(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_avg_pool2d(self):
        inputs = (torch.randn(1, 16, 24, 24),)
        self._test(AvgPool2d(), inputs)

    def test_fp32_avg_pool2d_with_stride(self):
        inputs = (torch.randn(1, 16, 24, 24),)
        self._test(AvgPool2d(stride=2), inputs)

    def test_fp32_avg_pool2d_with_kernel_size(self):
        inputs = (torch.randn(1, 16, 24, 24),)
        self._test(AvgPool2d(kernel_size=4), inputs)
