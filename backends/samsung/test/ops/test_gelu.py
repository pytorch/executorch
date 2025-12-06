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


class GELU(torch.nn.Module):
    def __init__(self, with_conv=False) -> None:
        super().__init__()
        self.module = (
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=3,
                    stride=(2, 2),
                    padding=(1, 1),
                    dilation=(1, 1),
                ).to(torch.float),
                torch.nn.GELU(),
            )
            if with_conv
            else torch.nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TestGELU(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.gelu.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_gelu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(atol=0.002, rtol=0.002)
        )

    def test_fp32_single_gelu(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(GELU(with_conv=False), inputs)

    def test_fp32_conv_gelu(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(GELU(with_conv=True), inputs)
