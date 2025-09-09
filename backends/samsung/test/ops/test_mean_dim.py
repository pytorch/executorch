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



class MeanDim(torch.nn.Module):
    def __init__(self, keep_dims=True) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[2, 3], keepdim=self.keep_dims)


class TestMeanDim(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module, inputs,
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
                .check_count({"torch.ops.aten.mean.dim": 1})
                .to_edge_transform_and_lower()
                .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .to_executorch()
        )

    def test_fp32_mean_with_keep_dims(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MeanDim(), inputs)

    def test_fp32_mean_without_keep_dims(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MeanDim(keep_dims=False), inputs)