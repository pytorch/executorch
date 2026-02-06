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


class UpsampleBilinear2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = [32, 32]
        return torch.nn.functional.interpolate(
            x,
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )


class TestUpsampleBilinear2d(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.upsample_bilinear2d.vec": 1})
            .to_edge_transform_and_lower()
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_upsample_bilinear2d_vec"]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_upsample_bilinear2d(self):
        inputs = (torch.randn(1, 16, 16, 16),)
        self._test(UpsampleBilinear2d(), inputs)
