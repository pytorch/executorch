# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details..

import unittest

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig


class GroupNorm(torch.nn.Module):
    def __init__(self, groups, in_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.module = torch.nn.GroupNorm(groups, self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TestGroupNorm(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.group_norm.default": 1})
            .to_edge_transform_and_lower()
            .check_not(
                ["executorch_exir_dialects_edge__ops_aten_native_group_norm_default"]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs()
        )

    def test_fp32_group_norm(self):
        groups = 3
        in_channels = 12
        inputs = (torch.randn(1, in_channels, 8, 8),)
        self._test(GroupNorm(groups, in_channels), inputs)
