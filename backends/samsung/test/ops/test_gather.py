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
from executorch.backends.samsung.test.utils.utils import TestConfig


class Gather(torch.nn.Module):
    def __init__(self, dim: int, index: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, dim=self.dim, index=self.index)


class GatherExpandedIndex(torch.nn.Module):
    def __init__(self, dim: int, index_size: torch.Size) -> None:
        super().__init__()
        self.dim = dim
        self.index_size = index_size

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, dim=self.dim, index=idx.expand(self.index_size))


class TestGather(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.gather.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_gather_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_gather_1d(self):
        index = torch.randint(0, 16, (8,), dtype=torch.int32)
        inputs = (torch.randn(16),)
        self._test(Gather(dim=0, index=index), inputs)

    def test_fp32_gather_dim1(self):
        idx = torch.randint(0, 8400, (1, 300, 1), dtype=torch.int64)
        inputs = (torch.randn(1, 8400, 80), idx)
        self._test(GatherExpandedIndex(dim=1, index_size=(1, 300, 80)), inputs)
