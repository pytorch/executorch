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


class BatchNorm(torch.nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm2d(num_features=self.num_features)
        self.bn.weight.data.uniform_(-0.1, 0.1)
        self.bn.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class TestBatchNorm(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module, inputs, [gen_samsung_backend_compile_spec("E9955")]
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def test_fp32_batch_norm(self):
        num_features = 16
        inputs = (torch.randn(4, num_features, 32, 32),)
        self._test(BatchNorm(num_features), inputs)
