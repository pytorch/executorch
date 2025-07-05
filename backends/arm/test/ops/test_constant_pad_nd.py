# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Test the pad_constant_nd op which pads the input tensor at specific dimension(s).
#
from typing import Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.pad.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_pad_default"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    "4dim_last1dim": lambda: (torch.rand(1, 1, 16, 16), (1, 1, 0, 0, 0, 0, 0, 0), 1),
    "4dim_last2dim": lambda: (torch.rand(1, 1, 16, 16), (1, 0, 1, 0, 0, 0, 0, 0), 2),
    "4dim_last3dim": lambda: (torch.rand(1, 1, 16, 16), (1, 1, 0, 2, 0, 2, 0, 0), 3),
    "4dim_last4dim": lambda: (torch.rand(1, 1, 16, 16), (1, 0, 1, 1, 0, 2, 0, 2), 4),
    "3dim_last1dim": lambda: (torch.rand(1, 1, 16), (1, 1, 0, 0, 0, 0), 1),
    "3dim_last2dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 1, 0, 0), 2),
    "3dim_last3dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 0, 1, 1), 3),
    "2dim_last1dim": lambda: (torch.rand(1, 1, 16), (1, 1, 0, 0), 1),
    "2dim_last2dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 1), 2),
}


class ConstantPadND(torch.nn.Module):
    def __init__(self, pad: Tuple, value: float | None = None):
        super().__init__()
        self.value = value
        nonzero_idx = len(pad)
        for i in range(0, len(pad), 2):
            if pad[i] + pad[i + 1] == 0:
                nonzero_idx = i
                break
        self.pad = pad[:nonzero_idx]

    def forward(self, x: torch.Tensor):
        x = F.pad(x, pad=self.pad, mode="constant", value=self.value)
        return x


@common.parametrize(
    "test_data",
    test_data_suite,
)
def test_constant_pad_nd_tosa_MI(test_data: Tuple):
    test_data, padding, value = test_data()
    pipeline = TosaPipelineMI[input_t1](
        ConstantPadND(padding, value),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_constant_pad_nd_tosa_BI(test_data: Tuple):
    test_data, padding, value = test_data()
    pipeline = TosaPipelineBI[input_t1](
        ConstantPadND(padding, value),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()
