# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t1 = Tuple[torch.Tensor]

aten_op = "torch.ops.aten.detach_copy.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__detach_copy_default"

test_data_suite = {
    "zeros_2d": torch.zeros(3, 5),
    "ones_3d": torch.ones(2, 3, 4),
    "rand_2d": torch.rand(10, 10) - 0.5,
    "ramp_1d": torch.arange(-8.0, 8.0, 0.5),
}


class DetachCopy(torch.nn.Module):
    aten_op = aten_op
    exir_op = exir_op

    def forward(self, x: torch.Tensor):
        return torch.detach_copy(x)


@common.parametrize("test_data", test_data_suite)
def test_detach_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        DetachCopy(),
        (test_data,),
        aten_op=DetachCopy.aten_op,
        exir_op=DetachCopy.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_detach_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        DetachCopy(),
        (test_data,),
        aten_op=DetachCopy.aten_op,
        exir_op=DetachCopy.exir_op,
    )
    pipeline.run()
