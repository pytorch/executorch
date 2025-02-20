# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test that tosa_supported_operators reject operators that are not
# quantized properly. This is typically a consequence of a torch op
# such a Softplus that is decompsed into many other ops without
# surrounding q/dq nodes.

from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t1 = Tuple[torch.Tensor]
aten_op: list[str] = ["torch.ops.aten.add.Tensor", "torch.ops.aten.softplus.default"]
exir_op: list[str] = [
    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
    "executorch_exir_dialects_edge__ops_aten_exp_default",
    "executorch_exir_dialects_edge__ops_aten_div_Tensor",
]


test_data: dict[input_t1] = {
    "3d_rand": (torch.rand(1, 5, 5),),
}


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor):
        return self.softplus(x + x)


@common.parametrize("test_data", test_data)
def test_softplus_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](
        Module(), test_data=test_data, aten_op=aten_op, exir_op=exir_op
    )
    # remove check_count.exir as there will be more than one delegate
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_softplus_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](
        Module(), test_data=test_data, aten_op=aten_op, exir_op=exir_op
    )
    pipeline.pop_stage("check_not.exir")
    # check that all ops in exir_op except add are rejected
    pipeline.add_stage_after(
        "partition", pipeline.tester.check, exir_op[1:], suffix="exir_post_partition"
    )
    pipeline.run()
