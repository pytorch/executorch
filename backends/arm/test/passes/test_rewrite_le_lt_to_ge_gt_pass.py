# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.rewrite_le_lt_to_ge_gt_pass import (
    RewriteLeLtToGeGtPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor, torch.Tensor]


class LtLe(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.randn(4, 4), torch.randn(4, 4))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (x < y, x <= y)


@common.parametrize("module", {"lt_le": LtLe()})
def test_rewrite_le_lt_to_ge_gt_no_target(module: LtLe) -> None:
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_lt_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_le_Tensor": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_gt_Tensor",
            "executorch_exir_dialects_edge__ops_aten_ge_Tensor",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_gt_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_ge_Tensor": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_lt_Tensor",
            "executorch_exir_dialects_edge__ops_aten_le_Tensor",
        ],
        pass_list=[RewriteLeLtToGeGtPass],
    )
    pipeline.run()
