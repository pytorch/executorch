# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_int32_clamp_pass import (
    DecomposeInt32ClampPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class ClampInt32(torch.nn.Module):
    test_data = {"rand": (torch.randint(-50, 50, (2, 3), dtype=torch.int32),)}

    def forward(self, x: torch.Tensor):
        return torch.clamp(x, -10, 5)


@common.parametrize("test_data", ClampInt32.test_data)
def test_decompose_int32_clamp_pass(test_data: input_t):
    module = ClampInt32()
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
            "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_clamp_default",
        ],
        pass_list=[DecomposeInt32ClampPass],
    )
    pipeline.run()
