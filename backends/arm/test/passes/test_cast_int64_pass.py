# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import CastInt64BuffersToInt32Pass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

from executorch.backends.test.harness.stages import StageType

input_t = Tuple[torch.Tensor]  # Input x


class Int64Model(torch.nn.Module):
    test_data = {
        "rand": (torch.rand(4),),
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 3


@common.parametrize("test_data", Int64Model.test_data)
def test_int64_model(test_data: input_t):
    module = Int64Model()
    op_checks = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass=op_checks,
        ops_after_pass=op_checks,
        passes_with_exported_program=[CastInt64BuffersToInt32Pass],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    for state in exported_program.state_dict:
        assert exported_program.state_dict[state].dtype == torch.int32
