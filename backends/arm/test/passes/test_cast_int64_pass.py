# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.cast_int64_pass import CastInt64ToInt32Pass

from executorch.backends.arm.test.tester.test_pipeline import TestPassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Int64Model(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return x + 3

    def get_inputs(self) -> input_t:
        return (torch.rand(4),)


def test_int64_model_tosa_BI():
    module = Int64Model()
    op_checks = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
    }
    pipeline = TestPassPipeline[input_t](
        module,
        module.get_inputs(),
        tosa_version="TOSA-0.80+BI",
        ops_before_pass=op_checks,
        ops_after_pass=op_checks,
        passes_with_exported_program=[CastInt64ToInt32Pass],
    )
    pipeline.pop_stage("quantize")
    pipeline.run()

    exported_program = pipeline.tester.get_artifact("RunPasses").exported_program()
    for state in exported_program.state_dict:
        assert exported_program.state_dict[state].dtype == torch.int32
