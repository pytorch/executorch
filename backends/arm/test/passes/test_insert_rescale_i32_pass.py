# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    InsertRescaleInt32Pass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline


class NeedsRescaleOps(torch.nn.Module):
    """A module containing ops that require INT32 inputs/outputs."""

    input_t = Tuple[torch.Tensor, torch.Tensor]

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        a = x > y
        return a

    def get_inputs(self, dtype) -> input_t:
        if dtype == torch.float32:
            return (torch.rand(1, 3, 5, 6), torch.rand(1, 3, 5, 6))
        elif dtype == torch.int32:
            return (
                torch.randint(3, 5, (3,), dtype=torch.int32),
                torch.randint(3, 5, (3,), dtype=torch.int32),
            )
        else:
            raise ValueError("Not a valid input dtype for model")


def test_insert_rescales():
    module = NeedsRescaleOps()
    input_t = Tuple[torch.Tensor, torch.Tensor]
    ops_not_before = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    ops_after = {
        # "number of op nodes with i8 output" + "number of i8 node inputs"
        "executorch_exir_dialects_backend__ops_tosa_RESCALE_default": 0
        + 2,
    }
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(torch.float32),
        quantize=True,
        ops_not_before_pass=ops_not_before,
        ops_after_pass=ops_after,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_dont_insert_rescales():
    module = NeedsRescaleOps()
    input_t = Tuple[torch.Tensor, torch.Tensor]
    ops_not_before = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    # All inputs are already i32. Rescales should not be added.
    ops_not_after = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(torch.int32),
        ops_not_before_pass=ops_not_before,
        ops_not_after_pass=ops_not_after,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
