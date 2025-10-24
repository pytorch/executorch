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


class MultipleOpsModel(torch.nn.Module):
    """A module containing ops that require INT32 inputs/outputs."""

    input_t = Tuple[torch.Tensor, torch.Tensor]

    def forward(self, x, y):
        a = x * y
        b = torch.maximum(a, y)
        c = torch.abs(b)
        d = c > b
        return d

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

    def get_num_expected_rescales(self):
        # "number of op nodes with i8 output" + "number of i8 node inputs"
        return 3 + 7


class SumModel(torch.nn.Module):
    input_t = Tuple[torch.Tensor]

    def forward(self, x):
        a = torch.sum(x, 2, keepdim=True)  # (1, 2, 1, 4)
        b = torch.sum(a, [1, 3], keepdim=True)  # (1, 1, 1, 1)
        c = torch.sum(b, [0, 2], keepdim=False)  # (1, 1)
        return c

    def get_inputs(self, dtype) -> input_t:
        if dtype == torch.float32:
            return (torch.rand(1, 2, 3, 4),)
        elif dtype == torch.int32:
            return (torch.randint(0, 10, (1, 2, 3, 4), dtype=torch.int32),)
        else:
            raise ValueError("Not a valid input dtype for model")

    def get_num_expected_rescales(self):
        # Two RESCALE nodes per SUM node
        return 6


def _test_model_with_f32_data(model):
    ops_not_before = {"executorch_exir_dialects_backend__ops_tosa_RESCALE_default"}
    ops_after = {
        "executorch_exir_dialects_backend__ops_tosa_RESCALE_default": model.get_num_expected_rescales(),
    }
    pipeline = PassPipeline[model.input_t](
        model,
        model.get_inputs(torch.float32),
        quantize=True,
        ops_not_before_pass=ops_not_before,
        ops_after_pass=ops_after,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_insert_rescales_sum_model():
    _test_model_with_f32_data(SumModel())


def test_insert_rescales_multiple_ops_model():
    _test_model_with_f32_data(MultipleOpsModel())


def test_dont_insert_rescales():
    module = MultipleOpsModel()
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
